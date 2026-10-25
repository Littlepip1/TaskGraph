"""Dynamic receiver selection with an explicit event-value omission constraint.

New development algorithm, NOT reconstructed historical dhg_set source.
No fixed recipient cardinality and no receiver-similarity redundancy penalty.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import math
import numpy as np
from airsim_ext.controller import clipped_velocity


@dataclass(frozen=True)
class DynamicConfig:
    # Dimensionless fraction of the CURRENT event's modeled value, not epsilon_J.
    omission_fraction: float = 0.10
    action_weight: float = 1.0
    uncertainty_weight: float = 1.0
    timeliness_discount: bool = True
    numerical_tolerance: float = 1e-12

    def validate(self):
        if not 0 <= self.omission_fraction <= 1 or not math.isfinite(self.omission_fraction):
            raise ValueError('omission_fraction must be in [0,1]')
        if any(not math.isfinite(x) or x<0 for x in [self.action_weight,self.uncertainty_weight]):
            raise ValueError('Nonnegative finite value weights required')
        if self.numerical_tolerance<=0 or not math.isfinite(self.numerical_tolerance):
            raise ValueError('Invalid tolerance')


@dataclass(frozen=True)
class TaskFactor:
    # A bonus only for serving ALL the listed control participants.
    # Must be justified by the task, never inferred from belief similarity.
    receivers: tuple[int,...]
    value: float
    name: str = ''


def construct_set(values, route_edges, edge_costs, *, cfg=DynamicConfig(), factors=()):
    cfg.validate()
    candidates=tuple(sorted(values))
    if set(route_edges)!=set(candidates):raise ValueError('Route/value candidates disagree')
    if any(not math.isfinite(v) or v<0 for v in values.values()):raise ValueError('Invalid receiver value')
    for r in candidates:
        if not route_edges[r]:raise ValueError('Each remote receiver needs a nonempty current route')
        for edge in route_edges[r]:
            if edge not in edge_costs or not math.isfinite(edge_costs[edge]) or edge_costs[edge]<=0:
                raise ValueError('Route edges require finite strictly positive service cost')
    # Only factors wholly present in the reachable/task-relevant universe can be
    # evaluated. Unsupported factors must be reported, not silently certified.
    valid=[];unavailable=[]
    universe=set(candidates)
    for f in factors:
        if len(set(f.receivers))!=len(f.receivers) or len(f.receivers)<2 or not math.isfinite(f.value) or f.value<0:
            raise ValueError('Invalid task factor')
        if set(f.receivers)<=universe:valid.append(f)
        else:unavailable.append(f.name or str(f.receivers))
    def value(selected):
        return sum(values[r] for r in selected)+sum(f.value for f in valid if set(f.receivers)<=selected)
    total=value(universe);budget=cfg.omission_fraction*total
    selected=set();covered=set();captured=0.;trace=[]
    # Finite because each accepted block adds at least one new receiver.
    while total-captured>budget+cfg.numerical_tolerance:
        options={frozenset([r]) for r in candidates if r not in selected}
        # A block move can realize complementarity even when singleton gains are 0.
        options.update(frozenset(set(f.receivers)-selected) for f in valid if not set(f.receivers)<=selected)
        best=None
        for block in sorted(options,key=lambda b:tuple(sorted(b))):
            new=set(block);new_value=value(selected|new);gain=new_value-captured
            if gain<=cfg.numerical_tolerance:continue
            edges=set().union(*(route_edges[r] for r in block))-covered
            cost=sum(edge_costs[e] for e in edges)
            # Already-covered route edges entail zero incremental snapshot service.
            ratio=math.inf if cost<=cfg.numerical_tolerance else gain/cost
            key=(ratio,gain,-cost,tuple(-r for r in sorted(block)))
            if best is None or key>best[0]:best=(key,new,new_value,edges,cost,gain)
        if best is None:break
        _,new,captured,edges,cost,gain=best
        selected|=new;covered|=edges
        trace.append(dict(added=sorted(new),marginal_value=float(gain),incremental_bit_hop=float(cost),
                          captured_value=float(captured),omitted_value=float(max(0,total-captured))))
    residual=max(0.,total-captured)
    return dict(selected_recipients=sorted(selected),candidate_count=len(candidates),
                total_value=float(total),captured_value=float(captured),omitted_value=float(residual),
                omission_budget=float(budget),surrogate_feasible=bool(residual<=budget+cfg.numerical_tolerance),
                snapshot_bit_hop=float(sum(edge_costs[e] for e in covered)),
                selection_trace=trace,unavailable_task_factors=unavailable,
                stop_reason='event_value_budget_met' if residual<=budget+cfg.numerical_tolerance else 'no_positive_block',
                no_fixed_recipient_cap=True)


def select_dynamic(source,reachable,positions,beliefs,observation,offsets,controller_cfg,network,dt,
                   deadline_ticks,*,cfg=DynamicConfig(),factors=()):
    """Benchmark receiver context is explicit; acquisition cost is NOT modeled.

    No target truth, future routes, future disturbances or future rollouts are read.
    Hypothetical fusion is on copied current receiver beliefs, not live estimators.
    """
    cfg.validate()
    if observation is None:
        result=construct_set({}, {}, {},cfg=cfg,factors=factors)
        result.update(scope_kind='dynamic_task_value',scope_fanout_cap=None,
                      context_mode='benchmark_current_receiver_beliefs_acquisition_unaccounted')
        return result
    values={};routes={};edge_costs={};components={}
    n=len(beliefs);bits=int(network.cfg['payload_bits']);cap=max(1,int(network.cfg['bits_per_link_tick']))
    prop=max(0,int(network.cfg.get('prop_delay_ticks',1)))
    scale=max(float(controller_cfg['reward_scale_m']),1e-9)
    for r in sorted(set(int(x) for x in reachable if int(x)!=int(source))):
        path=network.path(int(source),r)
        if path is None:continue
        edges={network._edge_key(a,b) for a,b in zip(path,path[1:])}
        routes[r]=edges
        delay=0.
        for edge in edges:
            edge_costs[edge]=float(bits)
            delay+=float(network._queue_bits(edge))/cap+bits/cap+prop
        prior=beliefs[r];posterior=prior.copy()
        posterior.update_position(observation.z_xy,observation.r_xy)
        old_action=clipped_velocity(positions[r],prior.mean[:2]+offsets[r],controller_cfg['kp'],controller_cfg['max_speed_mps'])
        new_action=clipped_velocity(positions[r],posterior.mean[:2]+offsets[r],controller_cfg['kp'],controller_cfg['max_speed_mps'])
        action=float(dt*np.linalg.norm(new_action-old_action)/scale)
        covariance=float(math.sqrt(max(0.,np.trace(prior.cov[:2,:2]-posterior.cov[:2,:2])))/scale)
        discount=math.exp(-delay/max(1,int(deadline_ticks))) if cfg.timeliness_discount else 1.
        values[r]=float((cfg.action_weight*action+cfg.uncertainty_weight*covariance)*discount/max(1,n))
        components[r]=dict(action_change=action,covariance_update=covariance,timeliness=discount)
    result=construct_set(values,routes,edge_costs,cfg=cfg,factors=factors)
    result.update(scope_kind='dynamic_task_value',scope_fanout_cap=None,
                  dynamic_config=asdict(cfg),receiver_value_components=components,
                  context_mode='benchmark_current_receiver_beliefs_acquisition_unaccounted',
                  task_factor_count=len(factors))
    return result
