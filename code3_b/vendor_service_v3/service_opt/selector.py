"""Service-efficient receiver sets: squared action impact and branch deletion.

All values are nonnegative importance proxies, NOT certified return losses.
No fixed K, no target truth, no similarity-substitution of independent receivers.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import math
import numpy as np
from airsim_ext.controller import clipped_velocity
from dynamic_set import DynamicConfig,construct_set


@dataclass(frozen=True)
class ServiceConfig:
    omission_fraction: float = .25
    absolute_event_budget: float = 1e-5
    # Receiver uncertainty safeguard activates only above the existing 6m scale.
    uncertainty_guard_m: float = 6.
    uncertainty_weight: float = 1.
    use_backward_pruning: bool = True
    numerical_tolerance: float = 1e-12

    def validate(self):
        if not math.isfinite(self.omission_fraction) or not 0<=self.omission_fraction<=1:
            raise ValueError('Invalid omission fraction')
        for x in [self.absolute_event_budget,self.uncertainty_weight]:
            if not math.isfinite(x) or x<0:raise ValueError('Invalid nonnegative parameter')
        if not math.isfinite(self.uncertainty_guard_m) or self.uncertainty_guard_m<=0:raise ValueError('Invalid uncertainty guard')
        if not math.isfinite(self.numerical_tolerance) or self.numerical_tolerance<=0:raise ValueError('Invalid tolerance')


def _prune_reference(values,route_edges,edge_costs,budget,factors=(),tol=1e-12):
    """Backward removal of singletons and complete physical route branches."""
    selected=set(values);factor_sets=[(set(f.receivers),f.value) for f in factors if set(f.receivers)<=selected]
    users={}
    for r,edges in route_edges.items():
        for e in edges:users.setdefault(e,set()).add(r)
    total=sum(values.values())+sum(w for _,w in factor_sets)
    omitted=0.;trace=[]
    while selected:
        options={frozenset([r]) for r in selected}
        options.update(frozenset(s & selected) for s in users.values() if s & selected)
        best=None
        for block in sorted(options,key=lambda b:tuple(sorted(b))):
            lost=sum(values[r] for r in block)+sum(w for members,w in factor_sets if members<=selected and members & block)
            if omitted+lost>budget+tol:continue
            freed={e for e,s in users.items() if s & selected and (s & selected)<=block}
            saving=sum(edge_costs[e] for e in freed)
            if saving<=tol:continue
            key=(math.inf if lost<=tol else saving/lost,saving,-lost,tuple(-r for r in sorted(block)))
            if best is None or key>best[0]:best=(key,block,lost,saving)
        if best is None:break
        _,block,lost,saving=best;selected-=block;omitted+=lost
        trace.append(dict(removed=sorted(block),lost_value=float(lost),saved_snapshot_bit_hop=float(saving),omitted_value=float(omitted)))
    edges=set().union(*(route_edges[r] for r in selected)) if selected else set()
    return dict(selected_recipients=sorted(selected),snapshot_bit_hop=float(sum(edge_costs[e] for e in edges)),
                omitted_value=float(omitted),total_value=float(total),captured_value=float(total-omitted),
                omission_budget=float(budget),surrogate_feasible=bool(omitted<=budget+tol),pruning_trace=trace)


def prune_set(values,route_edges,edge_costs,budget,factors=(),tol=1e-12):
    """Vectorized exact branch scoring for up to64 candidates; general fallback.

    64 is a bitset storage boundary, NOT a selected-recipient cap. Larger
    candidate universes use the reference implementation without truncation.
    """
    ids=tuple(sorted(values));n=len(ids)
    if n>64 or n==0:return _prune_reference(values,route_edges,edge_costs,budget,factors,tol)
    ix={r:i for i,r in enumerate(ids)}
    def mask(members):return sum(1<<ix[r] for r in members)
    def members(b):return tuple(ids[i] for i in range(n) if (b>>i)&1)
    edge_list=tuple(edge_costs)
    edge_users=np.array([mask(r for r in ids if e in route_edges[r]) for e in edge_list],dtype=np.uint64)
    weights=np.array([edge_costs[e] for e in edge_list],float)
    v=np.array([values[r] for r in ids],float)
    ff=[(mask(f.receivers),f.value) for f in factors if set(f.receivers)<=set(ids)]
    total=float(sum(values.values())+sum(w for _,w in ff));sel=(1<<n)-1;omitted=0.;trace=[]
    shifts=np.arange(n,dtype=np.uint64)
    while sel:
        active=edge_users & np.uint64(sel)
        options=set(int(b) for b in active if b)
        options.update(1<<i for i in range(n) if (sel>>i)&1)
        blocks=sorted(options,key=members)
        b=np.array(blocks,dtype=np.uint64)
        incidence=((b[:,None]>>shifts)&np.uint64(1)).astype(float)
        lost=incidence@v
        for f,w in ff:
            if f&sel==f:lost+=((b & np.uint64(f))!=0)*w
        freed=((active[None,:]!=0)&((active[None,:]&np.bitwise_not(b[:,None]))==0))
        saving=freed@weights
        valid=np.flatnonzero((omitted+lost<=budget+tol)&(saving>tol))
        if not len(valid):break
        best=max(valid,key=lambda k:(math.inf if lost[k]<=tol else saving[k]/lost[k],saving[k],-lost[k],tuple(-r for r in members(blocks[k]))))
        block=blocks[best];sel &= ~block;omitted+=float(lost[best])
        trace.append(dict(removed=list(members(block)),lost_value=float(lost[best]),saved_snapshot_bit_hop=float(saving[best]),omitted_value=float(omitted)))
    active=edge_users & np.uint64(sel)
    return dict(selected_recipients=list(members(sel)),snapshot_bit_hop=float(weights[active!=0].sum()),
                omitted_value=float(omitted),total_value=total,captured_value=float(total-omitted),
                omission_budget=float(budget),surrogate_feasible=bool(omitted<=budget+tol),pruning_trace=trace)


def select_service_optimized(source,reachable,positions,beliefs,observation,offsets,controller_cfg,network,dt,
                             deadline_ticks,*,cfg=ServiceConfig(),factors=()):
    cfg.validate();values={};routes={};costs={};components={}
    scale=max(float(controller_cfg['reward_scale_m']),1e-9);n=len(beliefs)
    bits=int(network.cfg['payload_bits'])
    if observation is not None:
        for r in sorted(set(int(x) for x in reachable if int(x)!=int(source))):
            path=network.path(int(source),r)
            if path is None:continue
            routes[r]={network._edge_key(a,b) for a,b in zip(path,path[1:])}
            for edge in routes[r]:costs[edge]=float(bits)
            prior=beliefs[r];post=prior.copy();post.update_position(observation.z_xy,observation.r_xy)
            old=clipped_velocity(positions[r],prior.mean[:2]+offsets[r],controller_cfg['kp'],controller_cfg['max_speed_mps'])
            new=clipped_velocity(positions[r],post.mean[:2]+offsets[r],controller_cfg['kp'],controller_cfg['max_speed_mps'])
            # Squared realized controller-output change concentrates importance.
            action=(dt*float(np.linalg.norm(new-old))/scale)**2
            u2=float(np.trace(prior.cov[:2,:2])/2.)
            activation=max(0.,u2/cfg.uncertainty_guard_m**2-1.)
            variance_reduction=max(0.,float(np.trace(prior.cov[:2,:2]-post.cov[:2,:2])))
            safeguard=cfg.uncertainty_weight*activation*(dt*float(controller_cfg['kp'])/scale)**2*variance_reduction
            # No timeliness discount is allowed to make an urgent receiver look
            # unimportant just because its route is congested. Network gate is unchanged.
            values[r]=float((action+safeguard)/max(1,n))
            components[r]=dict(squared_action_change=action,uncertainty_safeguard=safeguard)
    valid=[f for f in factors if set(f.receivers)<=set(values)]
    total=sum(values.values())+sum(f.value for f in valid)
    budget=min(total,cfg.omission_fraction*total+cfg.absolute_event_budget)
    fraction=budget/total if total>cfg.numerical_tolerance else 1.
    forward=construct_set(values,routes,costs,cfg=DynamicConfig(omission_fraction=min(1.,fraction),numerical_tolerance=cfg.numerical_tolerance),factors=factors)
    forward['construction']='forward_value_per_service'
    alternatives=[forward]
    if cfg.use_backward_pruning:
        backward=prune_set(values,routes,costs,budget,valid,cfg.numerical_tolerance)
        backward['construction']='backward_shared_branch_pruning';alternatives.append(backward)
    # Both satisfy the SAME budget; choose lower snapshot service, then fewer recipients.
    result=min(alternatives,key=lambda r:(r['snapshot_bit_hop'],len(r['selected_recipients']),r['omitted_value']))
    result=dict(result)
    result.update(scope_kind='service_optimized_dynamic',scope_fanout_cap=None,no_fixed_recipient_cap=True,
                  candidate_count=len(values),service_config=asdict(cfg),receiver_value_components=components,
                  event_absolute_budget=cfg.absolute_event_budget,
                  context_mode='benchmark_current_receiver_beliefs_acquisition_unaccounted',
                  alternative_snapshot_costs={r['construction']:r['snapshot_bit_hop'] for r in alternatives},
                  unavailable_task_factors=[f.name for f in factors if f not in valid],
                  stop_reason='event_proxy_budget_met')
    return result
