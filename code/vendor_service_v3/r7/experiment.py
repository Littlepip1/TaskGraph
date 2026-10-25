from __future__ import annotations
import copy
import numpy as np

from airsim_ext.belief import TargetBelief2D
from airsim_ext.controller import clipped_velocity, formation_offsets, normalized_tracking_reward
from airsim_ext.network import DynamicNetwork
from airsim_ext.sensing import sample_observation
from r4.estimation import ReplayEstimator
from r6f.selector import LocalGateConfig, decide_sender_local
from r7.selector import PressureConfig, ReceivedAgeHistory, local_telemetry, decide_pressure_local

ADAPTIVE_METHOD = 'adaptive_local_safe'
PRESSURE_METHODS = {'r7_local', 'r7_queue_only'}
VALID_METHODS = {'uniform_p16', 'static_7of8', ADAPTIVE_METHOD, 'no_comm'} | PRESSURE_METHODS


def _summary(v):
    a = np.asarray(v, float)
    if not len(a): return {'n': 0, 'mean': None, 'min': None, 'max': None}
    return {'n': int(len(a)), 'mean': float(a.mean()), 'min': float(a.min()), 'max': float(a.max())}


def due_sources(tick: int, n: int):
    """One opportunity per source per 16 ticks, independent of N.

    Sources are assigned one of the eight even phases by source_id mod 8.  Thus N=8
    has one due sender on an even tick, N=16 has two, and N=32 has four.  We do NOT
    lengthen the per-source communication clock as N grows.
    """
    phase = int(tick) % 16
    if phase % 2 or phase > 14:
        return ()
    cls = phase // 2
    return tuple(j for j in range(int(n)) if j % 8 == cls)


def static_keep(source: int) -> bool:
    """Frozen 7/8 periodic static comparator; delete modular class 2 at every scale."""
    return int(source) % 8 != 2


def _network_cfg(base_cfg, capacity):
    c = copy.deepcopy(base_cfg)
    c['bits_per_link_tick'] = int(capacity); c['drop_probability'] = 0.0
    return c


class R7ClosedLoopExperiment:
    """R7 development with unchanged R6-F and uniform comparators."""
    def __init__(self, config, r5_protocol, client, capacity, gate_cfg: LocalGateConfig, pressure_cfg=PressureConfig()):
        self.cfg = copy.deepcopy(config); self.r5p = copy.deepcopy(r5_protocol); self.client = client
        self.n = len(config['airsim']['trackers'])
        if self.n not in {8, 16, 32}: raise ValueError(self.n)
        self.dt = float(config['experiment'].get('dt_s', 0.5)); self.eval_start = int(config['experiment'].get('eval_start_step', 20))
        self.sensor_cfg = config['sensor']; self.planner_cfg = config['planner']; self.ctrl_cfg = config['controller']
        self.net_cfg = _network_cfg(config['network'], capacity)
        self.offsets = formation_offsets(self.n, float(self.ctrl_cfg['formation_radius_m']))
        self.gate_cfg = gate_cfg
        self.pressure_cfg = pressure_cfg
        if pressure_cfg.replay_lag != int(r5_protocol['replay_lag_ticks']):
            raise ValueError('Pressure expiry window must match estimator replay lag')

    def run(self, method, seed, steps):
        if method not in VALID_METHODS: raise ValueError(method)
        self.client.connect(); self.client.reset_and_prepare(); self.client.start_target_path()
        xy0 = self.client.positions_xy()
        estimators = [ReplayEstimator(TargetBelief2D.broad(
            xy0[i], self.planner_cfg['initial_pos_std_m'], self.planner_cfg['initial_vel_std_mps'],
            self.planner_cfg['process_accel_std_mps2']), self.dt, self.r5p['replay_lag_ticks']) for i in range(self.n)]
        net = DynamicNetwork(self.n, self.net_cfg, int(seed) + 17)
        last_sent = [None] * self.n
        age_history = [ReceivedAgeHistory(self.pressure_cfg.age_window) for _ in range(self.n)]
        too_old_deliveries = 0
        queue_trace = []
        rewards=[]; formation=[]; esterr=[]; actionerr=[]; ages=[]; eval_ages=[]
        decisions=[]; sent=0; opp=0; suppressed=0; max_reserved=0.0; disconnected_ticks=0
        payload_bits=int(self.net_cfg['payload_bits'])
        try:
            for tick in range(int(steps)):
                self.client.pause(True)
                positions=self.client.positions_xy(); target_state=self.client.target_state_xyv(); target=target_state[:2]
                net.rebuild(positions)
                # Connectivity is a scientific diagnostic, not silently repaired.
                if any(net.path(0, i) is None for i in range(self.n)):
                    disconnected_ticks += 1
                remote=[[] for _ in range(self.n)]
                for d in net.receive(tick):
                    remote[d.receiver].append(d.payload); age=tick-int(d.payload.tick); ages.append(age)
                    age_history[d.receiver].observe(tick, int(d.payload.tick))
                    too_old_deliveries += int(age > self.pressure_cfg.replay_lag)
                    if tick>=self.eval_start: eval_ages.append(age)
                obs=[]
                for j in range(self.n):
                    rng=np.random.default_rng(np.random.SeedSequence([int(seed), tick, j]))
                    o=sample_observation(j,tick,positions[j],target,rng,self.sensor_cfg)
                    if hasattr(self.client,'sensor_available') and not self.client.sensor_available(j,tick): o=None
                    obs.append(o)

                due = due_sources(tick, self.n)
                priors = {}
                if method == ADAPTIVE_METHOD or method in PRESSURE_METHODS:
                    for j in due:
                        shadow=copy.deepcopy(estimators[j]); priors[j]=shadow.advance(tick,list(remote[j])).copy()

                beliefs=[]
                for i,e in enumerate(estimators):
                    rec=list(remote[i])
                    if obs[i] is not None: rec.append(obs[i])
                    beliefs.append(e.advance(tick,rec))
                actions=np.stack([clipped_velocity(positions[i],beliefs[i].mean[:2]+self.offsets[i],
                                                    self.ctrl_cfg['kp'],self.ctrl_cfg['max_speed_mps']) for i in range(self.n)])
                oracle=np.stack([clipped_velocity(positions[i],target+self.offsets[i],
                                                   self.ctrl_cfg['kp'],self.ctrl_cfg['max_speed_mps']) for i in range(self.n)])
                if tick>=self.eval_start:
                    rewards.append(float(np.mean([normalized_tracking_reward(positions[i],target,self.offsets[i],self.ctrl_cfg['reward_scale_m']) for i in range(self.n)])))
                    formation.append(float(np.mean([np.linalg.norm((positions[i]-target)-self.offsets[i]) for i in range(self.n)])))
                    esterr.append(float(np.mean([np.linalg.norm(beliefs[i].mean[:2]-target) for i in range(self.n)])))
                    actionerr.append(float(np.mean(np.linalg.norm(actions-oracle,axis=1))))

                for j in due:
                    j=int(j); opp += 1; o=obs[j]
                    recipients=[i for i in range(self.n) if i!=j and net.path(j,i) is not None]
                    should_send = (o is not None) and method != 'no_comm'
                    decision={'tick':tick,'source':j,'method':method,'had_measurement':o is not None,'due_group_size':len(due)}
                    if method=='static_7of8':
                        should_send = should_send and static_keep(j)
                        decision.update(send=bool(should_send),reason=('static_keep' if should_send else ('no_measurement' if o is None else 'static_delete')))
                    elif method in PRESSURE_METHODS:
                        neighbors = tuple(net.adj[j])
                        incident = {v: net.next_free.get(net._edge_key(j,v), float(tick)) for v in neighbors}
                        telemetry = local_telemetry(j, neighbors, incident, tick, payload_bits,
                            int(self.net_cfg['bits_per_link_tick']), int(self.net_cfg.get('prop_delay_ticks',1)), age_history[j])
                        gd = decide_pressure_local(priors[j], o, last_sent[j], tick, self.gate_cfg,
                            telemetry, self.pressure_cfg, use_received_age=(method=='r7_local'))
                        decision.update(gd); should_send=bool(gd['send'])
                    elif method==ADAPTIVE_METHOD:
                        if o is None:
                            should_send=False; gd=decide_sender_local(priors[j],o,0,payload_bits,last_sent[j],tick,self.gate_cfg)
                        else:
                            cost=net.multicast_cost(j,recipients,payload_bits)
                            gd=decide_sender_local(priors[j],o,cost,payload_bits,last_sent[j],tick,self.gate_cfg); should_send=bool(gd.send)
                        decision.update(gd.to_dict())
                    elif method=='uniform_p16':
                        decision.update(send=bool(should_send),reason=('baseline_send' if should_send else 'no_measurement'))
                    else:
                        should_send=False; decision.update(send=False,reason=('no_measurement' if o is None else 'no_comm'))
                    if should_send and recipients:
                        net.schedule_multicast(j,recipients,payload_bits,tick,o); last_sent[j]=tick; sent+=1
                    else:
                        suppressed += int(o is not None)
                    decision['actually_scheduled'] = bool(should_send and recipients)
                    decisions.append(decision)
                max_reserved=max(max_reserved,max(0.0,max(net.next_free.values(),default=float(tick))-float(tick)))
                queue_trace.append(max(0.0,max(net.next_free.values(),default=float(tick))-float(tick)))
                self.client.pause(False); self.client.command_tracker_velocities(actions,self.dt)
            self.client.hover()
        finally:
            self.client.pause(False)
        ns=net.stats(); ns['max_reserved_work_ticks']=float(max_reserved)
        ns['too_old_received_deliveries']=int(too_old_deliveries)
        ns['too_old_received_fraction']=float(too_old_deliveries/max(1,ns['delivered_packets']))
        ns['mean_reserved_work_ticks']=float(np.mean(queue_trace)) if queue_trace else 0.0
        ns['estimator_too_old_observations']=int(sum(e.too_old for e in estimators))
        ns['pending_after_horizon_fraction']=float(ns['pending_deliveries']/max(1,ns['pending_deliveries']+ns['delivered_packets']))
        return {
            'method':method,'agents':self.n,'seed':int(seed),'steps':int(steps),'eval_start_step':self.eval_start,
            'gate_config':self.gate_cfg.__dict__ if method==ADAPTIVE_METHOD or method in PRESSURE_METHODS else None,
            'pressure_config': self.pressure_cfg.__dict__ if method in PRESSURE_METHODS else None,
            'evaluation':{'team_reward':_summary(rewards),'formation_error_m':_summary(formation),'estimation_error_m':_summary(esterr),'oracle_action_error_mps':_summary(actionerr)},
            'network':ns,'scheduled_opportunities':int(opp),'sent_payloads':int(sent),'suppressed_payloads':int(suppressed),
            'send_fraction_of_opportunities':float(sent/opp) if opp else 0.0,'delivered_age_ticks':_summary(ages),'eval_delivered_age_ticks':_summary(eval_ages),
            'disconnected_control_ticks':int(disconnected_ticks),'bit_hop_per_agent_control_tick':float(ns['total_bit_hop'])/max(1.0,float(self.n*steps)),
            'send_decisions':decisions,
            'scope':'R7 development software kinematics; inherited prebooked multicast network and public routing. Gate uses sender belief, observation, incident calendars and past received ages only. Not AirSim or independent final evidence.'
        }
