"""Deterministic kinematic integration client, NOT AirSim or old software benchmark.

Uses the uploaded R3 estimator/controller/network unchanged. SP/MIS-like paths
and explicit observation dropout exercise the new seam; they do not reproduce
missing historical r1 software or multi-target experiments.
"""
import math
import numpy as np


class IntegrationKinematicClient:
    fixed_step_control=False
    def __init__(self,airsim_cfg,offsets,family='SP',seed=0,dropout=False):
        self.cfg=airsim_cfg;self.offsets=np.asarray(offsets,float);self.family=family;self.seed=int(seed);self.dropout=dropout
        rng=np.random.default_rng(seed)
        self.heading=float(rng.uniform(-math.pi,math.pi))
        self.drop_sources=set(int(x) for x in rng.choice(len(offsets),len(offsets)//2,replace=False))
        self.reset_and_prepare()
    def connect(self):return {}
    def reset_and_prepare(self):
        self.time=0.;self.pos=self.target_state_xyv()[:2]+self.offsets.copy()
    def start_target_path(self):pass
    def mark_collision_baseline(self):pass
    def new_collision_events(self):return []
    def positions_xy(self):return self.pos.copy()
    def target_state_xyv(self):
        t=self.time
        if self.family=='SP':
            a=t*5/30
            state=np.array([30*math.cos(a),30*math.sin(a),-5*math.sin(a),5*math.cos(a)])
        elif self.family=='MIS':
            # A deliberately explicit integration fixture, turn at t=12 seconds.
            state=np.array([30.,3*t,0.,3.]) if t<12 else np.array([30+3*(t-12),36.,3.,0.])
        else:raise ValueError(self.family)
        c,s=math.cos(self.heading),math.sin(self.heading);rot=np.array([[c,-s],[s,c]])
        return np.r_[rot@state[:2],rot@state[2:]]
    def sensor_available(self,agent,tick):
        return not(self.dropout and self.family=='MIS' and 16<=tick<48 and agent in self.drop_sources)
    def pause(self,flag):pass
    def command_tracker_velocities(self,v,duration_s):
        self.pos+=np.asarray(v,float)*float(duration_s);self.time+=float(duration_s)
    def hover(self):pass
