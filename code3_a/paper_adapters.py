"""Core-idea controls from four 2024/2025 papers; transparent task adaptations.

These are not original full MARL algorithms. No TaskGraph score or quota is
used by any alternative. See PAPER_TO_CODE.md for preserved ideas and changes.
"""
from __future__ import annotations
import math
import numpy as np
from structure_adapters import StructurePolicy


def opinion(mean, covariance, position, reference, weight, ctrl):
    """Model-based action evidence for 8 directions plus stopping.

    Replaces a trained evidence encoder, explicitly a task-specific adaptation.
    Five positive sigma points describe the Gaussian position uncertainty.
    """
    covariance=(np.asarray(covariance,float)+np.asarray(covariance,float).T)/2
    eig,v=np.linalg.eigh(covariance)
    axes=v@np.diag(np.sqrt(np.maximum(eig,0.)*2.5))
    points=[mean,mean+axes[:,0],mean-axes[:,0],mean+axes[:,1],mean-axes[:,1]]
    p=np.zeros(9)
    from airsim_ext.controller import clipped_velocity
    for point in points:
        action=clipped_velocity(position,reference+weight*(point-mean),ctrl['kp'],ctrl['max_speed_mps'])
        if np.linalg.norm(action)<.05:k=8
        else:k=int(np.floor((np.arctan2(action[1],action[0])+np.pi)/(2*np.pi)*8))%8
        p[k]+=.2
    # Total evidence increases when the control-relevant variance decreases.
    variance=max(float(weight*weight*np.trace(covariance)),1e-8)
    strength=min(1e6,9.*ctrl['reward_scale_m']**2/variance)
    evidence=strength*p
    S=float(evidence.sum()+9.)
    return evidence/S,9./S


def combine_opinions(b0,u0,b1,u1):
    """Dempster-Shafer/subjective-logic fusion used in T2MAC's core idea."""
    conflict=float(b0.sum()*b1.sum()-np.dot(b0,b1))
    denominator=max(1.-conflict,1e-12)
    belief=(b0*b1+b0*u1+b1*u0)/denominator
    uncertainty=float(u0*u1/denominator)
    return belief,uncertainty,conflict


def graph_features(c,receiver):
    """Eight observable edge features; no action-impact score or target truth."""
    source,q=c['source'],c['task']
    source_belief=c['source_belief'];belief=c['beliefs'].get(receiver)
    source_unc=max(1e-6,float(np.sqrt(max(0.,np.trace(source_belief.cov[:2,:2])/2))))
    if belief is None:unc=mismatch=gain=0.
    else:
        unc=float(np.sqrt(max(0.,np.trace(belief.cov[:2,:2])/2)))
        mismatch=min(4.,float(np.linalg.norm(belief.mean[:2]-source_belief.mean[:2]))/(unc+source_unc))
        from r6f.selector import expected_position_gain
        gain=expected_position_gain(belief.cov,c['observation']['r_xy'])
    distance=float(np.linalg.norm(c['positions'][receiver]-c['positions'][source]))/55.
    return np.array([1.,-min(distance,10.),mismatch,min(np.log1p(unc/6.),4.),gain,
                     c['current_roles'][receiver,q],c['future_roles'][receiver,q],
                     -min(len(c['routes'][receiver])/10.,4.)])


class PaperPolicy(StructurePolicy):
    requires_controller_score=False

    def __init__(self,job):
        kind=job['baseline_name']
        if kind in {'hygma_group','expocomm_static','expocomm_one_peer'}:
            super().__init__(job)
        else:
            self.job,self.kind,self.n=job,kind,job['n']
            self.calls=self.excluded=self.updates=self.warning_count=0
            self.update_seconds=0.;self.groups=[list(range(self.n))];self.mask_sizes=[]
            self.last_tick=-1
        self.gate_values=[]

    def update(self,c):
        if self.kind in {'hygma_group','expocomm_static','expocomm_one_peer'}:super().update(c)

    def __call__(self,c):
        self.calls+=1;self.update(c)
        source,q=c['source'],c['task'];candidates=c['candidates']
        if self.kind=='commformer_idea':
            weights=np.asarray(self.job['graph_weights'],float)
            allowed={r for r in candidates if float(graph_features(c,r)@weights)>0.}
        elif self.kind=='t2mac_idea':
            allowed=set();threshold=self.job['evidence_threshold']
            for r in candidates:
                w=float(c['current_roles'][r,q])
                prior=c['beliefs'].get(r)
                if prior is None or w<=0:continue
                reference=c['references'][r]+c['offsets'][r]
                b0,u0=opinion(prior.mean[:2],prior.cov[:2,:2],c['positions'][r],reference,w,c['controller'])
                # Incoming observation translated into this receiver's action context.
                message_ref=reference+w*(c['observation']['z_xy']-prior.mean[:2])
                b1,u1=opinion(c['observation']['z_xy'],c['observation']['r_xy'],c['positions'][r],message_ref,w,c['controller'])
                _,combined,conflict=combine_opinions(b0,u0,b1,u1)
                reduction=max(0.,(u0-combined)/max(u0,1e-12))
                self.gate_values.append(reduction)
                if reduction>threshold+1e-12:allowed.add(r)
        elif self.kind=='hygma_group':
            allowed={r for r in candidates if self.group_map[r]==self.group_map[source]}
        elif self.kind.startswith('expocomm'):
            degree=math.ceil(math.log2(self.n))+1
            incoming=np.asarray(self.exp(1,self.n,degree))[0]
            if self.kind=='expocomm_one_peer':
                column=(c['tick']//c['period'])%degree
                allowed=set(np.flatnonzero(incoming[:,column]==source).tolist())
            else:allowed=set(np.flatnonzero(np.any(incoming==source,axis=1)).tolist())
            allowed.discard(source);allowed.intersection_update(candidates)
        else:raise ValueError(self.kind)
        self.excluded+=len(candidates)-len(allowed);self.mask_sizes.append(len(allowed))
        return sorted(allowed)

    def summary(self):
        result=super().summary()
        result.update(scope='recent-paper core-idea adaptation; independent of TaskGraph selection score',
                      uses_taskgraph_receiver_score=False,uses_taskgraph_omission_quota=False,
                      graph_trained=self.job.get('graph_trained',False),
                      mean_relative_evidence_gain=float(np.mean(self.gate_values)) if self.gate_values else None)
        return result


def create_policy(job):return PaperPolicy(job)

