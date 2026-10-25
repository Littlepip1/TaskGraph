from __future__ import annotations
from dataclasses import dataclass, asdict
import numpy as np


@dataclass(frozen=True)
class ScopeConfig:
    """Frozen R8-B recipient fanout limits.

    Routine messages use a bounded local fanout equal to the original N=8 team size.
    Critical innovation/uncertainty messages may reach a larger but still N-independent
    regional fanout.  No recipient belief, target truth, future queue state, or team-global
    coordination enters the scope rule.
    """
    local_fanout: int = 8
    critical_fanout: int = 16


def nearest_recipients(source: int, reachable, positions_xy, fanout: int) -> list[int]:
    source = int(source)
    fanout = max(0, int(fanout))
    pos = np.asarray(positions_xy, dtype=float)
    rows = []
    for r in sorted(set(int(x) for x in reachable if int(x) != source)):
        d = float(np.linalg.norm(pos[r] - pos[source]))
        rows.append((d, r))
    rows.sort(key=lambda z: (z[0], z[1]))
    return [r for _, r in rows[:fanout]]


def fixed_scope(source: int, reachable, positions_xy, fanout: int) -> dict:
    recipients = nearest_recipients(source, reachable, positions_xy, fanout)
    return {
        'scope_kind': f'local_{int(fanout)}',
        'scope_fanout_cap': int(fanout),
        'selected_recipients': recipients,
    }


def adaptive_scope(source: int, reachable, positions_xy, gate_reason: str,
                   cfg: ScopeConfig = ScopeConfig()) -> dict:
    """Choose only the recipient *scope* after the frozen R7 send gate.

    Expected-gain and refresh traffic are routine and remain local.  Innovation and
    uncertainty guards receive a larger regional fanout, but never a global fanout that
    grows with N.  This is deliberately simple and interpretable for the R8-B development
    comparison against fixed local fanout baselines.
    """
    reason = str(gate_reason)
    if reason in {'innovation_guard', 'uncertainty_guard'}:
        cap = int(cfg.critical_fanout)
        kind = 'critical_regional'
    else:
        cap = int(cfg.local_fanout)
        kind = 'routine_local'
    recipients = nearest_recipients(source, reachable, positions_xy, cap)
    return {
        'scope_kind': kind,
        'scope_fanout_cap': cap,
        'selected_recipients': recipients,
    }


def scope_config_dict(cfg: ScopeConfig) -> dict:
    return asdict(cfg)


def diverse_recipients(source: int, reachable, positions_xy, fanout: int) -> list[int]:
    """Balanced N-independent fanout using source-ID circulant offsets.

    R8-A assigns stable source IDs by polar angle. Applying the same set of cyclic ID
    offsets at every sender gives every receiver the same in-degree while spreading its
    incoming sources around the formation instead of keeping them only in one local sector.
    If a nominal target is currently unreachable, the nearest unused reachable ID on the
    cyclic index is substituted deterministically.
    """
    source=int(source); fanout=max(0,int(fanout)); n=len(np.asarray(positions_xy))
    reach=sorted(set(int(x) for x in reachable if int(x)!=source))
    if fanout >= len(reach):
        return reach
    if not reach or fanout<=0:
        return []
    rset=set(reach); chosen=[]
    ideals=[]
    for m in range(1,fanout+1):
        off=int(round(m*n/(fanout+1.0)))
        t=(source+off)%n
        if t==source:
            t=(t+1)%n
        ideals.append(t)
    def circ(a,b):
        d=abs(int(a)-int(b)); return min(d,n-d)
    for t in ideals:
        candidates=[x for x in reach if x not in chosen]
        if not candidates: break
        best=min(candidates,key=lambda x:(circ(x,t),x))
        chosen.append(best)
    return chosen


def fixed_diverse_scope(source: int, reachable, positions_xy, fanout: int) -> dict:
    recipients=diverse_recipients(source,reachable,positions_xy,fanout)
    return {'scope_kind':f'diverse_{int(fanout)}','scope_fanout_cap':int(fanout),'selected_recipients':recipients}


def adaptive_diverse_scope(source: int, reachable, positions_xy, gate_reason: str,
                           cfg: ScopeConfig = ScopeConfig()) -> dict:
    reason=str(gate_reason)
    if reason in {'innovation_guard','uncertainty_guard'}:
        cap=int(cfg.critical_fanout); kind='critical_diverse'
    else:
        cap=int(cfg.local_fanout); kind='routine_diverse'
    recipients=diverse_recipients(source,reachable,positions_xy,cap)
    return {'scope_kind':kind,'scope_fanout_cap':cap,'selected_recipients':recipients}


def rotating_diverse_recipients(source: int, reachable, positions_xy, fanout: int,
                                tick: int, opportunity_period: int = 16) -> list[int]:
    """Balanced rotating circulant fanout.

    Every cycle selects one residue class of cyclic source-ID offsets, so each individual
    message has bounded fanout while consecutive cycles expose a receiver to different,
    spatially distributed senders.  Source IDs are stable polar-angle IDs from R8-A.
    """
    source=int(source); n=len(np.asarray(positions_xy)); fanout=max(0,int(fanout))
    reach=sorted(set(int(x) for x in reachable if int(x)!=source)); rset=set(reach)
    if fanout>=len(reach): return reach
    if not reach or fanout<=0: return []
    step=max(1,n//fanout)
    cycle=max(0,int(tick)//max(1,int(opportunity_period)))
    phase=cycle % step
    offsets=[o for o in range(1,n) if (o-1) % step == phase]
    # The residue class is already evenly spaced around the cyclic ID geometry.  Fill
    # short classes from following phases, preserving deterministic balance.
    q=1
    while len(offsets)<fanout and q<step+1:
        ph=(phase+q)%step
        for o in range(1,n):
            if (o-1)%step==ph and o not in offsets:
                offsets.append(o)
                if len(offsets)>=fanout: break
        q+=1
    selected=[]
    for o in offsets:
        target=(source+o)%n
        if target in rset and target not in selected:
            selected.append(target)
        else:
            # Substitute the nearest reachable cyclic ID if topology removed the ideal.
            candidates=[x for x in reach if x not in selected]
            if not candidates: break
            def circ(a,b):
                d=abs(int(a)-int(b)); return min(d,n-d)
            selected.append(min(candidates,key=lambda x:(circ(x,target),x)))
        if len(selected)>=fanout: break
    return selected


def fixed_rotating_scope(source: int, reachable, positions_xy, fanout: int, tick: int,
                         opportunity_period: int = 16) -> dict:
    recipients=rotating_diverse_recipients(source,reachable,positions_xy,fanout,tick,opportunity_period)
    return {'scope_kind':f'rotating_{int(fanout)}','scope_fanout_cap':int(fanout),'selected_recipients':recipients}


def adaptive_rotating_scope(source: int, reachable, positions_xy, gate_reason: str, tick: int,
                            cfg: ScopeConfig = ScopeConfig(), opportunity_period: int = 16) -> dict:
    reason=str(gate_reason)
    if reason in {'innovation_guard','uncertainty_guard'}:
        cap=int(cfg.critical_fanout); kind='critical_rotating'
    else:
        cap=int(cfg.local_fanout); kind='routine_rotating'
    recipients=rotating_diverse_recipients(source,reachable,positions_xy,cap,tick,opportunity_period)
    return {'scope_kind':kind,'scope_fanout_cap':cap,'selected_recipients':recipients}
