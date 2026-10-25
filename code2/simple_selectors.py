"""Controlled, adaptive-quota baselines on identical values and physical routes.

The full selector is the unchanged original tgexp.selection.select. All simple
selectors use its omission budget and absolute numerical tolerance. Routes are
undirected physical edge sets, matching the original network implementation.
"""
from __future__ import annotations
import math
from tgexp.selection import select as original_select

STRATEGIES=('static_ratio','marginal_ratio','marginal_witness','single_prune','value_only','full')
TOL=1e-12


def select_simple(values,routes,bits,fraction=.1,absolute=0.,strategy='full',tol=TOL):
    if strategy not in STRATEGIES:raise ValueError(strategy)
    if not (math.isfinite(fraction) and 0<=fraction<=1 and math.isfinite(absolute) and absolute>=0):
        raise ValueError('Invalid omission settings')
    if not math.isfinite(bits) or bits<=0:raise ValueError('Positive payload required')
    if set(values)!=set(routes) or any(not math.isfinite(v) or v<0 for v in values.values()):
        raise ValueError('Invalid values/routes')
    if strategy=='full':
        ans=original_select(values,routes,bits,fraction,absolute,'shared',tol)
        return dict(ans,strategy=strategy)
    edges={e:k for k,e in enumerate(sorted(set().union(*routes.values()) if routes else set()))}
    masks={r:sum(1<<edges[e] for e in routes[r]) for r in values}
    total=math.fsum(values.values());budget=min(total,fraction*total+absolute);target=total-budget
    def union(rs):
        a=0
        for r in rs:a|=masks[r]
        return a
    def record(rs,construction):
        ids=sorted(rs);kept=math.fsum(values[r] for r in ids);omitted=max(0.,total-kept)
        ans=dict(strategy=strategy,construction=construction,selected=ids,
                 snapshot_cost=bits*union(ids).bit_count(),total_value=total,
                 omitted_value=omitted,omission_budget=budget,feasible=omitted<=budget+tol)
        if not ans['feasible']:raise AssertionError('Omission quota violated')
        return ans
    witness=[]
    for r in sorted(values,key=lambda r:(-values[r],r)):
        if math.fsum(values[x] for x in witness)+tol>=target:break
        witness.append(r)
    if strategy=='value_only':return record(witness,'value_prefix')
    if strategy=='static_ratio':
        # Cost is the full independent path, frozen before any member is added.
        def priority(r):
            cost=masks[r].bit_count()
            return (math.inf if cost==0 else values[r]/cost,values[r],-cost,-r)
        result=[]
        for r in sorted((r for r in values if values[r]>0),key=priority,reverse=True):
            if math.fsum(values[x] for x in result)+tol>=target:break
            result.append(r)
        return record(result,'static_independent_ratio')
    # Same priority and stopping condition as the original forward construction.
    forward=set();covered=0
    while math.fsum(values[x] for x in forward)+tol<target:
        options=[r for r in values if r not in forward and values[r]>0]
        if not options:raise AssertionError('No positive candidate before quota')
        def priority(r):
            cost=(masks[r]&~covered).bit_count()
            return (math.inf if cost==0 else values[r]/cost,values[r],-cost,-r)
        r=max(options,key=priority);forward.add(r);covered|=masks[r]
    if strategy=='marginal_ratio':return record(forward,'pure_forward')
    answers=[record(forward,'forward'),record(witness,'witness')]
    if strategy=='single_prune':
        # Same backward start and priority as full; only block operations differ.
        remaining={r for r in values if values[r]>0}
        omitted=max(0.,total-math.fsum(values[r] for r in remaining))
        while remaining:
            best=None;current=union(remaining).bit_count()
            for r in sorted(remaining):
                lost=values[r]
                if omitted+lost>budget+tol:continue
                freed=current-union(remaining-{r}).bit_count()
                if freed<=0:continue
                key=(math.inf if lost==0 else freed/lost,freed,-lost,(-r,))
                if best is None or key>best[0]:best=(key,r,lost)
            if best is None:break
            _,r,lost=best;remaining.remove(r);omitted+=lost
        answers.append(record(remaining,'backward_single'))
    ans=min(answers,key=lambda a:(a['snapshot_cost'],len(a['selected']),a['omitted_value'],a['selected']))
    ans['alternative_costs']={a['construction']:a['snapshot_cost'] for a in answers}
    ans['witness_cost']=record(witness,'witness')['snapshot_cost']
    return ans


def exact_snapshot(values,routes,bits,fraction=.1,absolute=0.,time_limit=1.):
    """Binary path-union problem, not an optimum for future closed-loop cost.

    Returns a certified optimum only for status=0 and a checked feasible integer
    solution. Time-limited feasible incumbents and lower bounds remain labeled.
    """
    import numpy as np
    from scipy.optimize import milp,Bounds,LinearConstraint
    from scipy.sparse import coo_matrix
    receivers=sorted(values);edges=sorted(set().union(*routes.values()) if routes else set())
    total=math.fsum(values.values());budget=min(total,fraction*total+absolute)
    required=max(0.,total-budget-TOL)
    if required==0:return dict(status='optimal',optimal=True,cost=0.,selected=[],lower_bound=0.,mip_gap=0.)
    n=len(receivers);edgeidx={e:k for k,e in enumerate(edges)};row=[];col=[];data=[];upper=[];lower=[]
    # Scaling makes tiny controller scores safe for the MILP feasibility tolerance.
    for k,r in enumerate(receivers):row.append(0);col.append(k);data.append(values[r]/total)
    lower.append(required/total);upper.append(np.inf)
    for k,r in enumerate(receivers):
        for edge in sorted(routes[r]):
            j=len(upper);row.extend([j,j]);col.extend([k,n+edgeidx[edge]]);data.extend([1.,-1.])
            lower.append(-np.inf);upper.append(0.)
    matrix=coo_matrix((data,(row,col)),shape=(len(upper),n+len(edges))).tocsc()
    objective=np.r_[np.zeros(n),np.ones(len(edges))]
    result=milp(objective,integrality=np.ones(len(objective)),bounds=Bounds(0,1),
                constraints=LinearConstraint(matrix,np.asarray(lower),np.asarray(upper)),
                options=dict(time_limit=float(time_limit),mip_rel_gap=0.))
    chosen=[] if result.x is None else [r for k,r in enumerate(receivers) if result.x[k]>.5]
    kept=math.fsum(values[r] for r in chosen)
    feasible=result.x is not None and total-kept<=budget+TOL
    cost=bits*len(set().union(*(routes[r] for r in chosen))) if chosen else 0.
    optimal=bool(result.status==0 and feasible)
    def finite(x):return float(x) if x is not None and math.isfinite(float(x)) else None
    return dict(status='optimal' if optimal else 'limit_feasible' if feasible else 'no_valid_solution',
                optimal=optimal,cost=float(cost) if feasible else None,selected=chosen if feasible else None,
                lower_bound=finite(getattr(result,'mip_dual_bound',None)*bits if getattr(result,'mip_dual_bound',None) is not None else None),
                mip_gap=finite(getattr(result,'mip_gap',None)),solver_status=int(result.status),message=str(result.message))
