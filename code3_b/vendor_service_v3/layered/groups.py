"""Static first-layer snapshot filter, applied before the frozen R3 selector."""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path


class GroupFilter:
    def __init__(self, snapshot, tracker_names):
        self.snapshot=copy.deepcopy(snapshot)
        self.names=tuple(tracker_names)
        ids=tuple(snapshot['agent_ids']);labels=list(snapshot['labels'])
        if len(set(ids))!=len(ids) or len(ids)!=len(labels) or set(ids)!=set(self.names) or len(set(self.names))!=len(self.names):
            raise ValueError('Snapshot must contain exactly the configured tracker IDs, without duplicates')
        if any(type(x) is not int or x<0 for x in labels):
            raise ValueError('Group labels must be nonnegative integers')
        lookup=dict(zip(ids,labels));self.labels=tuple(lookup[a] for a in self.names)
        valid=set(self.names);self.exceptions={i:set() for i in range(len(self.names))};ix={a:i for i,a in enumerate(self.names)}
        for a,b in snapshot.get('protected_pairs',[]):
            if a not in valid or b not in valid or a==b:raise ValueError('Invalid protected candidate pair')
            self.exceptions[ix[a]].add(ix[b])
        self.version=str(snapshot.get('group_version','preloaded-v1'))
        self.digest=hashlib.sha256(json.dumps(snapshot,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        self.reset_audit()

    def validate_agent_count(self,n):
        if n!=len(self.labels):raise ValueError('Group count does not match experiment')

    def reset_audit(self):
        self.calls=self.before=self.after=0

    def filter(self,source,candidates,tick):
        source=int(source)
        if not 0<=source<len(self.names):raise ValueError('Invalid source')
        raw=list(candidates)
        if any(type(r) is not int or not 0<=r<len(self.names) for r in raw):raise ValueError('Invalid reachable candidate')
        selected=[r for r in raw if r!=source and (self.labels[r]==self.labels[source] or r in self.exceptions[source])]
        self.calls+=1;self.before+=len(raw);self.after+=len(selected)
        return selected

    def summary(self):
        sizes={str(q):self.labels.count(q) for q in sorted(set(self.labels))}
        return dict(group_version=self.version,snapshot_sha256=self.digest,
                    group_sizes=sizes,filter_calls=self.calls,raw_candidate_total=self.before,
                    filtered_candidate_total=self.after,
                    maintenance_mode='preinstalled_static_snapshot',online_maintenance_generated_bits=0,
                    startup_provisioning_cost_accounted=False,
                    transit_relays_restricted=False,
                    note='No dynamic update or task discovery implemented; no claim of zero-cost deployment')


def load_filter(path,tracker_names):
    return GroupFilter(json.loads(Path(path).read_text(encoding='utf-8')),tracker_names)
