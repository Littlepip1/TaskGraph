"""Deterministic bottom-k event sampling, independent of solver performance."""
import hashlib,heapq,time

class SnapshotCapture:
    def __init__(self,job):
        self.job=job;self.limit=int(job.get('snapshot_count',0));self.heap=[];self.visited=0;self.seconds=0.
    def add(self,tick,source,task,values,routes,bits,protected):
        if not self.limit or protected:return
        start=time.perf_counter();self.visited+=1
        key=f'{self.job["seed"]}:{tick}:{source}:{task}'
        priority=int(hashlib.sha256(key.encode()).hexdigest(),16)
        if len(self.heap)>=self.limit and priority>=-self.heap[0][0]:
            self.seconds+=time.perf_counter()-start;return
        event=dict(tick=int(tick),source=int(source),task=int(task),values={str(r):float(v) for r,v in values.items()},
                   routes={str(r):[[int(x) for x in e] for e in sorted(es)] for r,es in routes.items()},bits=bits,
                   fraction=self.job['fraction'],absolute=self.job.get('absolute',0.),priority=str(priority),
                   host_strategy=self.job.get('simple_strategy','full'))
        item=(-priority,key,event)
        if len(self.heap)<self.limit:heapq.heappush(self.heap,item)
        else:heapq.heapreplace(self.heap,item)
        self.seconds+=time.perf_counter()-start
    def export(self):return [x[2] for x in sorted(self.heap,key=lambda x:x[1])]
