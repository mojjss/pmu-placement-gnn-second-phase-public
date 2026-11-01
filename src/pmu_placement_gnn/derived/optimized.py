"""Portable selected method definitions; no historical module top levels."""
import hashlib
import numpy as np




class Topology:
    def __init__(self,edge_index,n,masks=False):
        ei=np.asarray(edge_index,dtype=np.int64)
        assert ei.shape[0]==2 and np.all((ei>=0)&(ei<n))
        self.n=n;self.key=hashlib.sha256(str(n).encode()+ei.tobytes()).hexdigest()
        closed=[{i} for i in range(n)]
        for u,v in ei.T:closed[int(u)].add(int(v));closed[int(v)].add(int(u))
        self.closed=closed;self.masks=None
        if masks:
            self.masks=[]
            for group in closed:
                mask=0
                for node in group:mask|=1<<node
                self.masks.append(mask)
    def verify_key(self,edge_index,n):
        key=hashlib.sha256(str(n).encode()+np.asarray(edge_index,dtype=np.int64).tobytes()).hexdigest()
        assert key==self.key and n==self.n,'Topology changed; rebuild required'

def candidates(probs,K):
    assert 0<=K<=len(probs) and np.isfinite(probs).all()
    return np.argsort(-probs)[:min(len(probs),max(2*K,80))].tolist()

def incremental(probs,edge_index,K,topology=None,candidate_override=None):
    if topology is not None:topology.verify_key(edge_index,len(probs))
    topology=topology or Topology(edge_index,len(probs))
    cand=candidates(probs,K) if candidate_override is None else list(candidate_override)
    assert len(set(cand))==len(cand) and len(cand)>=K
    reverse=[[] for _ in range(topology.n)];gains=[]
    for j,node in enumerate(cand):
        gains.append(len(topology.closed[node]))
        for bus in topology.closed[node]:reverse[bus].append(j)
    observed=[False]*topology.n;chosen=[]
    for _ in range(K):
        best=None;bj=None;bg=-1;bp=-1.
        for j,node in enumerate(cand):
            gain=gains[j]
            if gain<0:continue
            if gain>bg or (gain==bg and probs[node]>bp):best=node;bj=j;bg=gain;bp=probs[node]
        chosen.append(best);gains[bj]=-1
        for bus in topology.closed[best]:
            if observed[bus]:continue
            observed[bus]=True
            for j in reverse[bus]:
                if gains[j]>=0:gains[j]-=1
    return chosen

def greedy_incremental(edge_index,n,topology=None):
    if topology is not None:topology.verify_key(edge_index,n)
    topology=topology or Topology(edge_index,n)
    reverse=[[] for _ in range(n)];gains=[len(c) for c in topology.closed]
    for node,closed in enumerate(topology.closed):
        for bus in closed:reverse[bus].append(node)
    observed=[False]*n;count=0;chosen=[];all_nodes=set(range(n))
    while count<n:
        best=None;bg=-1
        for node in all_nodes-set(chosen):
            if gains[node]>bg:best=node;bg=gains[node]
        chosen.append(best);gains[best]=-1
        for bus in topology.closed[best]:
            if observed[bus]:continue
            observed[bus]=True;count+=1
            for node in reverse[bus]:
                if gains[node]>=0:gains[node]-=1
    return chosen
