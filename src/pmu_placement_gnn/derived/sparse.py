"""Portable selected method definitions; no historical module top levels."""
import torch
from torch_geometric.nn.conv.gcn_conv import gcn_norm
opt = None




def adjacency(model,data,kind):
    a,b=model.conv1,model.conv2
    assert not model.training and not a.cached and not b.cached
    assert a.normalize and b.normalize
    assert (a.improved,a.add_self_loops,a.flow)==(b.improved,b.add_self_loops,b.flow)
    assert data.x.dtype==a.lin.weight.dtype==torch.float32
    edges,weights=gcn_norm(data.edge_index,None,data.num_nodes,a.improved,a.add_self_loops,a.flow,data.x.dtype)
    # COO normalization precedes coalescing: raw duplicate edges count toward degrees.
    target,source=(edges[1],edges[0]) if a.flow=='source_to_target' else (edges[0],edges[1])
    indices=torch.stack((target,source))
    if kind=='sparse':
        return torch.sparse_coo_tensor(indices,weights,(data.num_nodes,data.num_nodes),dtype=data.x.dtype,device=data.x.device).coalesce()
    if kind=='dense':
        matrix=torch.zeros((data.num_nodes,data.num_nodes),dtype=data.x.dtype,device=data.x.device)
        matrix.index_put_((target,source),weights,accumulate=True)
        return matrix
    raise ValueError(kind)

def with_adjacency(model,data,matrix,kind):
    x=data.x
    for layer in [model.conv1,model.conv2]:
        # Preserve linear-before-aggregation and bias-after-aggregation.
        x=layer.lin(x)
        x=torch.sparse.mm(matrix,x) if kind=='sparse' else torch.mm(matrix,x)
        if layer.bias is not None:x=x+layer.bias
        x=torch.relu(x)
    return model.lin(x)

def logits(model,data,kind,inference=False):
    context=torch.inference_mode() if inference else torch.no_grad()
    with context:return with_adjacency(model,data,adjacency(model,data,kind),kind)

def probabilities(model,data,kind,inference=False):
    context=torch.inference_mode() if inference else torch.no_grad()
    with context:return torch.softmax(logits(model,data,kind,inference),-1)[:,1].detach().cpu().numpy()

def supplied(model,data,K,kind,inference=False):
    return opt.incremental(probabilities(model,data,kind,inference),data.edge_index.detach().cpu().numpy(),K)

def acquired(model,data,kind,inference=False):
    edges=data.edge_index.detach().cpu().numpy();topology=opt.Topology(edges,data.num_nodes)
    K=len(opt.greedy_incremental(edges,data.num_nodes,topology))
    return opt.incremental(probabilities(model,data,kind,inference),edges,K,topology)
