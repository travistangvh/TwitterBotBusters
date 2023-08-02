import torch_geometric
import torch
# from torch_geometric.nn import SimpleConv

from typing import List, Optional, Union

import torch
from torch import Tensor

from torch_geometric.nn.aggr import Aggregation
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import (
    Adj,
    OptPairTensor,
    OptTensor,
    Size,
    SparseTensor,
    torch_sparse,
)
from torch_geometric.utils import add_self_loops, spmm


class SimpleConv(MessagePassing):
    r"""A simple message passing operator that performs (non-trainable)
    propagation

    .. math::
        \mathbf{x}^{\prime}_i = \bigoplus_{j \in \mathcal{N(i)}} e_{ji} \cdot
        \mathbf{x}_j

    where :math:`\bigoplus` defines a custom aggregation scheme.

    Args:
        aggr (str or [str] or Aggregation, optional): The aggregation scheme
            to use, *e.g.*, :obj:`"add"`, :obj:`"sum"` :obj:`"mean"`,
            :obj:`"min"`, :obj:`"max"` or :obj:`"mul"`.
            In addition, can be any
            :class:`~torch_geometric.nn.aggr.Aggregation` module (or any string
            that automatically resolves to it). (default: :obj:`"sum"`)
        combine_root (str, optional): Specifies whether or how to combine the
            central node representation (one of :obj:`"sum"`, :obj:`"cat"`,
            :obj:`"self_loop"`, :obj:`None`). (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **inputs:**
          node features :math:`(|\mathcal{V}|, F)` or
          :math:`((|\mathcal{V_s}|, F), (|\mathcal{V_t}|, *))`
          if bipartite,
          edge indices :math:`(2, |\mathcal{E}|)`
        - **outputs:** node features :math:`(|\mathcal{V}|, F)` or
          :math:`(|\mathcal{V_t}|, F)` if bipartite
    """
    def __init__(
        self,
        aggr: Optional[Union[str, List[str], Aggregation]] = "sum",
        combine_root: Optional[str] = None,
        **kwargs,
    ):
        if combine_root not in ['sum', 'cat', 'self_loop', None]:
            raise ValueError(f"Received invalid value for 'combine_root' "
                             f"(got '{combine_root}')")

        super().__init__(aggr, **kwargs)
        self.combine_root = combine_root

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_weight: OptTensor = None, size: Size = None) -> Tensor:

        if self.combine_root is not None:
            if self.combine_root == 'self_loop':
                if not isinstance(x, Tensor) or (size is not None
                                                 and size[0] != size[1]):
                    raise ValueError("Cannot use `combine_root='self_loop'` "
                                     "for bipartite message passing")
                if isinstance(edge_index, Tensor):
                    edge_index, edge_weight = add_self_loops(
                        edge_index, edge_weight, num_nodes=x.size(0))
                elif isinstance(edge_index, SparseTensor):
                    edge_index = torch_sparse.set_diag(edge_index)

        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                             size=size)

        x_dst = x[1]
        if x_dst is not None and self.combine_root is not None:
            if self.combine_root == 'sum':
                out = out + x_dst
            elif self.combine_root == 'cat':
                out = torch.cat([x_dst, out], dim=-1)

        return out


    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor,
                              x: OptPairTensor) -> Tensor:
        return spmm(adj_t, x[0], reduce=self.aggr)


class GNN_classifier_1_layer(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256, activate="relu", aggr = "mean"):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.activation = torch.nn.ReLU() if activate == "relu" else torch.nn.Sigmoid()
        
        self.conv1 = SimpleConv(aggr=aggr)
        
    def forward(self, x, edge_index):
        x = self.conv1(x ,edge_index)
        return x
    
class GNN_classifier(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256, activate="relu", aggr = "mean"):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.activation = torch.nn.ReLU() if activate == "relu" else torch.nn.Sigmoid()
        
        self.conv1 = SimpleConv(aggr=aggr)
        self.conv2 = SimpleConv(aggr=aggr)
        
    def forward(self, x, edge_index):
        x = self.conv1(x ,edge_index)
        x = self.activation(x)
        x = self.conv2(x, edge_index)
        return x

class GNN_classifier_3_layer(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256, activate="relu", aggr = "mean"):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.activation = torch.nn.ReLU() if activate == "relu" else torch.nn.Sigmoid()
        
        self.conv1 = SimpleConv(aggr=aggr)
        self.conv2 = SimpleConv(aggr=aggr)
        self.conv3 = SimpleConv(aggr=aggr)
        
    def forward(self, x, edge_index):
        x = self.conv1(x ,edge_index)
        x = self.activation(x)
        x = self.conv2(x, edge_index)
        x = self.activation(x)
        x = self.conv3(x, edge_index)
        return x
class GNN_classifier_4_layer(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256, activate="relu", aggr = "mean"):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.activation = torch.nn.ReLU() if activate == "relu" else torch.nn.Sigmoid()
        
        self.conv1 = SimpleConv(aggr=aggr)
        self.conv2 = SimpleConv(aggr=aggr)
        self.conv3 = SimpleConv(aggr=aggr)
        self.conv4 = SimpleConv(aggr=aggr)
        
    def forward(self, x, edge_index):
        x = self.conv1(x ,edge_index)
        x = self.activation(x)
        x = self.conv2(x, edge_index)
        x = self.activation(x)
        x = self.conv3(x, edge_index)
        x = self.activation(x)
        x = self.conv4(x, edge_index)
        return x
    
class GNN_classifier_5_layer(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256, activate="relu", aggr = "mean"):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.activation = torch.nn.ReLU() if activate == "relu" else torch.nn.Sigmoid()
        
        self.conv1 = SimpleConv(aggr=aggr)
        self.conv2 = SimpleConv(aggr=aggr)
        self.conv3 = SimpleConv(aggr=aggr)
        self.conv4 = SimpleConv(aggr=aggr)
        self.conv5 = SimpleConv(aggr=aggr)
        
    def forward(self, x, edge_index):
        x = self.conv1(x ,edge_index)
        x = self.activation(x)
        x = self.conv2(x, edge_index)
        x = self.activation(x)
        x = self.conv3(x, edge_index)
        x = self.activation(x)
        x = self.conv4(x, edge_index)
        x = self.activation(x)
        x = self.conv5(x, edge_index)
        return x
    
class GNN_classifier_6_layer(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256, activate="relu", aggr = "mean"):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.activation = torch.nn.ReLU() if activate == "relu" else torch.nn.Sigmoid()
        
        self.conv1 = SimpleConv(aggr=aggr)
        self.conv2 = SimpleConv(aggr=aggr)
        self.conv3 = SimpleConv(aggr=aggr)
        self.conv4 = SimpleConv(aggr=aggr)
        self.conv5 = SimpleConv(aggr=aggr)
        self.conv6 = SimpleConv(aggr=aggr)
        
    def forward(self, x, edge_index):
        x = self.conv1(x ,edge_index)
        x = self.activation(x)
        x = self.conv2(x, edge_index)
        x = self.activation(x)
        x = self.conv3(x, edge_index)
        x = self.activation(x)
        x = self.conv4(x, edge_index)
        x = self.activation(x)
        x = self.conv5(x, edge_index)
        x = self.activation(x)
        x = self.conv6(x, edge_index)
        return x

class GNN_classifier_7_layer(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256, activate="relu", aggr = "mean"):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.activation = torch.nn.ReLU() if activate == "relu" else torch.nn.Sigmoid()
        
        self.conv1 = SimpleConv(aggr=aggr)
        self.conv2 = SimpleConv(aggr=aggr)
        self.conv3 = SimpleConv(aggr=aggr)
        self.conv4 = SimpleConv(aggr=aggr)
        self.conv5 = SimpleConv(aggr=aggr)
        self.conv6 = SimpleConv(aggr=aggr)
        self.conv7 = SimpleConv(aggr=aggr)
        
    def forward(self, x, edge_index):
        x = self.conv1(x ,edge_index)
        x = self.activation(x)
        x = self.conv2(x, edge_index)
        x = self.activation(x)
        x = self.conv3(x, edge_index)
        x = self.activation(x)
        x = self.conv4(x, edge_index)
        x = self.activation(x)
        x = self.conv5(x, edge_index)
        x = self.activation(x)
        x = self.conv6(x, edge_index)
        x = self.activation(x)
        x = self.conv7(x, edge_index)
        return x

class GNN_classifier_8_layer(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256, activate="relu", aggr = "mean"):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.activation = torch.nn.ReLU() if activate == "relu" else torch.nn.Sigmoid()
        
        self.conv1 = SimpleConv(aggr=aggr)
        self.conv2 = SimpleConv(aggr=aggr)
        self.conv3 = SimpleConv(aggr=aggr)
        self.conv4 = SimpleConv(aggr=aggr)
        self.conv5 = SimpleConv(aggr=aggr)
        self.conv6 = SimpleConv(aggr=aggr)
        self.conv7 = SimpleConv(aggr=aggr)
        self.conv8 = SimpleConv(aggr=aggr)
        
    def forward(self, x, edge_index):
        x = self.conv1(x ,edge_index)
        x = self.activation(x)
        x = self.conv2(x, edge_index)
        x = self.activation(x)
        x = self.conv3(x, edge_index)
        x = self.activation(x)
        x = self.conv4(x, edge_index)
        x = self.activation(x)
        x = self.conv5(x, edge_index)
        x = self.activation(x)
        x = self.conv6(x, edge_index)
        x = self.activation(x)
        x = self.conv7(x, edge_index)
        x = self.activation(x)
        x = self.conv8(x, edge_index)
        return x
    