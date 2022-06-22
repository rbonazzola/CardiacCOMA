import itertools
import torch
from torch_scatter import scatter_add
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.cheb_conv import ChebConv
from torch_geometric.utils import remove_self_loops

from subprocess import check_output
import shlex
# import sys; sys.path.append(".")
repo_root = check_output(shlex.split("git rev-parse --show-toplevel")).strip().decode('ascii')

from IPython import embed
from utils.utils import normal

__author__ = ['Priyanka Patel', 'Rodrigo Bonazzola']

# N: number of subjects
# M: number of vertices in mesh
# F: number of features (typically, 3: x, y and z)

# https://pytorch-geometric.readthedocs.io/en/1.3.1/_modules/torch_geometric/nn/conv/cheb_conv.html
# https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/conv/cheb_conv.html#ChebConv
class ChebConv_Coma(ChebConv):

    def __init__(self, in_channels, out_channels, K, normalization=None, bias=True):
        super(ChebConv_Coma, self).__init__(in_channels, out_channels, K, normalization, bias)

   #def reset_parameters(self):
   #     embed()
   #     normal(self.weight, 0, 0.1) # Same as torch.nn.init.normal_ but handles None's
   #     normal(self.bias, 0, 0.1)


    # Normalized Laplacian. This is almost entirely copied from the parent class, ChebConv.
    @staticmethod
    def norm(edge_index, num_nodes, edge_weight=None, dtype=None):
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)

        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ),
                                     dtype=dtype,
                                     device=edge_index.device)
        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes) # TODO: Check what scatter_add does.
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        return edge_index, -deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]


    def forward(self, x, edge_index, norm, edge_weight=None):
        # Tx_i are Chebyshev polynomials of x, which are computed recursively
        Tx_0 = x # Tx_0 is the identity, i.e. Tx_0(x) == x

        #TOFIX: This is a workaround to make my code work with a newer version of PyTorch (1.10),
        #since the weight attribute seems to be absent in this version.
        self.weight = []
        #TODO: change this range
        for i in range(1, 7):            
            try:
              self.weight.append(next(itertools.islice(self.parameters(), i, None)).t())
            except:
              pass

        out = torch.matmul(Tx_0, self.weight[0])

        # if self.weight.size(0) > 1:
        if len(self.weight) > 1:
            Tx_1 = self.propagate(edge_index, x=Tx_0, norm=norm) # propagate amounts to operator composition
            out = out + torch.matmul(Tx_1, self.weight[1])

        # for k in range(2, self.weight.size(0)):
        for k in range(2, len(self.weight)):
            Tx_2 = 2 * self.propagate(edge_index, x=Tx_1, norm=norm) - Tx_0 # recursive definition of Chebyshev polynomials
            out = out + torch.matmul(Tx_2, self.weight[k])
            Tx_0, Tx_1 = Tx_1, Tx_2

        if self.bias is not None:
            out = out + self.bias

        return out

    def message(self, x_j, norm):
        # x_j is in the format (N, M, F)
        return norm.view(1, -1, 1) * x_j


class Pool(MessagePassing):

    '''
    This module can be used in two ways:
      - Provide a pool matrix on initialization.
      - Initialize it generically, and provide the pool matrix when calling the forward method.
    '''

    def __init__(self, pool_mat=None):
        # source_to_target is the default value for flow, but is specified here for explicitness
        self.pool_mat = pool_mat
        if self.pool_mat is not None:
            self.pool_mat = pool_mat.transpose(0, 1)
        super(Pool, self).__init__(flow='source_to_target')

    def forward(self, x, pool_mat=None,  dtype=None):
        if self.pool_mat is None:
            pool_mat = pool_mat.transpose(0, 1)
        if self.pool_mat is not None:
            pool_mat = self.pool_mat
        out = self.propagate(edge_index=pool_mat._indices(), x=x, norm=pool_mat._values(), size=pool_mat.size())
        return out

    def message(self, x_j, norm):
        return norm.view(1, -1, 1) * x_j