import numpy as np
import torch
from torch import nn
from torch.nn import ModuleList, ModuleDict
from .layers import ChebConv_Coma, Pool
from copy import copy
from typing import Sequence, Union, List
from IPython import embed # left there for debugging if needed

#TODO: Implement common parent class for encoder and decoder (GraphConvStack?), to capture common behaviour.

################# FULL AUTOENCODER #################

class Autoencoder3DMesh(nn.Module):

    def __init__(self, enc_config, dec_config, other_args):

        super(Autoencoder3DMesh, self).__init__()

        self.encoder = Encoder3DMesh(**enc_config)
        self.decoder = Decoder3DMesh(**dec_config)
        self._is_variational = other_args["is_variational"]
        self.template_mesh = other_args["template_mesh"]

        self.matrices = {}
        self.matrices["A_edge_index"] = self.encoder.matrices["A_edge_index"] 
        self.matrices["A_norm"] = self.encoder.matrices["A_norm"] 
        self.matrices["downsample"] = self.encoder.matrices["downsample"] 
        self.matrices["upsample"] = self.decoder.matrices["upsample"] 


    def forward(self, x):

        bottleneck = self.encoder(x)

        mu, log_var = tuple([bottleneck[k] for k in ["mu", "log_var"]])

        if self._is_variational and self.mode == "training" :
            z = self.sampling(mu, log_var)
        else:
            z = mu

        x_hat = self.decoder(z)
        return x_hat, bottleneck

    def set_mode(self, mode):
        self.mode = mode

################# ENCODER #################

ENCODER_ARGS = [
    "num_features",
    "n_layers",
    "n_nodes",
    "num_conv_filters_enc",
    "cheb_polynomial_order",
    "latent_dim",
    "template",
    "is_variational",
    "downsample_matrices",
    "adjacency_matrices",
    "activation_layers"
]

class Encoder3DMesh(nn.Module):

    '''
    '''

    def __init__(self,
        num_conv_filters_enc: Sequence[int],
        num_features: int,
        cheb_polynomial_order: int,
        n_layers: int,
        n_nodes: int,
        is_variational: bool,
        latent_dim: int,
        template,
        adjacency_matrices: List[torch.Tensor],
        downsample_matrices: List[torch.Tensor],
        activation_layers="ReLU"):

        super(Encoder3DMesh, self).__init__()

        self.n_nodes = n_nodes
        self.filters_enc = copy(num_conv_filters_enc)
        self.filters_enc.insert(0, num_features)
        self.K = cheb_polynomial_order

        self.matrices = {}
        A_edge_index, A_norm = self._build_adj_matrix(adjacency_matrices)
        self.matrices["A_edge_index"] = A_edge_index
        self.matrices["A_norm"] = A_norm
        self.matrices["downsample"] = downsample_matrices
        #self.matrices["A_edge_index"] = list(reversed(A_edge_index))
        #self.matrices["A_norm"] = list(reversed(A_norm))
        #self.matrices["downsample"] = list(reversed(downsample_matrices))

        self._n_features_before_z = self.matrices["downsample"][-1].shape[0] * self.filters_enc[-1]
        self._is_variational = is_variational
        self.latent_dim = latent_dim

        self.activation_layers = [activation_layers] * n_layers if isinstance(activation_layers, str) else activation_layers
        self.layers = self._build_encoder()

        # Fully connected layers connecting the last pooling layer and the latent space layer.
        self.enc_lin_mu = torch.nn.Linear(self._n_features_before_z, self.latent_dim)

        if self._is_variational:
            self.enc_lin_var = torch.nn.Linear(self._n_features_before_z, self.latent_dim)

    def _build_encoder(self):

        cheb_conv_layers = self._build_cheb_conv_layers(self.filters_enc, self.K)
        pool_layers = self._build_pool_layers(self.matrices["downsample"])
        activation_layers = self._build_activation_layers(self.activation_layers)

        encoder = ModuleDict()

        for i in range(len(cheb_conv_layers)):
            layer = f"layer_{i}"
            encoder[layer] = ModuleDict()            
            encoder[layer]["graph_conv"] = cheb_conv_layers[i]
            encoder[layer]["pool"] = pool_layers[i]
            encoder[layer]["activation_function"] = activation_layers[i]

        return encoder

    def _build_pool_layers(self, downsample_matrices:Sequence[np.array]):

        '''
        downsample_matrices: list of matrices binary matrices
        '''

        pool_layers = ModuleList()
        for i in range(len(downsample_matrices)):
            pool_layers.append(Pool())
        return pool_layers


    def _build_activation_layers(self, activation_type:Union[str, Sequence[str]]):

        '''
        activation_type: string or list of strings containing the name of a valid activation function from torch.functional
        '''

        activation_layers = ModuleList()

        for i in range(len(activation_type)):
            activ_fun = getattr(torch.nn.modules.activation, activation_type[i])()
            activation_layers.append(activ_fun)

        return activation_layers


    def _build_cheb_conv_layers(self, n_filters, K):
        # Chebyshev convolutions (encoder)

        cheb_enc = torch.nn.ModuleList([ChebConv_Coma(n_filters[0], n_filters[1], K[0])])
        cheb_enc.extend([
            ChebConv_Coma(
                n_filters[i],
                n_filters[i+1],
                K[i]
            ) for i in range(1, len(n_filters)-1)
        ])
        return cheb_enc


    def _build_adj_matrix(self, adjacency_matrices):
        adj_edge_index, adj_norm = zip(*[
            ChebConv_Coma.norm(adjacency_matrices[i]._indices(), self.n_nodes[i])
            for i in range(len(self.n_nodes))
        ])
        return list(adj_edge_index), list(adj_norm)

    
    def concatenate_graph_features(self, x):
        x = x.reshape(x.shape[0], self._n_features_before_z)
        return x


    def forward(self, x):

        # a "layer" here is: a graph convolution + pooling operation + activation function
        for i, layer in enumerate(self.layers): 
            
            if self.matrices["downsample"][i].device != x.device:
                self.matrices["downsample"][i] = self.matrices["downsample"][i].to(x.device)
            if self.matrices["A_edge_index"][i].device != x.device:
                self.matrices["A_edge_index"][i] = self.matrices["A_edge_index"][i].to(x.device)
            if self.matrices["A_norm"][i].device != x.device:
                self.matrices["A_norm"][i] = self.matrices["A_norm"][i].to(x.device)
  
            x = self.layers[layer]["graph_conv"](x, self.matrices["A_edge_index"][i], self.matrices["A_norm"][i])
            x = self.layers[layer]["pool"](x, self.matrices["downsample"][i])
            x = self.layers[layer]["activation_function"](x)
        
        
        x  = self.concatenate_graph_features(x)
       
        mu = self.enc_lin_mu(x)
        log_var = self.enc_lin_var(x) if self._is_variational else None

        return {"mu": mu, "log_var": log_var}

################# DECODER #################

DECODER_ARGS = [
    "num_features",
    "n_layers",
    "n_nodes",
    "num_conv_filters_dec",
    "cheb_polynomial_order",
    "latent_dim",
    "is_variational",
    "upsample_matrices",
    "adjacency_matrices",
    "activation_layers",
    "template"
]

class Decoder3DMesh(nn.Module):
    
    def __init__(self,
        num_features: int,
        n_layers: int,
        n_nodes: int,
        num_conv_filters_dec: Sequence[int],
        cheb_polynomial_order: int,
        latent_dim: int,
        is_variational: bool,
        template,
        upsample_matrices: List[torch.Tensor],
        adjacency_matrices: List[torch.Tensor],
        activation_layers="ReLU"):

        super(Decoder3DMesh, self).__init__()

        self.n_nodes = n_nodes
        self.filters_dec = copy(num_conv_filters_dec)
        self.filters_dec.insert(0, num_features)
        self.filters_dec = list(reversed(self.filters_dec))

        self.K = cheb_polynomial_order

        self.matrices = {}
        A_edge_index, A_norm = self._build_adj_matrix(adjacency_matrices)
        self.matrices["A_edge_index"] = list(reversed(A_edge_index))
        self.matrices["A_norm"] = list(reversed(A_norm))
        self.matrices["upsample"] = list(reversed(upsample_matrices))

        self._n_features_before_z = self.matrices["upsample"][0].shape[1] * self.filters_dec[0]

        self._is_variational = is_variational
        self.latent_dim = latent_dim

        self.activation_layers = [activation_layers] * n_layers if isinstance(activation_layers, str) else activation_layers

        # Fully connected layer connecting the latent space layer with the first upsampling layer.
        self.dec_lin = torch.nn.Linear(self.latent_dim, self._n_features_before_z)

        self.layers = self._build_decoder()


    def _build_decoder(self):

        cheb_conv_layers = self._build_cheb_conv_layers(self.filters_dec, self.K)
        pool_layers = self._build_pool_layers(self.matrices["upsample"])
        activation_layers = self._build_activation_layers(self.activation_layers)

        decoder = ModuleDict()

        for i in range(len(cheb_conv_layers)):
            layer = f"layer_{i}"
            decoder[layer] = ModuleDict()
            decoder[layer]["activation_function"] = activation_layers[i]
            decoder[layer]["pool"] = pool_layers[i]
            decoder[layer]["graph_conv"] = cheb_conv_layers[i]

        return decoder


    def _build_pool_layers(self, upsample_matrices:Sequence[np.array]):

        '''
        downsample_matrices: list of matrices binary matrices
        '''

        pool_layers = ModuleList()
        for i in range(len(upsample_matrices)):
            pool_layers.append(Pool())
        return pool_layers


    def _build_activation_layers(self, activation_type:Union[str, Sequence[str]]):

        '''
        activation_type: string or list of strings containing the name of a valid activation function from torch.functional
        '''

        activation_layers = ModuleList()

        for i in range(len(activation_type)):
            activ_fun = getattr(torch.nn.modules.activation, activation_type[i])()
            activation_layers.append(activ_fun)

        return activation_layers


    def _build_cheb_conv_layers(self, n_filters, K):

        # Chebyshev convolutions (decoder)
        cheb_dec = torch.nn.ModuleList([ChebConv_Coma(n_filters[0], n_filters[1], K[0])])
        for i in range(1, len(n_filters)-1):
            conv_layer = ChebConv_Coma(n_filters[i], n_filters[i+1], K[i])
            cheb_dec.extend([conv_layer])

        cheb_dec[-1].bias = None  # No bias for last convolution layer
        return cheb_dec


    def _build_adj_matrix(self, adjacency_matrices):
        adj_edge_index, adj_norm = zip(*[
            ChebConv_Coma.norm(adjacency_matrices[i]._indices(), self.n_nodes[i])
            for i in range(len(self.n_nodes))
        ])
        return list(adj_edge_index), list(adj_norm)


    def forward(self, x):

        x = self.dec_lin(x)
        batch_size = x.shape[0] if x.dim() == 2 else 1
        x = x.reshape(batch_size, -1, self.layers["layer_0"]["graph_conv"].in_channels)

        for i, layer in enumerate(self.layers):
            
            if self.matrices["upsample"][i].device != x.device:
                self.matrices["upsample"][i] = self.matrices["upsample"][i].to(x.device)
            if self.matrices["A_edge_index"][i].device != x.device:
                self.matrices["A_edge_index"][i] = self.matrices["A_edge_index"][i].to(x.device)
            if self.matrices["A_norm"][i].device != x.device:
                self.matrices["A_norm"][i] = self.matrices["A_norm"][i].to(x.device)

            x = self.layers[layer]["activation_function"](x)
            x = self.layers[layer]["pool"](x, self.matrices["upsample"][i])
            x = self.layers[layer]["graph_conv"](x, self.matrices["A_edge_index"][i], self.matrices["A_norm"][i])

        return x
