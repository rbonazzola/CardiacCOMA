import torch
import os
import sys; sys.path.append("..")

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from pytorch_lightning.callbacks import RichModelSummary
from data.DataModules import CardiacMeshPopulationDM
from data.SyntheticDataModules import SyntheticMeshesDM
from utils import mesh_operations
from utils.mesh_operations import Mesh

import pickle as pkl

def scipy_to_torch_sparse(scp_matrix):

    import numpy as np
    indices = np.vstack((scp_matrix.row, scp_matrix.col))
    i = torch.LongTensor(indices)
    values = scp_matrix.data
    v = torch.FloatTensor(values)
    shape = scp_matrix.shape

    sparse_tensor = torch.sparse.FloatTensor(i, v, torch.Size(shape))
    return sparse_tensor


def get_datamodule(config, perform_setup=True):

    '''

    '''

    # TODO: MERGE THESE TWO INTO ONE DATAMODULE CLASS
    if config.dataset.data_type.startswith("cardiac"):
        dm = CardiacMeshPopulationDM(cardiac_population=data, batch_size=config.batch_size)
    elif config.dataset.data_type.startswith("synthetic"):
        dm = SyntheticMeshesDM(
            batch_size=config.batch_size,
            data_params=config.dataset.parameters.__dict__,
            preprocessing_params=config.dataset.preprocessing
        )

    if perform_setup:
        dm.setup()

    return dm


def get_coma_matrices(config, dm, cache=True, from_cached=True):
    '''
    :param config: configuration Namespace, with a list called "network_architecture.pooling.parameters.downsampling_factors" as attribute.
    :param dm: a PyTorch Lightning datamodule, with attributes train_dataset.dataset.mesh_popu and train_dataset.dataset.mesh_popu.template
    :param cache: if True, will cache the matrices in a pkl file, unless this file already exists.
    :param from_cached: if True, will try to fetch the matrices from a previously cached pkl file.
    :return: a dictionary with keys "downsample_matrices", "upsample_matrices", "adjacency_matrices" and "n_nodes",
    where the first three elements are lists of matrices and the last is a list of integers.
    '''

    mesh_popu = dm.train_dataset.dataset.mesh_popu
    matrices_hash = hash(
        (mesh_popu._object_hash, tuple(config.network_architecture.pooling.parameters.downsampling_factors))) % 1000000
    cached_file = f"data/cached/matrices/{matrices_hash}.pkl"

    if from_cached and os.path.exists(cached_file):
        A_t, D_t, U_t, n_nodes = pkl.load(open(cached_file, "rb"))
    else:
        template_mesh = Mesh(mesh_popu.template.vertices, mesh_popu.template.faces)
        M, A, D, U = mesh_operations.generate_transform_matrices(
            template_mesh, config.network_architecture.pooling.parameters.downsampling_factors,
        )
        n_nodes = [len(M[i].v) for i in range(len(M))]
        A_t, D_t, U_t = ([scipy_to_torch_sparse(x).float() for x in X] for X in (A, D, U))
        if cache:
            os.makedirs(os.path.dirname(cached_file), exist_ok=True)
            with open(cached_file, "wb") as ff:
                pkl.dump((A_t, D_t, U_t, n_nodes), ff)

    return {
        "downsample_matrices": D_t,
        "upsample_matrices": U_t,
        "adjacency_matrices": A_t,
        "n_nodes": n_nodes,
        "template": template_mesh
    }


def get_coma_args(config, dm):

    net = config.network_architecture

    convs = net.convolution
    coma_args = {
        "num_features": net.n_features,
        "n_layers": len(convs.channels_enc),  # REDUNDANT
        "num_conv_filters_enc": convs.channels_enc,
        "num_conv_filters_dec_c": convs.channels_dec_c,
        "num_conv_filters_dec_s": convs.channels_dec_s,
        "cheb_polynomial_order": convs.parameters.polynomial_degree,
        "latent_dim_content": net.latent_dim_c,
        "latent_dim_style": net.latent_dim_s,
        "is_variational": config.loss.regularization.weight != 0,
        "mode": "testing",
        "n_timeframes": config.dataset.parameters.T,
        "phase_input": net.phase_input,
        "z_aggr_function": net.z_aggr_function
    }

    matrices = get_coma_matrices(config, dm, from_cached=False)
    coma_args.update(matrices)
    return coma_args


def get_lightning_module(config, dm):

    # Initialize PyTorch model
    coma_args = get_coma_args(config, dm)

    if config.only_decoder:

        from models.Model4D import DecoderTemporalSequence, DECODER_C_ARGS, DECODER_S_ARGS
        from models.lightning.DecoderLightningModule import TemporalDecoderLightning

        dec_c_config = {k: v for k,v in coma_args.items() if k in DECODER_C_ARGS}
        dec_s_config = {k: v for k,v in coma_args.items() if k in DECODER_S_ARGS}

        decoder = DecoderTemporalSequence(
            dec_c_config, dec_s_config,
            phase_embedding_method="exp",
            n_timeframes=config.dataset.parameters.T
        )

        model = TemporalDecoderLightning(decoder, config)

    elif config.only_encoder:

        from models.Model4D import EncoderTemporalSequence, ENCODER_ARGS
        from models.lightning.EncoderLightningModule import TemporalEncoderLightning

        enc_config = {k: v for k, v in coma_args.items() if k in ENCODER_ARGS}

        encoder = EncoderTemporalSequence(
            enc_config, z_aggr_function=config.network_architecture.z_aggr_function,
            n_timeframes=config.dataset.parameters.T
        )

        model = TemporalEncoderLightning(encoder, config)

    else:
        from models.Model4D import AutoencoderTemporalSequence
        from models.lightning.ComaLightningModule import CoMA
        autoencoder = AutoencoderTemporalSequence(**coma_args)
        # Initialize PyTorch Lightning module
        model = CoMA(autoencoder, config)

    return model


def get_lightning_trainer(trainer_args):

    # trainer
    trainer_kwargs = {
        "callbacks": [
            EarlyStopping(monitor="val_loss", mode="min", patience=3),
            RichModelSummary(max_depth=-1)
        ],
        "gpus": [trainer_args.gpus],
        "auto_select_gpus": trainer_args.auto_select_gpus,
        "min_epochs": trainer_args.min_epochs, "max_epochs": trainer_args.max_epochs,
        "auto_scale_batch_size": trainer_args.auto_scale_batch_size,
        "logger": trainer_args.logger,
        "precision": trainer_args.precision,
        "overfit_batches": trainer_args.overfit_batches,
        "limit_test_batches": trainer_args.limit_test_batches
    }

    try:
        trainer = pl.Trainer(**trainer_kwargs)
    except:
        trainer_kwargs["gpus"] = None
        trainer = pl.Trainer(**trainer_kwargs)
    return trainer


def get_dm_model_trainer(config, trainer_args):
    '''
    Returns a tuple of (PytorchLightning datamodule, PytorchLightning model, PytorchLightning trainer)
    '''

    # LOAD DATA
    dm = get_datamodule(config)
    model = get_lightning_module(config, dm)
    trainer = get_lightning_trainer(trainer_args)

    return dm, model, trainer
