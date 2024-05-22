import pytorch_lightning as pl
import yaml
from pytorch_lightning.loggers import MLFlowLogger

import mlflow
from mlflow.tracking import MlflowClient

from config.cli_args import CLI_args, overwrite_config_items
from config.load_config import load_yaml_config, to_dict

from utils.helpers import *
from utils.mlflow_helpers import get_mlflow_parameters, get_mlflow_dataset_params

from model import Model3D
from model.Model3D import Autoencoder3DMesh as Autoencoder
from model.lightning.AutoencoderLightningModule import AutoencoderLightning as LitVAE

import os
import argparse
import pprint

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme
from pytorch_lightning.callbacks import RichModelSummary
from pytorch_lightning.callbacks import RichProgressBar

from easydict import EasyDict

from IPython import embed

####### ####### ####### ####### ####### ####### ####### ####### ####### ####### ####### ####### 

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger()    

####### ####### ####### ####### ####### ####### ####### ####### ####### ####### ####### ####### 

progress_bar = RichProgressBar(
  theme=RichProgressBarTheme(
    description="green_yellow",
    progress_bar="green1",
    progress_bar_finished="green1",
    progress_bar_pulse="#6206E0",
    batch_progress="green_yellow",
    time="grey82",
    processing_speed="grey82",
    metrics="grey82",
  )
)

####### ####### ####### ####### ####### ####### ####### ####### ####### ####### ####### ####### 

CARDIAC_COMA_REPO = f"{os.environ['HOME']}/01_repos/CardiacCOMA"
ORIGINAL_MLRUNS_DIR = f"{CARDIAC_COMA_REPO}/replication_mlruns"      
ORIGINAL_MLRUNS_DIR = f"{CARDIAC_COMA_REPO}/mlruns"      

OUTPUT_MLRUNS_DIR = f"{CARDIAC_COMA_REPO}/final_runs_my_segmentation/finetuned_runs"
OUTPUT_MLRUNS_DIR = f"{CARDIAC_COMA_REPO}/final_runs_my_segmentation/original_mlruns"

# EXPERIMENT_ID = "3"

PROCRUSTES_ORIGINAL_FILE = f"{CARDIAC_COMA_REPO}/data/procrustes_transforms_35k.pkl"
MESHES_ORIGINAL_FILE = f"{CARDIAC_COMA_REPO}/data/LV_meshes_at_ED_35k.pkl"

MESHES_REPLICATION_FILE = f"{CARDIAC_COMA_REPO}/data/LV_meshes_at_ED_replication_25k.pkl"
PROCRUSTES_REPLICATION_FILE = f"{CARDIAC_COMA_REPO}/data/procrustes_transforms_LV_25k_new_meshes.pkl"

MESHES_REPLICATION_FILE = f"{CARDIAC_COMA_REPO}/data/LVED_all_Rodrigos_segmentation_60728_subjects.pkl"
PROCRUSTES_REPLICATION_FILE = f"{CARDIAC_COMA_REPO}/data/procrustes_transforms_LVED_all_Rodrigos_segmentation_60728_subjects.pkl"

def get_datamodules():
    
    '''
    Return a tuple with the discovery and replication datamodules
    '''
    
    dm_discovery = get_datamodule(EasyDict({
      "dataset": {
        "data_type": 'cardiac',
        "filename": MESHES_ORIGINAL_FILE,
        "preprocessing": {
            "procrustes": PROCRUSTES_ORIGINAL_FILE
        }
      },
      "sample_sizes": [10240, 2048, 2048, -1],
      "batch_size": 1024
      }),
      z_filename="latent_vector.csv",
      mse_filename="mse.csv"
    
    )
    
    logger.info("Loading replication DataModule...")
    dm_replication = get_datamodule(EasyDict({
      "dataset": {
        "data_type": 'cardiac',
        "filename": MESHES_REPLICATION_FILE,
        "preprocessing": {
            "procrustes": PROCRUSTES_REPLICATION_FILE
        }
      },
      "sample_sizes": [1, 1, 1, -1],
      "batch_size": 2048       
    }),
      z_filename="latent_vector_replication.csv",
      mse_filename="mse_replication.csv"
    )
    
    return dm_discovery, dm_replication

####### ####### ####### ####### ####### ####### ####### ####### ####### ####### ####### ####### 

def scipy_to_torch_sparse(scp_matrix):

    import numpy as np
    indices = np.vstack((scp_matrix.row, scp_matrix.col))
    i = torch.LongTensor(indices)
    values = scp_matrix.data
    v = torch.FloatTensor(values)
    shape = scp_matrix.shape

    sparse_tensor = torch.sparse.FloatTensor(i, v, torch.Size(shape))
    return sparse_tensor


def get_coma_matrices(config, dm, cache=True, from_cached=True):

    '''
    :param config: configuration Namespace, with a list called "network_architecture.pooling.parameters.downsampling_factors" as attribute.
    :param dm: a PyTorch Lightning datamodule, with attributes train_dataset.dataset.mesh_popu and train_dataset.dataset.mesh_popu.template
    :param cache: if True, will cache the matrices in a pkl file, unless this file already exists.
    :param from_cached: if True, will try to fetch the matrices from a previously cached pkl file.
    :return: a dictionary with keys "downsample_matrices", "upsample_matrices", "adjacency_matrices" and "n_nodes",
    where the first three elements are lists of matrices and the last is a list of integers.
    '''

    from utils import mesh_operations
    from utils.mesh_operations import Mesh

    mesh_popu = dm.train_dataset.dataset.meshes

    matrices_hash = hash(
        (tuple(config.network_architecture.pooling.parameters.downsampling_factors))) % 1000000

    template_point_cloud = mesh_popu.mean(axis=0)
    template_faces = np.array(pkl.load(open(config.dataset.template, "rb")))


    cached_file = f"data/cached/matrices/cardio_{matrices_hash}.pkl"

    if from_cached and os.path.exists(cached_file):
        A_t, D_t, U_t, n_nodes = pkl.load(open(cached_file, "rb"))
    else:
        template_mesh = Mesh(template_point_cloud, template_faces)
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
        "num_conv_filters_dec": convs.channels_dec,
        "cheb_polynomial_order": convs.parameters.polynomial_degree,
        "latent_dim": net.latent_dim,
        "is_variational": config.loss.regularization.weight != 0,
        "mode": "testing",
    }

    matrices = get_coma_matrices(config, dm, from_cached=False)
    coma_args.update(matrices)
    return coma_args


def get_datamodule(config, z_filename, mse_filename, perform_setup=True):

    '''

    '''
    
    from data.DataModules import CardiacMeshPopulationDataset
    from data.DataModules import DataModule
    
    # TODO: MERGE THESE TWO INTO ONE DATAMODULE CLASS
    if config.dataset.data_type.startswith("cardiac"):
        dataset_cls = CardiacMeshPopulationDataset
        dataset_args = {
            "meshes": pkl.load(open(config.dataset.filename, "rb")),
            "procrustes_transforms": config.dataset.preprocessing.procrustes
        }
    
    elif config.dataset.data_type.startswith("synthetic"):
        dataset_cls = SyntheticMeshPopulationDataset
        dataset_args = None

    dm = DataModule(
       dataset_cls, dataset_args,
       split_lengths=config.sample_sizes,
       batch_size=config.batch_size,
       z_filename=z_filename,
       mse_filename=mse_filename
    )


    if perform_setup:
        dm.setup()

    return dm
    
    
def get_torch_model(config, dm):
    
    # Initialize PyTorch model
    coma_args = get_coma_args(config, dm)

    dec_config = {k: v for k,v in coma_args.items() if k in Model3D.DECODER_ARGS}
    enc_config = {k: v for k,v in coma_args.items() if k in Model3D.ENCODER_ARGS}
    
    other_args = {
      "is_variational": config.loss.regularization.weight != 0,
      "template_mesh": coma_args["template"]
    }

    autoencoder = Autoencoder(enc_config, dec_config, other_args)
    
    return autoencoder
    
    
def get_lightning_module(config, dm):
   
    # Initialize PyTorch model
    coma_args = get_coma_args(config, dm)

    dec_config = {k: v for k,v in coma_args.items() if k in Model3D.DECODER_ARGS}
    enc_config = {k: v for k,v in coma_args.items() if k in Model3D.ENCODER_ARGS}
    
    other_args = {
      "is_variational": config.loss.regularization.weight != 0,
      "template_mesh": coma_args["template"]
    }

    autoencoder = Autoencoder(enc_config, dec_config, other_args)

    # Initialize PyTorch Lightning module
    model = LitVAE(autoencoder, config)

    return model


def load_model(config, chkpt_file, dm, exp_id="3"):
        
    ckpt = torch.load(chkpt_file, map_location=torch.device('cpu'))
    _model_pretrained_weights = ckpt["state_dict"]
    
    # Remove "model." prefix from state_dict's keys.
    model_pretrained_weights = {k.replace("model.", ""): v for k, v in _model_pretrained_weights.items()}
    
    logger.info("Creating model...")
    model = get_lightning_module(config, dm)    
    logger.info("Model created.")
    
    model.model.load_state_dict(model_pretrained_weights)
    
    optimizer_state = ckpt["optimizer_states"][0]
    model.optimizer.load_state_dict(optimizer_state)
    logger.info("Weights loaded from checkpoint")
    
    return model# , model_pretrained_weights


def get_checkpoint_locations(mlflow_uri, experiment_ids):
    
    import pandas as pd 
    import re 
    
    mlflow.set_tracking_uri(mlflow_uri)
    runs_df = mlflow.search_runs(experiment_ids=experiment_ids)
    
    if len(runs_df) == 0:
        raise ValueError(f"No runs found under URI {mlflow_uri} and experiment {experiment_ids}.")
    
    runs_df = runs_df[runs_df["metrics.test_recon_loss"] < 1.5]
    runs_df = runs_df.set_index(["experiment_id", "run_id"], drop=False)
    # print(runs_df)
    
    runs_df.artifact_uri = runs_df.artifact_uri.apply(lambda x: x.replace("/home/rodrigo/CISTIB/repos/", "/mnt/data/workshop/workshop-user1/output/"))
    runs_df.artifact_uri = runs_df.artifact_uri.apply(lambda x: x.replace("/home/home01/scrb/01_repos/", "/mnt/data/workshop/workshop-user1/output/"))    
    runs_df.artifact_uri = runs_df.artifact_uri.apply(lambda x: x.replace("/1/", "/3/"))
    
    checkpoint_locations = {}

    for i, row in runs_df.iterrows():
        
        artifact_uri = row.artifact_uri
        artifact_uri = artifact_uri.replace("file://", "")
        checkpoints = []
        
        try:
            basepath = os.path.join(artifact_uri, "restored_model_checkpoint")        
            checkpoints += [ os.path.join(basepath, x) for x in os.listdir(basepath)]
        except FileNotFoundError:
            pass
        
        try:
            basepath = os.path.join(os.path.dirname(artifact_uri), "checkpoints")        
            checkpoints += [ os.path.join(basepath, x) for x in os.listdir(basepath)]
        except FileNotFoundError:
            pass
        
        if len(checkpoints) > 1:
            # finetuned_runs_2/1/3b09d025cc1446f3a0c27f9b27b69340/checkpoints/epoch=129-val_recon_loss=0.4883_val_kld_loss=94.8229.ckpt.ckpt
            regex = re.compile(".*epoch=(.*)-.*.ckpt")
            epochs = []
            for chkpt_file in checkpoints:
                epoch = int(regex.match(chkpt_file).group(1))
                epochs.append(epoch)
            argmax = epochs.index(max(epochs))
            chkpt_file = checkpoints[argmax]
        elif len(checkpoints) == 1:
            chkpt_file = checkpoints[0]
        elif len(checkpoints) == 0:
            chkpt_file = None
            
        checkpoint_locations[row.run_id] = chkpt_file
      
    checkpoint_locations = { 
        k:v for k,v in checkpoint_locations.items() if v is not None
    }
    
    print(checkpoint_locations)
    return checkpoint_locations
    # runs = checkpoint_locations.keys()

    
def easydict_to_dict(edict):
    return {k: easydict_to_dict(v) if isinstance(v, EasyDict) else v for k, v in edict.items()}    
    
    
def save_to_yaml(edict, file_path):
    dict_obj = easydict_to_dict(edict)
    with open(file_path, 'w') as yaml_file:
        yaml.dump(dict_obj, yaml_file, default_flow_style=False)    
    

####### ####### ####### ####### ####### ####### ####### ####### ####### ####### ####### ####### 
def main(config, trainer_args, dm, dm2):

    '''

    '''
         
    model = load_model(config, config.chkpt_file, dm=dm)
    
    trainer = pl.Trainer(
        callbacks = [
            EarlyStopping(monitor="val_recon_loss", mode="min", patience=5, check_finite=True),
    #        RichModelSummary(max_depth=-1),
            ModelCheckpoint(monitor='val_recon_loss', save_top_k=1, filename='{epoch}-{val_recon_loss:.4f}_{val_kld_loss: .4f}.ckpt'),
            progress_bar
        ],
        gpus = 1,
        min_epochs=10, max_epochs=10000,
        logger = trainer_args.logger,
        precision = 16,
    )

    if config.mlflow:

        mlflow.pytorch.autolog(log_models=False)

        try:
            exp_id = mlflow.create_experiment(
                config.mlflow.experiment_name, 
                artifact_location=config.mlflow.artifact_location
            )
        except:
            # If the experiment already exists, we can just retrieve its ID
            experiment = mlflow.get_experiment_by_name(config.mlflow.experiment_name)
            exp_id = experiment.experiment_id

        logging.info(f"Experiment ID: {exp_id} ({config.mlflow.experiment_name})")
        logging.info(f"Run ID: {trainer.logger.run_id}")

        run_info = {
            "run_id": trainer.logger.run_id,
            "experiment_id": exp_id,
            "run_name": config.mlflow.run_name,
        }

        mlflow.start_run(**run_info)
    
        save_to_yaml(config, "config.yaml")        
        mlflow.log_artifact("config.yaml")
        
        mlflow_params = get_mlflow_parameters(config)
        mlflow_dataset_params = get_mlflow_dataset_params(config)
        mlflow_params.update(mlflow_dataset_params)

        mlflow.log_params(mlflow_params)
        # mlflow.log_params(config.additional_mlflow_params)

    #trainer.fit(model, datamodule=dm_discovery)
    #trainer.test(model, datamodule=dm_discovery) # Generates metrics for the full test dataset
    trainer.predict(model=model, datamodule=dm) # Generates figures for a few samples
    # trainer.predict(model=model, datamodule=dm2) # Generates figures for a few samples
    mlflow.end_run()

    
####### ####### ####### ####### ####### ####### ####### ####### ####### ####### ####### ####### 
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Pytorch Trainer for Convolutional Mesh Autoencoders",
        argument_default=argparse.SUPPRESS
    )

    #to avoid a little bit of boilerplate
    for k, v in CLI_args.items():
        parser.add_argument(*k, **v)
    
    parser.add_argument("--original_mlflow_uri", "--mlflow-uri", dest="original_mlflow_uri", type=str)
    parser.add_argument("--output_mlflow_uri", dest="output_mlflow_uri", type=str)
    parser.add_argument("--experiment_id", "--exp_id", dest="experiment_id", nargs="+", type=str)
    parser.add_argument("--base-run", "--base_run", type=str)
    
    # parser.add_argument("--checkpoint_path", type=str )
    # parser.add_argument("--original_config_file")
            
       
    # adding arguments specific to the PyTorch Lightning trainer.
    parser = pl.Trainer.add_argparse_args(parser)

    args = parser.parse_args()

    ###
    ORIGINAL_MLRUNS_DIR = args.original_mlflow_uri
    EXPERIMENT_ID = args.experiment_id[0]
    OUTPUT_MLRUNS_DIR = args.output_mlflow_uri
    
    checkpoint_locations = get_checkpoint_locations(
        mlflow_uri=ORIGINAL_MLRUNS_DIR, 
        experiment_ids=[EXPERIMENT_ID]
    )
    
    compatible_runs = sorted(checkpoint_locations.keys())
   
    print(compatible_runs)
    print(len(compatible_runs))
    # compatible_runs = [x for x in os.listdir(f"{ORIGINAL_MLRUNS_DIR}/{EXPERIMENT_ID}") if x.startswith(args.base_run)]
    
    if args.base_run != "all":
        
        compatible_runs = [x for x in checkpoint_locations if x.startswith(args.base_run)]    
    
        if len(compatible_runs) == 0:
            raise ValueError(f"The prefix {args.base_run} does not correspond to any existing run.")
        elif len(compatible_runs) >= 2:
            raise ValueError(f"The prefix {args.base_run} as more than one compatible run ({', '.join(compatible_runs)}) ")
        else:
            runs = compatible_runs
            logger.info(f"The run {runs[0]} will be processed")
            # exit()
    else:
        # use all runs that have a checkpoint in the specified MLflow folder
        runs = compatible_runs
    
    ##################################################################
    
    logger.info("Loading discovery DataModule...")

    MESHES_ALL_FILE = f"{CARDIAC_COMA_REPO}/data/LV_meshes_at_ED_55k.pkl"
    PROCRUSTES_ALL_FILE = f"{CARDIAC_COMA_REPO}/data/procrustes_transforms_LVED_55k.pkl"
    
    MESHES_ALL_FILE = MESHES_REPLICATION_FILE
    PROCRUSTES_ALL_FILE = PROCRUSTES_REPLICATION_FILE
    
    # MESHES_ALL_FILE = MESHES_ORIGINAL_FILE
    # PROCRUSTES_ALL_FILE = PROCRUSTES_ORIGINAL_FILE
    
    #dm_discovery = get_datamodule(EasyDict({
    #  "dataset": {
    #    "data_type": 'cardiac',
    #    "filename": MESHES_ALL_FILE,
    #    "preprocessing": {
    #        "procrustes": PROCRUSTES_ALL_FILE
    #    }
    #  },
    #  "sample_sizes": [1, 1, 1, -1],
    #  "batch_size": 1024
    #  }),
    #  z_filename="latent_vector.csv",
    #  mse_filename="mse.csv",
    #)
    
    dm_discovery, dm_replication = get_datamodules()         
    
    ##################################################################
    
    for run in runs:
        
        ### Load configuration
        
        original_config_file = f"{ORIGINAL_MLRUNS_DIR}/{EXPERIMENT_ID}/{run}/artifacts/config.yaml"
        if not os.path.exists(original_config_file):
            logger.error("Config not found: " + original_config_file)
            continue
    
        ref_config = load_yaml_config(original_config_file)
    
        try:
            config_to_replace = args.config
            config = overwrite_config_items(ref_config, config_to_replace)
        except:
            # If there are no elements to replace
            config = ref_config
            
        #TOFIX: args contains other arguments that do not correspond to the trainer
        trainer_args = args
    
        # Override MLflow URI
        config.mlflow.tracking_uri = OUTPUT_MLRUNS_DIR
        
        if args.disable_mlflow_logging:
            config.mlflow = None
                
        if config.mlflow:
    
            if config.mlflow.experiment_name is None:
                config.mlflow.experiment_name = "rbonazzola - Default"
    
            trainer_args.logger = MLFlowLogger(
                tracking_uri=config.mlflow.tracking_uri,
                experiment_name=config.mlflow.experiment_name,
                artifact_location=config.mlflow.artifact_location
            )
    
            mlflow.set_tracking_uri(config.mlflow.tracking_uri)
    
        else:
            trainer_args.logger = None
           
        config.chkpt_file = checkpoint_locations[run]
        config.original_run_id = run
        
        if args.show_config or args.dry_run:
            pp = pprint.PrettyPrinter(indent=2, compact=True)
            pp.pprint(to_dict(config))
            if args.dry_run:
                exit()        
            
        logging.info(f"Checkpoint file found at {config.chkpt_file}")
        
        try:
            main(config, trainer_args, dm=dm_discovery)
        except FileNotFoundError:
            pass
