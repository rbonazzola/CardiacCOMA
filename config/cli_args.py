import argparse
from .load_config import load_yaml_config, to_dict, flatten_dict, rsetattr, rgetattr
from copy import deepcopy


class ArgumentAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        rsetattr(namespace, self.dest, values)


class kwargs_append_action(argparse.Action):
    """
    argparse action to split an argument into KEY=VALUE form
    on append to a dictionary.
    """

    def __call__(self, parser, args, values, option_string=None):
        try:
            d = dict(map(lambda x: x.split('='),values))
        except ValueError as ex:
            raise argparse.ArgumentError(self, f"Could not parse argument \"{values}\" as k1=v1 k2=v2 ... format")
        setattr(args, self.dest, d)


network_architecture_args = {
    ("--n_channels",): {
        "help": "Number of channels (feature maps). If the rest of the --n_channels_* arguments are not provided, it will assign these numbers to the encoder, content decoder and style decoder.",
        "nargs": "+", "type": int,
        "action": ArgumentAction,
        "dest": "config.network_architecture.convolution.channels"},
    ("--reduction_factors",): {
        "help": "Decimation factors for the mesh",
        "nargs": "+", "type": int,
        "action": ArgumentAction,
        "dest": "config.network_architecture.pooling.downsampling_factors"},
    ("--n_channels_enc",): {
        "help": "Number of channels (feature maps) in the encoder, from input to the most hidden layer.",
        "nargs": "+", "type": int,
        "action": ArgumentAction,
        "dest": "config.network_architecture.convolution.channels_enc"},
    ("--n_channels_dec",): {
        "help": "Number of channels (feature maps) in the style decoder, from the most hidden layer to the output.",
        "nargs": "+", "type": int,
        "action": ArgumentAction,
        "dest": "config.network_architecture.convolution.channels_dec"},
    ("--latent_dim",): {
        "help": "Dimension of the latent space",
        "type": int,
        "action": ArgumentAction,
        "dest": "config.network_architecture.latent_dim"},
    ("--activation_function",): {
        "help": "Activation functions to be used",
        "nargs": "+", "type": str,
        "action": ArgumentAction,
        "dest": "config.network_architecture.activation_function"},
    ("--polynomial_degree",): {
        "help": "Chebyshev polynomial degree",
        "nargs": "+", "type": int,
        "action": ArgumentAction,
        "dest": "config.network_architecture.convolution.parameters.polynomial_degree"},
    ("--only_decoder",): {
        "help": "Flag to run only the decoder",
        "action": "store_true"},
    ("--only_encoder",): {
        "help": "Flag to run only the encoder",
        "action": "store_true"},

    #("--phase_input" ): {
    #    "help": "If this flag is set, the phase embedding is not applied to the input mesh coordinates.",
    #    "default": True,
    #    "dest": "config.network_architecture.phase_input",
    #    "action": argparse.BooleanOptionalAction}
}

loss_args = {
    ("--reconstruction_loss_type",): {
        "help": "Type of reconstruction loss",
        "dest": "config.loss.reconstruction_c.type",
        "type": str,
        "action": ArgumentAction},
    ("--w_kl",): {
        "help": "weight of KL term",
        "dest": "config.loss.regularization.weight",
        "type": float,
        "action": ArgumentAction},
}

cardiac_dataset_args = {
    ("--cardiac_dataset.meshes_file",): {
        "help": "",
        "dest": "config.cardiac_dataset.meshes_file",
        "type": str,
        "action": ArgumentAction},
    ("--cardiac_dataset.procrustes_transforms_file", ): {
        "help": "Pickle binary file containing the rotation and traslation matrix from generalized Procrustes analysis",
        "dest": "config.cardiac_dataset.procrustes_transforms_file",
        "type": str,
        "action": ArgumentAction},
}


dataset_args = {
    ("--synthetic_dataset.amplitude_static_max",): {
        "help": "" ,
        "dest": "config.dataset.parameters.amplitude_static_max" , 
        "type": float,
        "action": ArgumentAction},
    ("--dataset.amplitude_dynamic_max",): {
        "help": "",
        "dest": "config.dataset.parameters.amplitude_dynamic_max" , 
        "type": float,
        "action": ArgumentAction},
    ("--dataset.N_subjects",): {
        "help": "",
        "dest": "config.dataset.parameters.N" , 
        "type": int,
        "action": ArgumentAction},
    ("--dataset.freq_max",): {
        "help": "",
        "dest": "config.dataset.parameters.freq_max" , 
        "type": int,
        "action": ArgumentAction},
    ("--dataset.l_max",): {
        "help": "",
        "dest": "config.dataset.parameters.l_max" , 
        "type": int,
        "action": ArgumentAction},
    ("--dataset.mesh_resolution",): {
        "help": "",
        "dest": "config.dataset.parameters.mesh_resolution" , 
        "type": int,
        "action": ArgumentAction},
    ("--dataset.center_around_mean_shape",): {
        "help": "Not working! Always sets this value to True.",
        "dest": "config.dataset.preprocessing.center_around_mean" , 
        "type": bool,
        "action": ArgumentAction}
}

training_args = {
    ("--learning_rate", "-lr",): {
        "help": "Learning rate",
        "dest": "config.optimizer.parameters.lr",
        "type": float,
        "action": ArgumentAction},
    ("--batch_size",): {
        "help": "Training batch size. If provided will overwrite the batch size from the configuration file.",
        "dest": "config.batch_size",
        "type": int,
        "action": ArgumentAction},
    ("--partition_lengths", "--partition-lengths"): {
        "nargs":"+", "help": "List of 4 integers representing the number of samples (fraction of samples) to be used for training, test, validation and prediction. If the last one if -1, the full dataset is taken.",
        "dest": "config.sample_sizes",
        "type": int,
        "action": ArgumentAction},
}

mlflow_args = {
    ("--disable_mlflow_logging",): {
        "help": "Set this flag if you don't want to log the run's data to MLflow.",
        "default": False,
        "action": "store_true"},
    ("--mlflow_experiment",): {
        "help": "MLflow experiment's name",
        "dest": "config.mlflow.experiment_name",
        "action": ArgumentAction
    },
    ("--additional_mlflow_params",): {
        "nargs": '+',
        "required": False,
        "dest": "config.additional_mlflow_params",
        "action": kwargs_append_action,
        "metavar": "KEY=VALUE",
        "help": "Add additional key/value params to MLflow."
    },
    ("--additional_mlflow_tags",): {
        "nargs": '+',
        "required": False,
        "dest": "config.additional_mlflow_tags",
        "action": kwargs_append_action,
        "metavar": "KEY=VALUE",
        "help": "Add additional key/value tags to MLflow."
    }
}

#   ("--mlflow_config",): {
#       "action": LoadYamlConfig,
#       "help": "YAML configuration file containing information to log model information to MLflow.",
#       "dest": "config.mlflow"},

#########################################################################
#### Put all the arguments together
#########################################################################

CLI_args = {
    ("-c", "--conf",): {
        "help": "Path of a YAML configuration file to be used as a reference configuration.",
        "default": "config_files/config.yaml",
        "dest": "yaml_config_file"
    },
    ** network_architecture_args,
    ** loss_args,
    ** training_args,
    ** cardiac_dataset_args,
    ** dataset_args,
    ** mlflow_args,
    ("--show_config",): {
        "default": False,
        "action": "store_true",
        "help": "Display run's configuration"
    },
    ("--dry-run", "--dry_run"): {
        "dest": "dry_run",
        "default": False,
        "action": "store_true",
        "help": "Dry run: just prints out the parameters of the execution but performs no training.",
    },
    ("--log_computational_graph",): {
        "default": False,
        "action": "store_true",
        "help": "If True, will log the computational graph as an artifact (not fully functional due to limitations of the torchviz library)"
    }
}


def overwrite_config_items(ref_config, config_to_replace):
    '''
    params:
    :: ref_config ::
    :: config_to_replace ::
    '''

    config = deepcopy(ref_config)

    for k, v in flatten_dict(to_dict(config_to_replace)).items():
        rsetattr(config, k, v)

    return config
