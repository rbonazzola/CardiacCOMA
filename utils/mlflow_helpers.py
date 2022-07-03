from subprocess import check_output

def get_mlflow_parameters(config):
    loss = config.loss
    net = config.network_architecture
    loss_params = {
        "w_kl": loss.regularization.weight,
    }
    net_params = {
        "latent_dim": net.latent_dim,
        "convolution_type": net.convolution.type,
        "n_channels_enc": net.convolution.channels_enc,
        "n_channels_dec": net.convolution.channels_dec,
        "reduction_factors": net.pooling.parameters.downsampling_factors,
    }

    mlflow_parameters = {
        "platform": check_output(["hostname"]).strip().decode(),
        **loss_params,
        **net_params,
    }

    return mlflow_parameters


###
def get_mlflow_dataset_params(config):
    '''
    Returns a dictionary containing the dataset parameters, to be logged to MLflow.
    '''
    d = config.dataset


    mlflow_dataset_params = {
        "dataset_type": d.data_type,
    }

    if mlflow_dataset_params["dataset_type"] == "synthetic":
        mlflow_dataset_params.update({
          "dataset_max_static_amplitude": d.parameters.amplitude_static_max,
          "dataset_max_dynamic_amplitude": d.parameters.amplitude_dynamic_max,
          "dataset_n_timeframes": d.parameters.T,
          "dataset_freq_max": d.parameters.freq_max,
          "dataset_l_max": d.parameters.l_max,
          "dataset_resolution": d.parameters.mesh_resolution,
          "dataset_complexity_c": (d.parameters.l_max + 1) ** 2,
          "dataset_complexity_s": ((d.parameters.l_max + 1) ** 2) * d.parameters.freq_max,
          "dataset_complexity": ((d.parameters.l_max + 1) ** 2) * (d.parameters.freq_max + 1),
          "dataset_random_seed": d.parameters.random_seed,
          "dataset_template": "icosphere",  # TODO: add this as parameter in the configuration
          "dataset_center_around_mean": d.preprocessing.center_around_mean
        })

    mlflow_dataset_params.update({
        "n_training": config.sample_sizes[0],
        "n_validation": config.sample_sizes[1],
        "n_test": config.sample_sizes[2],
        "dataset_center_around_mean": False
    })

    return mlflow_dataset_params


def print_auto_logged_info(r):

    tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
    artifacts = [f.path for f in MlflowClient().list_artifacts(r.info.run_id, "model")]
    print("run_id: {}".format(r.info.run_id))
    print("artifacts: {}".format(artifacts))
    print("params: {}".format(r.data.params))
    print("metrics: {}".format(r.data.metrics))
    print("tags: {}".format(tags))
