import mlflow
import os
from typing import List, Union
import pandas as pd
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

    
def list_experiments_by_name():
    options = [exp.name for exp in mlflow.list_experiments()]
    return options


def get_metrics_cols(df):
    metrics_cols = df.columns[df.columns.str.startswith("metrics")]
    return metrics_cols


def get_params_cols(df):
    params_cols = df.columns[df.columns.str.startswith("params")]
    return params_cols


def get_runs_df(exp_name="Cardiac - ED", sort_by="metrics.test_recon_loss", only_finished=True):

  # runs_list = mlflow.search_runs(experiment_ids=[exp_id], output_format="list")
  exp_id = mlflow.get_experiment_by_name(exp_name).experiment_id
  runs_df = mlflow.search_runs(experiment_ids=[exp_id],)

  # Keep only the runs that ended successfully
  if only_finished:
      runs_df = runs_df[runs_df.status == "FINISHED"].reset_index(drop=True)

  # Use experiment ID and run ID as indices
  runs_df = runs_df.set_index(["experiment_id", "run_id"])
  runs_df = runs_df.sort_values(by=sort_by)
  return runs_df


def get_good_runs(
    exp_name="Cardiac - ED",
    metric: str = 'metrics.test_recon_loss',
    metric_thres: float = 1,
    cols_of_interest = ['experiment_id', 'run_id', 'params.latent_dim', 'metrics.test_recon_loss'],
    output_file: Union[None, str] = None
  ) -> pd.DataFrame:

    '''
    Returns a DataFrame with the runs that satisfy a performance criterion.

    Parameters:
        exp_name (str): MLflow experiment name.
        metric (str):
        metric_thres (float): 
        cols_of_interest (List[str]):
    '''

    exp_id = mlflow.get_experiment_by_name(exp_name).experiment_id
    runs_df = mlflow.search_runs(experiment_ids=[exp_id],)
    good_runs_df = runs_df[runs_df[metric] < metric_thres][cols_of_interest]
    good_runs_df = good_runs_df.sort_values(metric).reset_index(drop=True)

    if output_file is not None:
        good_runs_df.to_csv(output_file, header=True, index=False)

    return good_runs_df


def list_artifacts(
      experiment_id: str, run_id: str, path=".", 
      recursive=True, client=mlflow.tracking.MlflowClient()
    ) -> List[mlflow.entities.file_info.FileInfo]:
    
    '''
    Lists all artifacts available for the given MLflow run
    
    Examples:
        list_artifacts(exp_id, run_id, path="GWAS")
        list_artifacts(exp_id, run_id, path="GWAS/summaries")
    '''
    
    from copy import deepcopy
    
    artifacts = client._tracking_client.list_artifacts(run_id, path=path)
    
    if not recursive or all([not x.is_dir for x in artifacts]):
        return artifacts
    
    else:
        kk = deepcopy(artifacts)
        for artifact in kk:
            if artifact.is_dir:
                more_artifacts = list_artifacts(experiment_id, run_id, artifact.path)                
                artifacts.extend(more_artifacts)
        return artifacts


def get_model_pretrained_weights(runs_df, experiment_id, run_id):
    
    '''
    
    '''
    
    run_info = runs_df.loc[experiment_id, run_id].to_dict()
    artifact_uri = run_info["artifact_uri"].replace("file://", "")   
 
    #TODO: modify this
    try:
        chkpt_dir = os.path.join(artifact_uri, "restored_model_checkpoint")
        os.listdir(chkpt_dir)
    except:
        chkpt_dir = os.path.join(os.path.dirname(artifact_uri), "checkpoints")
    
    chkpt_file = os.path.join(chkpt_dir, os.listdir(chkpt_dir)[0])
    
    model_pretrained_weights = torch.load(chkpt_file, map_location=torch.device('cpu'))["state_dict"]
    
    # Remove "model." prefix from state_dict's keys.
    _model_pretrained_weights = {k.replace("model.", ""): v for k, v in model_pretrained_weights.items()}
    # print(_model_pretrained_weights)
    return _model_pretrained_weights
    

def get_significant_loci(
    runs_df,
    experiment_id, run_id, 
    p_threshold=1e-8, 
    client=mlflow.tracking.MlflowClient()
) -> pd.DataFrame:
    
    '''    
    Returns a DataFrame with the loci that have a stronger p-value than a given threshold
    '''
    
    LOCUS_NAMES = {
      "chr2_108": "TTN",
      "chr6_78": "PLN",
      "chr6_79": "PLN",
      "chr17_27": "GOSR2"
    }   
 
    def get_phenoname(path):
        
        filename = os.path.basename(path)
        phenoname = filename.split("__")[0]
        return phenoname
    
    run_info = runs_df.loc[experiment_id, run_id].to_dict()
    artifact_uri = run_info["artifact_uri"].replace("file://", "")    
    
    summaries_fileinfo = client._tracking_client.list_artifacts(run_id, path="GWAS/summaries")
    if len(summaries_fileinfo) == 0:
        return pd.DataFrame(columns=["run", "pheno", "region"])
    
    region_summaries = {get_phenoname(x.path): os.path.join(artifact_uri, x.path) for x in summaries_fileinfo}
    dfs = [pd.read_csv(path).assign(pheno=pheno) for pheno, path in region_summaries.items()]
    df = pd.concat(dfs)
    df['locus_name'] = df.apply(lambda row: LOCUS_NAMES.get(row["region"], "Unnamed"), axis=1)
    df = df.set_index(["pheno", "region"])    
    
    return df[df.P < p_threshold].sort_values(by="P")


def summarize_loci_across_runs(runs_df: pd.DataFrame):

    '''
    Parameters: run_ids
    Return: pd.DataFrame with ["count", "min_P"].
    '''

    # run_ids = sorted([x[1] for x in runs_df[runs_df["metrics.test_recon_loss"] < RECON_LOSS_THRES].index])
    run_ids = sorted([x[1] for x in runs_df.index])

    all_signif_loci = pd.concat([
      get_significant_loci(runs_df, "1", run).\
        assign(run=run).\
        reset_index().\
        set_index(["run", "pheno", "region"]) 
      for run in run_ids
    ])

    df = all_signif_loci.\
      groupby(["region", "locus_name"]).\
      aggregate({"CHR":"count", "P": "min"}).\
      rename({"CHR":"count", "P":"min_P"}, axis=1).\
      sort_values("count", ascending=False)    
    
    return df

# class ComaRun(mlflow.entities.Run):
#     
#     def __init__(self, exp_id, run_id):
#         
#         super
#         
#     def model(self):
#         
# 
#     def gwas_summary(self):
#         
#         return [os.path.join(artifact_uri, x.path) for x in client._tracking_client.list_artifacts(run_id, path="GWAS/summaries")]
