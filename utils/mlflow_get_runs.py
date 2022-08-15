import mlflow

def list_experiments_by_name():
    options = [exp.name for exp in mlflow.list_experiments()]
    return options

def get_metrics_cols(df):
    metrics_cols = runs_df.columns[runs_df.columns.str.startswith("metrics")]
    return metrics_cols

def get_params_cols(df):
    params_cols = runs_df.columns[runs_df.columns.str.startswith("params")]
    return params_cols


def get_good_runs(exp_name="Cardiac - ED", metric='metrics.test_recon_loss',  metric_thres=1, cols_of_interest = ['experiment_id', 'run_id', 'params.latent_dim', 'metrics.test_recon_loss']):

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
    return good_runs_df
