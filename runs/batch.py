import shlex
#import torch
import subprocess
import pytorch_lightning as pl
import argparse

'''
Script to launch a batch of training executions
'''

pyvista_commands = '''
export DISPLAY=:99.0
export PYVISTA_OFF_SCREEN=true
export PYVISTA_USE_IPYVTK=true
Xvfb :99 -screen 0 1024x768x24 > /dev/null 2>&1
'''

def grid_search(args):

    for latent_dim in args.latent_dim:
        for lr in args.learning_rate:
            for training_sample_size in args.training_sample_size:
                command = f'''python main.py \
--n_channels_enc {' '.join(args.n_channels)} \
--n_channels_dec {' '.join(args.n_channels)} \
--latent_dim {latent_dim} \
--batch_size {args.batch_size} \
--learning_rate {lr} \
--partition_lengths {training_sample_size} 1024 1024 -1 \
--auto_select_gpus \
--precision 16 \
--mlflow_experiment test \
--gpus {args.device}'''
                print(shlex.split(command))
                subprocess.Popen(shlex.split(command))
  

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Pytorch Trainer for Convolutional Mesh Autoencoders",
        argument_default=argparse.SUPPRESS
    )

    parser.add_argument("--learning_rate", nargs="+", type=str, default=["1e-4"])
    parser.add_argument("--batch_size", type=str, default=32)
    parser.add_argument("--latent_dim", nargs="+", type=str, default=[8])
    parser.add_argument("--n_channels", nargs="+", type=str, default=["16", "32", "64", "128"])
    parser.add_argument("--activation_function", nargs="+", type=str, default="ReLU")
    parser.add_argument("--reduction_factors", nargs="+", type=int, default=[2]*4)
    parser.add_argument("--training_sample_size", nargs="+", type=str, default=[1280, 2560, 5120])
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--dry-run", "--dry_run", action="store_true")
    #parser.add_argument()

    # adding arguments specific to the PyTorch Lightning trainer.
    parser = pl.Trainer.add_argparse_args(parser)

    args = parser.parse_args()

    subprocess.Popen(shlex.split(pyvista_commands))

    grid_search(args)
