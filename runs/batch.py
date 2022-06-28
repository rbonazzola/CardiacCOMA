'''
Script to launch a batch of training executions
'''

import torch

CLI_args = {
    ("--log10_lr",): {
        "help": "-log10 of learning rate",
        "nargs": "+", "type": int
    },
    ("--latent_dim",): {
        "help": "Dimension of the latent space",
        "nargs": "+", "type": int
    },
    ("--run_again_if_present",): {
        "help": "If set to true, will run again the hyperparameter configuration even if it's already present.",
        "action": "store_true"
    },
    ("--activation_function",): {
        "help": "If set to true, will run again the hyperparameter configuration even if it's already present."
    }

}

def ():


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Pytorch Trainer for Convolutional Mesh Autoencoders",
        argument_default=argparse.SUPPRESS
    )

    # to avoid a little bit of boilerplate
    for k, v in CLI_args.items():
        parser.add_argument(*k, **v)

    # adding arguments specific to the PyTorch Lightning trainer.
    parser = pl.Trainer.add_argparse_args(parser)

    args = parser.parse_args()
