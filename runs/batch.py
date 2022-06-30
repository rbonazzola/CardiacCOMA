#import torch
import argparse

def grid_search(args):

  print(f"python main.py \
    --n_channels_enc {' '.join(args.n_channels)} \
    --n_channels_dec {' '.join(args.n_channels)} \
    --latent_dim {args.latent_dim} \
    --batch_size 256 \
    --learning_rate 1e-${args.log_learning_rate} \
    --gpus {args.device}")


  

if __name__ == "__main__":

  parser = argparse.ArgumentParser()

  parser.add_argument("--log_learning_rate", nargs="+", type=float)
  parser.add_argument("--latent_dim", nargs="+", type=str, default=8)
  parser.add_argument("--n_channels", nargs="+", type=str, default=["16", "32", "64", "128"])
  parser.add_argument("--activation_function", nargs="+", type=str, default="ReLU")
  parser.add_argument("--reduction_factors", nargs="+", type=int, default=[2]*4)
  parser.add_argument("--device", type=int, default=0)
  parser.add_argument("--dry-run", "--dry_run", action="store_true")
  #parser.add_argument()

  args = parser.parse_args()
    
  grid_search(args)
