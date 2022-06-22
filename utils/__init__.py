from subprocess import check_output
import shlex
import sys, os

# add the repository's root directory to Python path 
repo_root = check_output(shlex.split("git rev-parse --show-toplevel")).strip().decode('ascii')
sys.path.append(repo_root)
sys.path.append(os.path.join(repo_root, "utils"))
