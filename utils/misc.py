import shlex
from subprocess import check_output

def get_current_commit_hash():
    return check_output(shlex.split("git rev-parse HEAD")).decode().strip()