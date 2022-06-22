import os
import logging
import shlex
from subprocess import check_output
from datetime import datetime

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

repo_root = check_output(shlex.split("git rev-parse --show-toplevel")).strip().decode('ascii')

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger()