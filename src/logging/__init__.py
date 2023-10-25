import os
import sys
import logging

log_info = '[%(asctime)s: %(levelname)s: %(module)s: %(message)s]'
log_dir = 'results/logs'
log_path = os.path.join(log_dir, 'logging.log')
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format=log_info,
    handlers=[
        logging.FileHandler(log_path),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("Synthetic-Data-Generator-Logger")
