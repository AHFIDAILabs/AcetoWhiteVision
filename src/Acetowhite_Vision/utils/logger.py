import os
import sys
import logging
from datetime import datetime

# Define the log file name with a timestamp
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# Create the full path for the log file within a dedicated 'logs' directory
logs_path = os.path.join(os.getcwd(), "logs")
os.makedirs(logs_path, exist_ok=True)

LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

# Configure the logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s: %(levelname)s: %(module)s: %(message)s]",
    handlers=[
        logging.FileHandler(LOG_FILE_PATH), # Log to a file
        logging.StreamHandler(sys.stdout)  # Log to the console
    ]
)

# Create and export a logger instance for the application
logger = logging.getLogger("AcetowhiteVisionLogger")