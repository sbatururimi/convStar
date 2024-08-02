import os

ZUERI_CROP_DATA_PATH = "/mnt/disk1/projects/multi-stage-convSTAR-network/data.h5"
STORAGE_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), "storage"))
LOGS_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), "logs"))
LIGHTNING_LOGS_FOLDER = os.path.join(LOGS_FOLDER, "lightning")
MODELS_CHECKPOINTS_FOLDER = os.path.join(STORAGE_FOLDER, "checkpoints")