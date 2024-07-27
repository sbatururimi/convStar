import dataset
import utils
import os
import config


def test():
    os.makedirs(config.LOGS_FOLDER, exist_ok=True)
    log_filename = os.path.join(config.LOGS_FOLDER, "train.log")
    logger = utils.get_logger(__name__, log_filename=log_filename)
    cdm = dataset.CropsDataModule(
        data_dir="/mnt/disk1/projects/multi-stage-convSTAR-network/data.h5",
        storage_dir=config.STORAGE_FOLDER,
        logger=logger,
    )
    cdm.setup(stage="fit")


if __name__ == "__main__":
    test()
