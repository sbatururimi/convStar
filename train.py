import dataset
import utils
import os
import config
import lightning as L
from model.model import MultistagesConvStarNet


def datamodule_test():
    os.makedirs(config.LOGS_FOLDER, exist_ok=True)
    log_filename = os.path.join(config.LOGS_FOLDER, "datamodule_test.log")
    logger = utils.get_logger(__name__, log_filename=log_filename)

    # https://lightning.ai/docs/pytorch/stable/data/datamodule.html#using-a-datamodule
    cdm = dataset.CropsDataModule(
        data_dir=config.ZUERI_CROP_DATA_PATH,
        storage_dir=config.STORAGE_FOLDER,
        logger=logger,
    )
    cdm.setup(stage="fit")
    train_dl = cdm.train_dataloader()
    batches = list(train_dl)

def train():
    os.makedirs(config.LOGS_FOLDER, exist_ok=True)
    log_filename = os.path.join(config.LOGS_FOLDER, "train.log")
    logger = utils.get_logger(__name__, log_filename=log_filename)

    trainer = L.Trainer()
    model = MultistagesConvStarNet()
    crops_datamodule = dataset.CropsDataModule(
        data_dir=config.ZUERI_CROP_DATA_PATH,
        storage_dir=config.STORAGE_FOLDER,
        logger=logger,
    )
    trainer.fit(model=model, datamodule=crops_datamodule)
    # trainer.test(datamodule=crops_datamodule)
    # trainer.validate(datamodule=crops_datamodule)
    # trainer.predict(datamodule=crops_datamodule)

if __name__ == "__main__":
    utils.set_seed(42)
    # datamodule_test()
    train()