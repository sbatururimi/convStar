import os

import lightning as L
import numpy as np
import torch

import config
import dataset
import utils
from model.convstar.multistages_convstar_net import MultistagesConvStarNet
from model.label_refinement.label_refinement_net import LabelRefinementNet
from model.model import HierarchicalConvRNN
from torchinfo import summary


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


def train_test_run():
    os.makedirs(config.LOGS_FOLDER, exist_ok=True)
    log_filename = os.path.join(config.LOGS_FOLDER, "train.log")
    logger = utils.get_logger(__name__, log_filename=log_filename)

    # prepare the data module and get the number of classes per level from the train dataset
    crops_datamodule = dataset.CropsDataModule(
        data_dir=config.ZUERI_CROP_DATA_PATH,
        storage_dir=config.STORAGE_FOLDER,
        logger=logger,
    )
    crops_datamodule.setup(stage="fit")
    nclasses_local_1 = (
        crops_datamodule.traindataset.n_classes_local_1
    )  # highest, coarsed level
    nclasses_local_2 = crops_datamodule.traindataset.n_classes_local_2
    nclasses = (
        crops_datamodule.traindataset.n_classes
    )  # lowest, fine grained level we are really interested in (= level 3)

    # model networks creations
    ms_convstar_net = MultistagesConvStarNet(
        nclasses_level1=nclasses_local_1,
        nclasses_level2=nclasses_local_2,
        nclasses_level3=nclasses,
    )
    logger.info(summary(ms_convstar_net))

    label_refinement_net = LabelRefinementNet(
        num_classes_l1=nclasses_local_1,
        num_classes_l2=nclasses_local_2,
        num_classes_l3=nclasses,
    )
    logger.info(summary(label_refinement_net))

    model = HierarchicalConvRNN(
        ms_convstar_net=ms_convstar_net, label_refinement_net=label_refinement_net
    )

    # some model info
    model_parameters = filter(lambda p: p.requires_grad, ms_convstar_net.parameters())
    model_parameters2 = filter(
        lambda p: p.requires_grad, label_refinement_net.parameters()
    )

    params = sum([np.prod(p.size()) for p in model_parameters]) + sum(
        [np.prod(p.size()) for p in model_parameters2]
    )
    logger.info(
        f"Model trainable num params (MultiStages ConvStar netword & Label Refinement Network): {params}"
    )

    # setup for training and start training
    logger.debug("CUDA available: ", torch.cuda.is_available())


    trainer = L.Trainer(fast_dev_run=True)
    trainer.fit(model=model, datamodule=crops_datamodule)
    # trainer.test(datamodule=crops_datamodule)
    # trainer.validate(datamodule=crops_datamodule)
    # trainer.predict(datamodule=crops_datamodule)


if __name__ == "__main__":
    utils.set_seed(42)

    # datamodule_test() #comment train and uncomment this line for datamodule test preparation
    train_test_run() # quick test/run to check correctness
