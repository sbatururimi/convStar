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

from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    DeviceStatsMonitor,
    EarlyStopping,
    LearningRateMonitor,
    # LearningRateFinder
)
from torch.optim import lr_scheduler
from lightning.pytorch.tuner import Tuner


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


def get_weights(nclasses, nclasses_local_1, nclasses_local_2):
    """Retuns the weights for each level

    Args:
        nclasses (_type_): level 3, i.e fine grained
        nclasses_local_1 (_type_): level 1, most coarsed
        nclasses_local_2 (_type_): level 2: middle level

    Returns:
        (weights level 3, weights level 1, weights level 2)): returns the weights for each level
    """
    # FIXME: class imbalance weights? why?
    LOSS_WEIGHT = torch.ones(nclasses)
    LOSS_WEIGHT[0] = (
        0  # looks like we assign a weight of 0 to `SummerBarley`. Check how `nclasses`` is obtained via `prepare_ground_truth_labels` in CropDataset
    )
    LOSS_WEIGHT_LOCAL_1 = torch.ones(nclasses_local_1)
    LOSS_WEIGHT_LOCAL_1[0] = 0
    LOSS_WEIGHT_LOCAL_2 = torch.ones(nclasses_local_2)
    LOSS_WEIGHT_LOCAL_2[0] = 0

    return LOSS_WEIGHT, LOSS_WEIGHT_LOCAL_1, LOSS_WEIGHT_LOCAL_2


def train_test_run():
    os.makedirs(config.LOGS_FOLDER, exist_ok=True)
    log_filename = os.path.join(config.LOGS_FOLDER, "train_test.log")
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

    loss_weights, loss_weights_level1, loss_weights_level2 = get_weights(
        nclasses, nclasses_local_1, nclasses_local_2
    )

    model = HierarchicalConvRNN(
        ms_convstar_net=ms_convstar_net,
        label_refinement_net=label_refinement_net,

        # uncomment if you want to give the same weights as in the repository (not in the paper)
        # loss_weights_level1=loss_weights_level1,
        # loss_weights_level2=loss_weights_level2,
        # loss_weights_level3=loss_weights,

        # uncomment if you want the same behaviour as in the repository. Default values follows the papers
        # lambda_1 = 0.1,
        # lambda_2 = 0.5,
        # lambda_3 = 1
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
    logger.debug(f"CUDA available: {torch.cuda.is_available()}")

    trainer = L.Trainer(fast_dev_run=True)
    trainer.fit(model=model, datamodule=crops_datamodule)
    # trainer.test(datamodule=crops_datamodule)
    # trainer.validate(datamodule=crops_datamodule)
    # trainer.predict(datamodule=crops_datamodule)

def train():
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

    loss_weights, loss_weights_level1, loss_weights_level2 = get_weights(
        nclasses, nclasses_local_1, nclasses_local_2
    )

    # now the decorator with Pytorch Lightning ------
    def schedular(optimizer):
        return lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=3)

    model = HierarchicalConvRNN(
        ms_convstar_net=ms_convstar_net,
        label_refinement_net=label_refinement_net,
        shedular=schedular

        # uncomment if you want to give the same weights as in the repository (not in the paper)
        # loss_weights_level1=loss_weights_level1,
        # loss_weights_level2=loss_weights_level2,
        # loss_weights_level3=loss_weights,

        # uncomment if you want the same behaviour as in the repository. Default values follows the papers
        # lambda_1 = 0.1,
        # lambda_2 = 0.5,
        # lambda_3 = 1
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
    logger.debug(f"CUDA available: {torch.cuda.is_available()}")

    # training settings
    lightning_logs_folder = os.path.join(config.LIGHTNING_LOGS_FOLDER, model.name)
    os.makedirs(lightning_logs_folder, exist_ok=True)

    model_checkpoints_dir = os.path.join(config.MODELS_CHECKPOINTS_FOLDER, model.name)
    os.makedirs(model_checkpoints_dir, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath=model_checkpoints_dir,
        filename="{epoch}-{val_loss:.2f}",
        monitor="val_loss",
        save_top_k=2,
    )

    device_stats_callback = DeviceStatsMonitor()
    early_stoping_callback = EarlyStopping(monitor="val_loss", patience=5)
    lr_monitor_callback = LearningRateMonitor(logging_interval=None) # https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.LearningRateMonitor.html#lightning.pytorch.callbacks.LearningRateMonitor

    trainer = L.Trainer(
        # deterministic=True, # FIXME: throwing error, see the init of the model for losses
        default_root_dir=lightning_logs_folder,
        callbacks=[checkpoint_callback, device_stats_callback, early_stoping_callback, lr_monitor_callback],
    )
    # Create a Tuner
    tuner = Tuner(trainer)

    # finds learning rate automatically
    # sets hparams.lr or hparams.learning_rate to that learning rate
    lr_finder = tuner.lr_find(model, datamodule=crops_datamodule)
    suggested_lr = lr_finder.suggestion()
    # logger.info(f"learning rate: {lr_finder.results}") # prints a list of tested learning rates
    logger.info(f"learning rate: {suggested_lr}")

    # Update model's learning rate
    model.hparams.lr = suggested_lr

    # start training
    best_model_ckpt_path = None  # TODO: add the path to best model checkpoint for resuming training from checkpoint
    trainer.fit(model=model, datamodule=crops_datamodule, ckpt_path=best_model_ckpt_path)

    best_k_models_dict_path = os.path.join(model_checkpoints_dir, "best_k_models.yml")
    checkpoint_callback.to_yaml(best_k_models_dict_path)
    logger.info(f"best model saved at: `{checkpoint_callback.best_model_path}`")
    logger.info(f"best k models info saved at: `{best_k_models_dict_path}`")

if __name__ == "__main__":
    utils.set_seed(42)

    # datamodule_test() #comment train and uncomment this line for datamodule test preparation
    # train_test_run()  # quick test/run to check correctness
    train()
