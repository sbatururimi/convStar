import lightning as L
from torch.utils.data import DataLoader, Dataset
import h5py
from enum import Enum
import numpy as np
import os
import csv
import logging
import torch


class TrainingMode(Enum):
    Train = "train"
    Val = "val"
    Test = "test"

    def __str__(self):
        return self.value


class CropDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        time_downsample_factor: float = 2,
        mode: TrainingMode = TrainingMode.Train,
        eval_mode: bool = False,
        threshold: float = 0.9,
        num_channel: int =4,
        gt_filepath: str = "labels.csv",
        apply_cloud_masking: bool = False,
        return_cloud_cover: bool = False,
        augment_rate: float = 0.66,
        logger: logging.Logger = None,
    ):
        self.data = h5py.File(data_path, "r", libver="latest", swmr=True)
        self.samples = self.data["data"].shape[0]
        self.spatial = self.data["data"].shape[2:-1]
        self.max_obs = self.data[
            "data"
        ].shape[
            1
        ]  # here we have the number of satellite data without distinction between sentinel-1 or 2

        self.time_downsample_factor = time_downsample_factor
        self.max_obs = int(
            142 / self.time_downsample_factor
        )  # apparently it's 142  images divided by 2 satellites sentinel-2, so to have images from 1 satellite

        self.mode = mode
        self.threshold = threshold
        self.logger = logger
        self.eval_mode = eval_mode
        self.num_channel = num_channel # channels to be used
        self.augment_rate = augment_rate

        # clouds related
        self.apply_cloud_masking = apply_cloud_masking
        # self.cloud_threshold = cloud_threshold
        self.return_cloud_cover = return_cloud_cover

        # simple split in 75% for train, 15% val and 10% test
        self.valid_list = self.split(mode)
        self.valid_samples = self.valid_list.shape[0] # number of valid samples to be used

        # prepare ground truth labels
        self.prepare_ground_truth_labels(gt_filepath)

        self.logger.info("-"*100)
        self.logger.info(f"Dataset size: {self.samples}")
        self.logger.info(f"Valid dataset size: {self.valid_samples}")
        self.logger.info(f"Sequence length: {self.max_obs}")
        self.logger.info(f"Spatial size: {self.spatial}")
        self.logger.info(f"Number of classes (most detailed, finest): {self.n_classes}")
        self.logger.info(f"Number of classes (most coarse) - local-1: {self.n_classes_local_1}")
        self.logger.info(f"Number of classes (middle level)- local-2: {self.n_classes_local_2}")
        self.logger.info("-"*100)

        # TODO: consistency loss part not understandable for now

    def __len__(self):
        return self.valid_samples

    def __getitem__(self, idx):
        idx = self.valid_list[idx]
        X = self.data["data"][idx]

        if self.apply_cloud_masking or self.return_cloud_cover:
            CC = self.data["cloud_cover"][idx]

        target_ = self.data["gt"][idx,...,0] # gt -> ground truth, we got the label index here
        if self.eval_mode: # why this is different?
            gt_instance = self.data["gt_instance"][idx,...,0]

        X = np.transpose(X, (0, 3, 1, 2)) # we got (142, 9, 24, 24): (samples, height, width, channels) -> (samples, channels, height, width)

        # temporal downsampling and channel selection.
        # 1) `0::self.time_downsample_factor`: taking every 2nd sample starting from index 0
        # = > 142 // 2 = 71 samples
        # 2) ``:self.num_channel`: keeps only the first 4 channels
        #
        # => after downsampling and channel selection, the shape of X is (71, 4, 12, 12).
        X = X[0::self.time_downsample_factor,:self.num_channel,...]

        if self.apply_cloud_masking or self.return_cloud_cover:
            CC = CC[0::self.time_downsample_factor,...]

        # Change labels to our previously built indexes of labels
        target = np.zeros_like(target_)
        target_local_1 = np.zeros_like(target_)
        target_local_2 = np.zeros_like(target_)
        for i in range(len(self.label_list)):
            target[target_ == self.label_list[i]] = self.label_list_glob[i] # finest level (level 3)
            target_local_1[target_ == self.label_list[i]] = self.label_list_local_1[i] # level 1
            target_local_2[target_ == self.label_list[i]] = self.label_list_local_2[i] # level 2

        # transform to tensors
        X = torch.from_numpy(X)
        target = torch.from_numpy(target).float()
        target_local_1 = torch.from_numpy(target_local_1).float()
        target_local_2 = torch.from_numpy(target_local_2).float()

        if self.apply_cloud_masking or self.return_cloud_cover:
            CC = torch.from_numpy(CC).float()

        if self.eval_mode:
            gt_instance = torch.from_numpy(gt_instance).float()

        # keep values between 0-1
        X = X * 1e-4
        # FIXME: Previous line should be modified as X = X / 4095 but not tested yet!
        # FIXME: (Stas): why 4095?

        # Cloud masking
        if self.apply_cloud_masking:
            CC_mask = CC < self.cloud_threshold
            CC_mask = CC_mask.view(CC_mask.shape[0],1,CC_mask.shape[1],CC_mask.shape[2])
            X = X * CC_mask.float()

        #augmentation
        if self.eval_mode == False and np.random.rand() < self.augment_rate:
            flip_dir  = np.random.randint(3)
            if flip_dir == 0:
                X = X.flip(2)
                target = target.flip(0)
                target_local_1 = target_local_1.flip(0)
                target_local_2 = target_local_2.flip(0)
            elif flip_dir == 1:
                X = X.flip(3)
                target = target.flip(1)
                target_local_1 = target_local_1.flip(1)
                target_local_2 = target_local_2.flip(1)
            elif flip_dir == 2:
                X = X.flip(2,3)
                target = target.flip(0,1)
                target_local_1 = target_local_1.flip(0,1)
                target_local_2 = target_local_2.flip(0,1)

        if self.return_cloud_cover:
            if self.eval_mode:
                return X.float(), target.long(), target_local_1.long(), target_local_2.long(), gt_instance.long(), CC.float()
            else:
                return X.float(), target.long(), target_local_1.long(), target_local_2.long(), CC.float()
        else:
            if self.eval_mode:
                return X.float(), target.long(), target_local_1.long(), target_local_2.long(), gt_instance.long()
            else:
                return X.float(), target.long(), target_local_1.long(), target_local_2.long()

    def prepare_ground_truth_labels(self, gt_filepath: str):
        self.logger.debug(f"Preparing ground truth labels [mode={self.mode}]...")
        with open(gt_filepath, mode="r") as file:
            reader = csv.reader(file)
            tier_1 = []
            tier_2 = []
            tier_3 = []
            tier_4 = []
            for line in reader:
                tier_1.append(line[-5])
                tier_2.append(line[-4])
                tier_3.append(line[-3])
                tier_4.append(line[-2])

        tier_2[0] = "0_unknown"  # first, broader level (coarse)
        tier_3[0] = "0_unknown"
        tier_4[0] = "0_unknown"  # granular level

        # tier 1 helps to differentiate empty soil/infrastructure/vegetation/undefined
        # => we keep only the vegetation by creating an index of relevant labels
        self.label_list = []
        for i in range(len(tier_2)):
            if tier_1[i] == "Vegetation" and tier_4[i] != "":
                self.label_list.append(i)

            if tier_2[i] == "":
                tier_2[i] = "0_unknown"
            if tier_3[i] == "":
                tier_3[i] = "0_unknown"
            if tier_4[i] == "":
                tier_4[i] = "0_unknown"

        # unique labels & sort
        tier_2_elements = list(set(tier_2))
        tier_3_elements = list(set(tier_3))
        tier_4_elements = list(set(tier_4))
        tier_2_elements.sort()
        tier_3_elements.sort()
        tier_4_elements.sort()

        # now we use a mapping of unique labels per hierarchy
        # and the occurence in the initial level based label to create indexes.
        # example: ['0_unknown', 'Field crops', 'Field crops', 'Special crops'] -> [0, 1, 1, 5]
        tier_2_ = []
        tier_3_ = []
        tier_4_ = []
        for i in range(len(tier_2)):
            tier_2_.append(tier_2_elements.index(tier_2[i]))
            tier_3_.append(tier_3_elements.index(tier_3[i]))
            tier_4_.append(tier_4_elements.index(tier_4[i]))

        # in each line of the file we have a hierarchy of labels,
        # let's use our previously built map of label indexes for each line
        self.label_list_local_1 = []
        self.label_list_local_2 = []
        self.label_list_glob = []
        self.label_list_local_1_name = []
        self.label_list_local_2_name = []
        self.label_list_glob_name = []
        for gt in self.label_list:
            # built indexes of relted label for each line of the ground truth file
            self.label_list_local_1.append(tier_2_[int(gt)]) # level 1 for each line
            self.label_list_local_2.append(tier_3_[int(gt)]) # level 2 for each line
            self.label_list_glob.append(tier_4_[int(gt)]) # level 3 (more granular) for each line
            # names of label per hierarchy level
            self.label_list_local_1_name.append(tier_2[int(gt)])
            self.label_list_local_2_name.append(tier_3[int(gt)])
            self.label_list_glob_name.append(tier_4[int(gt)])

        self.n_classes = max(self.label_list_glob) + 1 # number of distinct most granular labels (lowest levels)
        self.n_classes_local_1 = max(self.label_list_local_1) + 1 # most coarse level, number of labels
        self.n_classes_local_2 = max(self.label_list_local_2) + 1 # middle level, number of distinct labels

    def split(
        self, mode: TrainingMode, train_perc: float = 0.75, val_perc: float = 0.15
    ):
        self.logger.debug(f"Splitting data [mode={mode}]...")
        valid = np.zeros(self.samples)
        train_split = int(self.samples * train_perc)
        # this part is different than the intial repository, there are no valid split per-ser, just train/test
        valid_split = int(self.samples * (train_perc + val_perc))

        if mode == TrainingMode.Train:
            valid[:train_split] = 1.0
        if mode == TrainingMode.Val:
            valid[train_split:valid_split] = 1.0
        if mode == TrainingMode.Test:
            valid[valid_split:] = 1.0

        w, h = self.data["gt"][
            0, ..., 0
        ].shape  # 24x24 as we have (num_samples, features, w_patch, h_pathc, channles) =>  (27977, 142, 24, 24, 9), type "<i2">
        for i in range(self.samples):
            # retrieves the i-th sample from the gt dataset, specifically the first channel and all values in the 24x24 grid
            # then by dividing by (wxh) we check the proportion of non-zero elements in the grid.
            # Шf we have less than 90% (default еркуырщдв), the we say that the sample is not a valid sample
            if np.sum(self.data["gt"][i, ..., 0] != 0) / (w * h) < self.threshold:
                valid[i] = 0

        # return the indices of non-zero elements from the valid array.
        self.logger.debug("split done.")
        return np.nonzero(valid)[0]


class CropsDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        storage_dir: str,
        batch_size: int = 4,
        workers: int = 8,
        logger: logging.Logger = None,
    ):
        self.batch_size = batch_size
        self.workers = workers
        self.data_dir = data_dir
        self.storage_dir = storage_dir
        self.logger = logger

    def setup(self, stage: str):
        self.logger.debug(f"Setting datasets for stage `{stage}`...")
        gt_filepath = os.path.join(self.storage_dir, "labels.csv")
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            self.traindataset = CropDataset(
                self.data_dir,
                mode=TrainingMode.Train,
                eval_mode=False,
                threshold=0,
                gt_filepath=gt_filepath,
                logger=self.logger,
            )
        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            pass
        if stage == "predict":
            pass

        self.logger.debug("setting done.")

    def train_dataloader(self):
        return DataLoader(
            self.traindataset,
            batch_size=self.batch_size,
            num_workers=self.workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.mnist_val, batch_size=self.batch_size, num_workers=self.workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.mnist_test, batch_size=self.batch_size, num_workers=self.workers
        )
