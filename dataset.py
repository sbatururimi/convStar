import lightning as L
from torch.utils.data import DataLoader, Dataset
import h5py
from enum import Enum
import numpy as np
import os
import csv

STORAGE_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__)), "storage")

class TrainingMode(Enum):
    Train = "train"
    Val = "val"
    Test = "test"

class CropDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        time_downsample_factor: float = 2,
        mode: TrainingMode = TrainingMode.Train,
        threshold: float = 0.9,
        gt_filename: str = "labels.csv"):

        self.data = h5py.File(data_path, "r", libver='latest', swmr=True)
        self.samples = self.data["data"].shape[0]
        self.max_obs = self.data["data"].shape[1] # here we have the number of satellite data without distinction between sentinel-1 or 2
        self.time_downsample_factor = time_downsample_factor
        self.max_obs = int(142/self.time_downsample_factor) # apparently it's 142  images divided by 2 satellites sentinel-2, so to have images from 1 satellite
        self.mode = mode
        self.threshold = threshold

        # simple split in 75% for train, 15% val and 10% test
        self.valid_list = self.split(mode)

        # prepare ground truth labels
        self.prepare_ground_truth_labels(gt_filename)

    def prepare_ground_truth_labels(self, gt_filename: str):
        with open(gt_filename, mode='r') as file:
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

        tier_2[0] = '0_unknown' # first, broader level (coarse)
        tier_3[0] = '0_unknown'
        tier_4[0] = '0_unknown' # granular level

        # tier 1 helps to differentiate empty soil/infrastructure/vegetation/undefined
        # => we keep only the vegetation by creating an index of relevant labels
        self.label_list = []
        for i in range(len(tier_2)):
            if tier_1[i] == 'Vegetation' and tier_4[i] != '':
                self.label_list.append(i)

            if tier_2[i] == '':
                tier_2[i] = '0_unknown'
            if tier_3[i] == '':
                tier_3[i] = '0_unknown'
            if tier_4[i] == '':
                tier_4[i] = '0_unknown'

        # unique labels & sort
        tier_2_elements = list(set(tier_2))
        tier_3_elements = list(set(tier_3))
        tier_4_elements = list(set(tier_4))
        tier_2_elements.sort()
        tier_3_elements.sort()
        tier_4_elements.sort()

        # tier_2_ = []
        # tier_3_ = []
        # tier_4_ = []
        # for i in range(len(tier_2)):
        #     tier_2_.append(tier_2_elements.index(tier_2[i]))
        #     tier_3_.append(tier_3_elements.index(tier_3[i]))
        #     tier_4_.append(tier_4_elements.index(tier_4[i]))



    def split(self, mode: TrainingMode, train_perc:float = 0.75, val_perc: float = 0.15):
        valid = np.zeros(self.samples)
        train_split = int(self.samples * train_perc)
        valid_split = int(self.samples * (train_perc + val_perc))

        if mode == TrainingMode.Train:
            valid[:train_split] = 1.
        if mode == TrainingMode.Val:
            valid[train_split:valid_split] = 1.
        if mode == TrainingMode.Test:
            valid[valid_split:] = 1.

        w, h = self.data["gt"][0,...,0].shape # 24x24 as we have (num_samples, features, w_patch, h_pathc, channles) =>  (27977, 142, 24, 24, 9), type "<i2">
        for i in range(self.samples):
            # retrieves the i-th sample from the gt dataset, specifically the first channel and all values in the 24x24 grid
            # then by dividing by (wxh) we check the proportion of non-zero elements in the grid.
            # Шf we have less than 90% (default еркуырщдв), the we say that the sample is not a valid sample
            if np.sum( self.data["gt"][i,...,0] != 0 )/(w*h) < self.threshold:
                valid[i] = 0
        
        # return the indices of non-zero elements from the valid array.
        return np.nonzero(valid)[0]
            
        
    

class CropsDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int = 4, workers: int = 8):
        self.batch_size = batch_size
        self.workers = workers
        self.data_dir = data_dir
        
    def setup(self, stage: str):
         # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            self.traindataset = 
        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            pass
        if stage == "predict":
            pass
    
    
    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size, num_workers=self.workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size, num_workers=self.workers)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size, num_workers=self.workers)