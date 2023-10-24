import os
from traceback import print_tb
from typing import Callable, Optional, Tuple
import sys

import SimpleITK as sitk
from pathlib import Path
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

import nibabel as nib
import pathlib
from einops import rearrange

from joblib import Parallel, delayed

from sklearn.preprocessing import scale

from torchmtlr.utils import make_time_bins, encode_survival


def get_paths_to_patient_files(path_to_imgs, PatientID, append_mask=True):
    path_to_imgs = pathlib.Path(path_to_imgs)

    patients = [p for p in PatientID] # if os.path.isdir(path_to_imgs / p)
    paths = []
    for p in patients:
        print('p is ',p)
        print('path_to_imgs is ',path_to_imgs)
        path_to_ct = path_to_imgs / (p + '_image.nii.gz')

        if append_mask:
            path_to_mask = path_to_imgs/ (p + '_mask_GTV.nii.gz')
            paths.append((path_to_ct, path_to_mask))
        else:
            paths.append((path_to_ct))
    return paths

class HecktorDataset(Dataset):

    def __init__(self,
                 root_directory:str, 
                 clinical_data_path:str, 
                 time_bins:int = 14,
                 cache_dir:str = "data_cropped/data_cache/",
                 transform: Optional[Callable] = None,
                 num_workers: int = 1,
                 num_classes: int = 2,
                 patient_split:bool =True
    ):
        print(cache_dir)
        self.num_of_seqs = 1 #CT only
        # self.num_of_seqs = 2 #CT PT
        
        self.root_directory = root_directory

        self.transforms = transform
        self.num_workers = num_workers
        self.num_classes = num_classes

        self.clinical_data = self.make_data(clinical_data_path)
        
        # important, whether the time bins determine by number of event ################################

        if patient_split:
            self.time_bins1 = make_time_bins(times=self.clinical_data["time1"], num_bins=time_bins)
            self.time_bins2 = make_time_bins(times=self.clinical_data["time2"], num_bins=time_bins)
            self.time_bins3 = make_time_bins(times=self.clinical_data["time3"], num_bins=time_bins)
            self.time_bins4 = make_time_bins(times=self.clinical_data["time4"], num_bins=time_bins)
            self.time_bins5 = make_time_bins(times=self.clinical_data["time5"], num_bins=time_bins)
        else:
            self.time_bins1 = make_time_bins(times=self.clinical_data["time1"], num_bins=time_bins, event = self.clinical_data["event1"])
            self.time_bins2 = make_time_bins(times=self.clinical_data["time2"], num_bins=time_bins, event = self.clinical_data["event2"])
            self.time_bins3 = make_time_bins(times=self.clinical_data["time3"], num_bins=time_bins, event = self.clinical_data["event3"])
            self.time_bins4 = make_time_bins(times=self.clinical_data["time4"], num_bins=time_bins, event = self.clinical_data["event4"])
            self.time_bins5 = make_time_bins(times=self.clinical_data["time5"], num_bins=time_bins, event = self.clinical_data["event5"])

        self.y1 = encode_survival(self.clinical_data["time1"].values, self.clinical_data["event1"].values, self.time_bins1) # single event
        self.y2 = encode_survival(self.clinical_data["time2"].values, self.clinical_data["event2"].values, self.time_bins2) # single event
        self.y3 = encode_survival(self.clinical_data["time3"].values, self.clinical_data["event3"].values, self.time_bins3) # single event
        self.y4 = encode_survival(self.clinical_data["time4"].values, self.clinical_data["event4"].values, self.time_bins4) # single event
        self.y5 = encode_survival(self.clinical_data["time5"].values, self.clinical_data["event5"].values, self.time_bins5) # single event


        self.cache_path = get_paths_to_patient_files(cache_dir, self.clinical_data['ID'])


    def make_data(self, path):

        try:
            print(path)
            df = pd.read_csv(path + '/Clinical_List_5_Outcome.csv')
        except:
            df = path

        clinical_data = df
        clinical_data = clinical_data.rename(columns={"Death": "event1", "Dig2Follow": "time1",\
            "LF": "event2", "Dig2LF": "time2",\
            "RF": "event3", "Dig2RF": "time3",\
            "DF": "event4", "Dig2DF": "time4",\
            "SP": "event5", "Dig2SP": "time5"})

        clinical_data["Age"] = scale(clinical_data["Age"])
        clinical_data["SMOKE1"] = scale(clinical_data["SMOKE1"])
        clinical_data["Dig2RT"] = scale(clinical_data["Dig2RT"])
        clinical_data["Dose"] = scale(clinical_data["Dose"])
        clinical_data["Fraction"] = scale(clinical_data["Fraction"])
        clinical_data["ECOG"] = scale(clinical_data["ECOG"])
        clinical_data["T"] = scale(clinical_data["T"])
        clinical_data["N"] = scale(clinical_data["N"])
        clinical_data["AJCC"] = scale(clinical_data["AJCC"])
        
        cols_to_drop = []

        clinical_data = clinical_data.drop(cols_to_drop, axis=1)
        
        return clinical_data

    def __getitem__(self, idx: int):
        """Get an input-target pair from the dataset.

        The images are assumed to be preprocessed and cached.

        Parameters
        ----------
        idx
            The index to retrieve (note: this is not the subject ID).

        Returns
        -------
        tuple of torch.Tensor and int
            The input-target pair.
        """
        
        try:      # training data
            # clin_var_data = self.clinical_data.drop(["target_binary", 'time', 'event', 'Study ID'], axis=1) # single event
            clin_var_data = self.clinical_data.drop(['ID','time1', 'event1','time2', 'event2','time3', 'event3','time4', 'event4','time5', 'event5'], axis=1)
        except:   # test data
            clin_var_data = self.clinical_data.drop(['ID'], axis=1)

        clin_var = clin_var_data.iloc[idx].to_numpy(dtype='float32')
        
        target = (self.y1[idx], self.y2[idx], self.y3[idx], self.y4[idx], self.y5[idx])
        
        labels = self.clinical_data.iloc[idx].to_dict()
 
        # path = self.cache_path, f"{subject_id}_ct.nii.gz")
#         print('hi:', path)
        
        # image = sitk.ReadImage(path)
        # if self.transform is not None:
        #     image = self.transform(image)
        
        
        sample = dict()
        
        id_ = self.cache_path[idx][0].parent.stem

        sample['id'] = id_
        img = [self.read_data(self.cache_path[idx][i]) for i in range(self.num_of_seqs)]
        img = np.stack(img, axis=-1)
        sample['input'] = img 
        
        mask = self.read_data(self.cache_path[idx][-1])
        mask = mask/255
        mask = np.expand_dims(mask, axis=3)
        sample['target_mask'] = mask
        
        if self.transforms:
            sample = self.transforms(sample)
    
        return (sample, clin_var), target, labels
    
    

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.clinical_data)
    
    @staticmethod
    def read_data(path_to_nifti, return_numpy=True):
        if return_numpy:
            return sitk.GetArrayFromImage(sitk.ReadImage(str(path_to_nifti)))
        return sitk.ReadImage(str(path_to_nifti))
        # """Read a NIfTI image. Return a numpy array (default) or `nibabel.nifti1.Nifti1Image` object"""
        # if return_numpy:
        #     return nib.load(str(path_to_nifti)).get_fdata()
        # return nib.load(str(path_to_nifti))

    @staticmethod
    def to_categorical(y, num_classes):
        """ 1-hot encodes a tensor """
        y = np.eye(num_classes+1, dtype='uint8')[y]
        return y[:,:,:,1:num_classes+1]
