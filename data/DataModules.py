from IPython import embed
import torch
import random
import pickle as pkl
import numpy as np
from utils.VTKHelpers.CardiacMesh import CardiacMeshPopulation, Cardiac3DMesh
from torch.utils.data import TensorDataset, DataLoader, random_split
from typing import Any, List, Mapping, Optional, Sequence, Tuple, Union, Dict

import pytorch_lightning as pl
import logging
from argparse import Namespace

NUM_WORKERS = 1

def load_procrustes_transforms(filename):    
    return pkl.load(open(filename, "rb"))   

class CardiacMeshPopulationDataset(TensorDataset):

    def __init__(
       self, 
       meshes: Union[Mapping[str, np.array]],
       procrustes_transforms: Union[str, Mapping[str, Dict], None] = None,
       context=Namespace(logger=logging.getLogger())
    ):
        
       '''    
       procrustes_transforms: either a path to a pkl file or a a nested dictionary where each key is an subject and each inner dict contains "rotation" and "traslation" keys.
       '''
    
       if not isinstance(meshes, dict):                       
           raise(f"Argument should be a dictionary but is a {type(meshes)}")
           
       self.ids = list(meshes.keys())
       self.meshes = np.array(list(meshes.values()))            
        
       if isinstance(procrustes_transforms, str):
           procrustes_transforms = load_procrustes_transforms(procrustes_transforms)
       
       ids = list(meshes.keys())
       procrustes_transforms = { id: procrustes_transforms.get(id, {"discard": True}) for id in ids }
                        
       for id in self.ids:
           idx = self.ids.index(id)
           self.meshes[idx] = self.transform_mesh(self.meshes[idx], **procrustes_transforms[id])
       
       self.meshes = torch.Tensor(self.meshes)
        
       self._data_dict = { 
            self.ids[i]:self.meshes[i] for i, _ in enumerate(self.meshes) 
       }

    
    def __getitem__(self, id):
   
       if isinstance(id, int):
           return {
               "id": self.ids[id],
               "s": self._data_dict[self.ids[id]]
           }
       elif isinstance(id, str):
           try:
                return {
                    "id": id,
                    "s": self._data_dict[id]
                }
           except:
                return None
       
    def __len__(self):
       return len(self.ids)        
        
    
    def transform_mesh(self, mesh, rotation: Union[None, np.array] = None, traslation: Union[None, np.array] = None, discard=False):
        
        if discard:
            return None
        
        if traslation is not None:
            mesh = mesh - traslation
            
        if rotation is not None:
            centroid = mesh.mean(axis=0)
            mesh -= centroid
            mesh = mesh.dot(rotation)
            mesh += centroid
            
        return mesh 

class DataModule(pl.LightningDataModule):    
    
    '''
    PyTorch datamodule wrapping the CardiacMeshPopulation class
    '''
    
    def __init__(self, 
        # data_dir: Union[None, str] = None,
        torch_dataset_cls, 
        dataset_args: dict,
        #cardiac_population: Union[Mapping[str, np.array], CardiacMeshPopulation, None] = None, 
        #procrustes_transforms: Union[str, Mapping[str, Dict[str, np.array]], None] = None,
        batch_size: Union[int, list] = [32, 2, 1],
        split_lengths: Union[None, List[int]]=None
    ):

        '''
        params:
            torch_dataset_cls: the name of a class inheriting from torch.Dataset
            dataset_args: dictionary with the value of the arguments to build the dataset.            
            split_lengths: 
        '''
        
        super().__init__()
        
        self._torch_dataset_cls = torch_dataset_cls
        self.dataset_args = dataset_args   
        self.batch_size = batch_size if isinstance(batch_size, list) else [batch_size]*3
        self.split_lengths = self._get_lengths(split_lengths)


    def setup(self, stage: Optional[str] = None):

        self.dataset = self._torch_dataset_cls(**self.dataset_args)
        indices = list(range(sum(self.split_lengths)))
        random.shuffle(indices)
        
        train_indices = indices[:self.split_lengths[0]]
        val_indices = indices[self.split_lengths[0]: self.split_lengths[0]+self.split_lengths[1]]
        test_indices = indices[self.split_lengths[0]+self.split_lengths[1]:]
        predict_len = self.split_lengths[-1]       
 
        full_index_set = list(range(len(self.dataset)))
        random.shuffle(full_index_set)

        if predict_len > 0:
            predict_indices = full_index_set[:predict_len]
        else:
            predict_indices = full_index_set

        self.train_dataset = torch.utils.data.Subset(self.dataset, train_indices)
        self.val_dataset = torch.utils.data.Subset(self.dataset, val_indices)
        self.test_dataset = torch.utils.data.Subset(self.dataset, test_indices)
        self.predict_dataset = torch.utils.data.Subset(self.dataset, predict_indices)
        # self.train_dataset, self.val_dataset, self.test_dataset = random_split(self.dataset, self.split_lengths)        

        
    def _get_lengths(self, split_lengths):

        '''
        :param split_lengths: if len(split_lengths) == 2, the third one is taken as the complement to the total.
        if the contents are float numbers smaller than 1, are interpreted as fractions of the total
        :return:
        '''
                
        predict_len = split_lengths.pop(-1)

        if split_lengths is None:            
            train_len = int(0.6 * len(self.dataset))
            test_len = int(0.2 * len(self.dataset))
            val_len = len(self.dataset) - train_len - test_len            
            
        elif all([l > 1 for l in split_lengths]):            
            
            train_len = split_lengths[0]
            test_len = split_lengths[1]
            val_len = split_lengths[2]
               
            #raise ValueError("Bad values for split lengths. Expecting 2 or 3 fractions/integers.")
                
        elif all([l < 1 for l in split_lengths]):
            
            train_len = int(split_lengths[0] * len(self.dataset))
            test_len = int(split_lengths[1] * len(self.dataset))
            if len(split_lengths) == 2:
                val_len = len(self.dataset) - train_len - test_len
            elif len(split_lengths) == 3:            
                val_len = int(split_lengths[2] * len(self.dataset))
            else:
                raise ValueError("Bad values for split lengths. Expecting 2 or 3 fractions/integers.")
                
        return [train_len, val_len, test_len, predict_len]

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size[0], num_workers=NUM_WORKERS)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size[1], num_workers=NUM_WORKERS)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size[2], num_workers=NUM_WORKERS)

    def predict_dataloader(self):
        return DataLoader(self.predict_dataset, batch_size=self.batch_size[0], num_workers=NUM_WORKERS)
