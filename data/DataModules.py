import torch
from utils.VTKHelpers.CardiacMesh import CardiacMeshPopulation, Cardiac3DMesh
from torch.utils.data import TensorDataset, DataLoader, random_split
from typing import Any, List, Mapping, Optional, Sequence, Tuple, Union

import pytorch_lightning as pl
import logging
from argparse import Namespace

class CardiacMeshPopulationDataset(TensorDataset):
    
    '''
    PyTorch dataset wrapping the CardiacMeshPopulation class
    
    input:
    - cardiac_population: if root_dir is set, this argument should not be provided.
    - root_dir: if cardiac_population is set, this argument should not be provided.
    - context: namespace
    '''
    
    def __init__(
        self, 
        cardiac_population: Union[Mapping,CardiacMeshPopulation, None]=None, 
        root_dir: Union[str, None]=None, 
        context=Namespace(logger=logging.getLogger())
    ):
        
        if cardiac_population is None and root_dir is None:            
            raise ValueError("Provide either cardiac_population or root_dir as argument")
        elif cardiac_population is not None and root_dir is not None:
            raise ValueError("Provide only one of cardiac_population or root_dir as argument")
        
        if root_dir is not None:
            cardiac_population = CardiacMeshPopulation(root_dir)
        
        # For retrieving cached data in the form of a dictionary
        if isinstance(cardiac_population, dict):
            self.ids = cardiac_population["ids"]
            self.meshes = torch.Tensor(cardiac_population["meshes"])
        else:
            self.ids = cardiac_population.ids
            self.meshes = torch.Tensor(cardiac_population.as_numpy_array())
        
        #TODO: check that this does not produce a copy of self.data (I think it does not)
        self._data_dict = { 
            self.ids[i]:self.meshes[i] for i, _ in enumerate(self.meshes) 
        }
        
    def __getitem__(self, id):        
        return self._data_dict[self.ids[id]]
        
    def __len__(self):
        return len(self.ids)


class CardiacMeshPopulationDM(pl.LightningDataModule):    
    
    '''
    PyTorch datamodule wrapping the CardiacMeshPopulation class
    '''
    
    def __init__(self, 
        data_dir: Union[None, str] = None, 
        cardiac_population: Union[Mapping, CardiacMeshPopulation, None] = None, 
        batch_size: int = 32,
        split_lengths: Union[None, List[int]]=None
    ):

        '''
        params:
            data_dir:
            batch_size:
            split_lengths:
        '''
        
        super().__init__()
        self.data_dir = data_dir
        self.cardiac_population = cardiac_population

        self.batch_size = batch_size
        self.split_lengths = None


    def setup(self, stage: Optional[str] = None):

        popu = CardiacMeshPopulationDataset(root_dir=self.data_dir, cardiac_population=self.cardiac_population)

        if self.split_lengths is None:
            train_len = int(0.6 * len(popu))
            test_len = int(0.2 * len(popu))
            val_len = len(popu) - train_len - test_len
            self.split_lengths = [train_len, val_len, test_len]

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(popu, self.split_lengths)        


    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=2, num_workers=8)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, num_workers=8)
