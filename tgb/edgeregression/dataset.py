from typing import Optional, cast, Union, List, overload, Literal
import os.path as osp
import numpy as np
from tgb.info import PROJ_DIR

class EdgeRegressionDataset(object):
    def __init__(
        self, 
        name: str, 
        root: Optional[str] = 'datasets', 
        meta_dict: Optional[dict] = None,
        preprocess: Optional[bool] = True,
        ):
        r"""Dataset class for edge regression tasks. Stores meta information about each dataset such as evaluation metrics etc.
        also automatically pre-processes the dataset.
        Args:
            name: name of the dataset
            root: root directory to store the dataset folder
            meta_dict: dictionary containing meta information about the dataset, should contain key 'dir_name' which is the name of the dataset folder
            preprocess: whether to pre-process the dataset
        """
        self.name = name ## original name
        root = PROJ_DIR + root

        if meta_dict is None:
            self.dir_name = '_'.join(name.split('-')) ## replace hyphen with underline
            meta_dict = {'dir_name': self.dir_name}
        else:
            self.dir_name = meta_dict['dir_name']
        self.root = osp.join(root, self.dir_name)
        self.meta_dict = meta_dict
        if ("fname" not in self.meta_dict):
            self.meta_dict["fname"] = self.root + "/" + self.name + ".csv"

        #check if the root directory exists, if not create it
        if osp.isdir(self.root):
            print("Dataset directory is ", self.root)
        else:
            raise FileNotFoundError(f"Directory not found at {self.root}")


        #TODO Andy: add url logic here from info.py to manage the urls in a centralized file

        if preprocess:
            self.pre_process()
        


    def pre_process(self):
        r"""Turns raw data .csv file into TG learning ready format such as for TGN, stores the processed file locally for faster access later
        Returns:
            'ml_<network>.csv': source, destination, timestamp, state_label, index 	# 'index' is the index of the line in the edgelist
            'ml_<network>.npy': contains the edge features; this is a numpy array where each element corresponds to the features of the corresponding line specifying one edge. If there are no features, should be initialized by zeros
            'ml_<network>_node.npy': contains the node features; this is a numpy array that each element specify the features of one node where the node-id is equal to the element index.
        """

        #check if path to file is valid 
        if not osp.exists(self.meta_dict['fname']):
            raise FileNotFoundError(f"File not found at {self.meta_dict['fname']}")
        
    



def main():
    dataset = EdgeRegressionDataset(name="un_trade", root="datasets")

if __name__ == "__main__":
    main()