import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

import os, random
from torch.utils.data import Dataset, DataLoader
from typing import Union

#---------------------------------------------
# custom dataset
#https://discuss.pytorch.org/t/custom-data-loader-for-big-data/129361
class CustomDataset(Dataset):
    def __init__(self, path_dir_X:str, path_dir_Y:str, n_test:Union[int,float], n_val:Union[int,float], batch_size:int): # n_test -> float:ratio of test, int:number of test
        #-----------------
        # batch_size
        self.batch_size = batch_size
        # path_dir_X, path_dir_Y
        self.path_dir_X = path_dir_X
        self.path_dir_Y = path_dir_Y
        # list_file_name_all
        self.list_file_name_all = os.listdir(path_dir_X)
        # n_data_all
        self.n_data_all = len(self.list_file_name_all)
        #check
        if len(os.listdir(path_dir_X)) != len(os.listdir(path_dir_Y)):
            raise ValueError("error!!!")
        if len(set(os.listdir(path_dir_X)) - set(os.listdir(path_dir_Y))) != 0:
            raise ValueError("error!!!")
        #-----------------
        # suffle
        random.shuffle(self.list_file_name_all)
        #-----------------
        # n_test
        if type(n_test)==int:
            self.n_test = n_test
        elif type(n_test)==float:
            self.n_test = int(len(self.list_file_name_all)*n_test)
        else:
            raise ValueError("error!!!")
        # n_val
        if type(n_val)==int:
            self.n_val = n_val
        elif type(n_val)==float:
            self.n_val = int(len(self.list_file_name_all)*n_val)
        else:
            raise ValueError("error!!!")
        #check
        if self.n_data_all <= self.n_test+self.n_val:
            raise ValueError("error!!!")
        #-----------------
        # list_file_name_test / _val / _train
        self.list_file_name_test = self.list_file_name_all[:self.n_test]
        self.list_file_name_val = self.list_file_name_all[self.n_test:self.n_test+self.n_val]
        self.list_file_name_train = self.list_file_name_all[self.n_test+self.n_val:]
        
    def __len__(self):
        return len(self.list_file_name_train)
    
    def __getitem__(self, x):
        #return
        return self.getdata(list_file_name=self.list_file_name_train, index=x)
    
    def getdata(self, list_file_name, index):
        #file_name
        file_name = list_file_name[index]
        #data_X
        path_file_X = "{0}/{1}".format(self.path_dir_X, file_name)
        data_X = np.load(path_file_X, allow_pickle=True)
        #data_X = torch.from_numpy(data_X).to(torch.float32).requires_grad_(True)
        #data_Y
        path_file_Y = "{0}/{1}".format(self.path_dir_Y, file_name)
        data_Y = np.load(path_file_Y, allow_pickle=True)
        #data_Y = torch.from_numpy(data_Y).to(torch.float32).requires_grad_(True)
        #return
        return data_X, data_Y
    
    def return_n_data_all(self):
        return self.n_data_all
    
    def return_n_test(self):
        return self.n_test
    
    def return_n_val(self):
        return self.n_val
    
    def return_n_train(self):
        return self.n_data_all - self.n_val - self.n_test
    
    def return_batch_size(self):
        return self.batch_size
    
    def return_shape_X(self):
        data_sample = self.getdata(self.list_file_name_all, 0)[0]
        return data_sample.shape
    
    def return_shape_Y(self):
        data_sample = self.getdata(self.list_file_name_all, 0)[1]
        return data_sample.shape
    
    def return_test_data(self):
        #https://www.tutorialspoint.com/how-to-join-tensors-in-pytorch
        data_X_test = torch.stack([torch.from_numpy(self.getdata(self.list_file_name_test, i)[0]) for i in range(self.n_test)])
        data_Y_test = torch.stack([torch.from_numpy(self.getdata(self.list_file_name_test, i)[1]) for i in range(self.n_test)])
        return data_X_test, data_Y_test
    
    def return_val_data(self):
        #https://www.tutorialspoint.com/how-to-join-tensors-in-pytorch
        data_X_val = torch.stack([torch.from_numpy(self.getdata(self.list_file_name_val, i)[0]) for i in range(self.n_val)])
        data_Y_val = torch.stack([torch.from_numpy(self.getdata(self.list_file_name_val, i)[1]) for i in range(self.n_val)])
        return data_X_val, data_Y_val
    
    def change_data_setting_to_train(self, data):
        return data.to(torch.float32).requires_grad_(True)

    


#---------------------------------------------
#init weight
#https://stackoverflow.com/questions/49433936/how-do-i-initialize-weights-in-pytorch
#https://www.geeksforgeeks.org/initialize-weights-in-pytorch/
def init_normal_dist(model):
    if type(model) == nn.Linear:
        y = model.in_features
        # model.weight.data shoud be taken from a normal distribution
        model.weight.data.normal_(0.0,1/np.sqrt(y))
        # model.bias.data should be 0
        model.bias.data.fill_(0)
    #if type(model) == nn.Conv1d: #Conv1d = keernel weight
    #    nn.init.zeros_(model.weight)
    #    print(len(model.weight))