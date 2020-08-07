import numpy as np
import re
import pandas as pd
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
import os
from scipy.ndimage.interpolation import rotate
from skimage.transform import AffineTransform, warp
import shutil
import pickle
import torch
import datetime
from skimage import data, img_as_float
from skimage import io, exposure


class DataLoader(object):

    def __init__(self, data_path, dataloader_type='train', batchsize=1, device='cpu',image_resolution=[512,512]):
        
        np.random.seed(1)

        self.data_path = data_path
        if dataloader_type == 'test':
            self.lung_mask_path = self.data_path + '/prediction_dir'

        self.lung_path = data_path + '/CXR_png' 
        self.lung_mask_path_left = data_path  + '/ManualMask/leftMask' 
        self.lung_mask_path_right = data_path  + '/ManualMask/rightMask' 

        self.image_resolution = image_resolution
        self.device = device
        self.dataloader_type = dataloader_type
        self.batchsize = batchsize

        self.generate_data_list()

    def sorted_alphanumeric(self, data):
        convert = lambda text: int(text) if text.isdigit() else text.lower()
        alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
        return sorted(data, key=alphanum_key)

    def generate_data_list(self):

        if self.dataloader_type != 'test':

            if not os.path.isdir(self.data_path + '/pickle_dict'):

                self.lung_pickle_array = dict()
                self.lung_mask_pickle_array = dict()
                self.keys_array = []
                self.original_size_array = dict()
                lung_left_mask_pickle_array = dict()
                lung_right_mask_pickle_array = dict()
                        
                print('Lung')
                for i,files in enumerate(self.sorted_alphanumeric(os.listdir(self.lung_path))[:]):
                    print(i)
                    temp_image = plt.imread(self.lung_path+'/'+files)
                    if len(temp_image.shape)==3:
                        temp_image = temp_image[:,:,0]

                    self.original_size_array[files] = np.asarray(temp_image.shape)
                    temp_image = (temp_image - temp_image.min())/(temp_image.max()-temp_image.min())
                    temp_mask =  np.where(temp_image!=0,1,0)
                    temp_image = exposure.equalize_hist(temp_image, mask = np.where(temp_image!=0,1,0))
                    temp_image = temp_image*temp_mask
                    temp_image = ndimage.zoom(temp_image, np.asarray(self.image_resolution) / np.asarray(temp_image.shape), order=0)
                    temp_image = ndimage.median_filter(temp_image,5)
                    self.lung_pickle_array[files] = temp_image
                    self.keys_array.append(files)

                print('Lung Mask Left')
                for i,files in enumerate(self.sorted_alphanumeric(os.listdir(self.lung_mask_path_left))):
                    print(i)
                    temp_image = plt.imread(self.lung_mask_path_left+'/'+files)
                    if len(temp_image.shape)==3:
                        temp_image = temp_image[:,:,0]
                    
                    temp_image = ndimage.zoom(temp_image, np.asarray(self.image_resolution) / np.asarray(temp_image.shape), order=0)
                    lung_left_mask_pickle_array[files] = temp_image

                print('Lung Mask Right')
                for i,files in enumerate(self.sorted_alphanumeric(os.listdir(self.lung_mask_path_right))):
                    print(i)
                    temp_image = plt.imread(self.lung_mask_path_right+'/'+files)
                    if len(temp_image.shape)==3:
                        temp_image = temp_image[:,:,0]
                    
                    temp_image = ndimage.zoom(temp_image, np.asarray(self.image_resolution) / np.asarray(temp_image.shape), order=0)
                    lung_right_mask_pickle_array[files] = temp_image


                for file_name in self.keys_array:
                    temp_image = lung_left_mask_pickle_array[file_name] + lung_right_mask_pickle_array[file_name]
                    self.lung_mask_pickle_array[file_name] = np.where(temp_image>=1,1,0)

                self.keys_array = np.array(self.keys_array)

                os.mkdir(self.data_path + '/pickle_dict')

                with open(self.data_path + '/pickle_dict/lung_pickle_array.pickle', 'wb') as f:
                    pickle.dump(self.lung_pickle_array, f)
                f.close()   

                with open(self.data_path + '/pickle_dict/lung_mask_pickle_array.pickle', 'wb') as f:
                    pickle.dump(self.lung_mask_pickle_array, f)
                f.close()  

                with open(self.data_path + '/pickle_dict/original_size_array.pickle', 'wb') as f:
                    pickle.dump(self.original_size_array, f)
                f.close()     

                with open(self.data_path + '/pickle_dict/keys_array.pickle', 'wb') as f:
                    pickle.dump(self.keys_array, f)
                f.close() 

            else:
                with open(self.data_path + '/pickle_dict/lung_pickle_array.pickle', 'rb') as f:
                    self.lung_pickle_array = pickle.load(f)
                f.close()

                with open(self.data_path + '/pickle_dict/lung_mask_pickle_array.pickle', 'rb') as f:
                    self.lung_mask_pickle_array = pickle.load(f)
                f.close()

                with open(self.data_path + '/pickle_dict/original_size_array.pickle', 'rb') as f:
                    self.original_size_array = pickle.load(f)
                f.close()

                with open(self.data_path + '/pickle_dict/keys_array.pickle', 'rb') as f:
                    self.keys_array = pickle.load(f)
                f.close()

            np.random.seed(0)
            np.random.shuffle(self.keys_array)

            if self.dataloader_type == 'train':
                self.data_list = self.keys_array[:self.keys_array.shape[0]*70//100]

            elif self.dataloader_type == 'valid':
                self.data_list = self.keys_array[self.keys_array.shape[0]*70//100:]

            elif self.dataloader_type == 'complete':
                self.data_list = self.keys_array

            del self.keys_array

        elif self.dataloader_type == 'test':

            self.lung_pickle_array = dict()
            self.keys_array = []      
            self.lung_mask_pickle_array = dict()
            self.original_size_array = dict()

            for i,files in enumerate(self.sorted_alphanumeric(os.listdir(self.data_path))):
                print(i)

                temp_image = plt.imread(self.data_path+'/'+files)
                if len(temp_image.shape)==3:
                        temp_image = temp_image[:,:,0]

                self.original_size_array[files] = np.asarray(temp_image.shape)
                temp_image = (temp_image - temp_image.min())/(temp_image.max()-temp_image.min())
                temp_mask =  np.where(temp_image!=0,1,0)
                temp_image = exposure.equalize_hist(temp_image, mask = np.where(temp_image!=0,1,0))
                temp_image = temp_image*temp_mask
                temp_image = ndimage.zoom(temp_image, np.asarray(self.image_resolution) / np.asarray(temp_image.shape), order=0)
                temp_image = ndimage.median_filter(temp_image,5)
                self.lung_pickle_array[files] = temp_image
                self.keys_array.append(files)

            self.data_list = np.array(self.keys_array)

            del self.keys_array

            print(len(self.data_list))

    def load_image_and_label(self, image_list):

        images = np.zeros((image_list.shape[0], 1, self.image_resolution[0], self.image_resolution[1]))
        labels = np.zeros((image_list.shape[0], self.image_resolution[0], self.image_resolution[1]))

        if self.dataloader_type == 'train':
            for i,image_id in enumerate(image_list):
                images[i][0], labels[i] = self.augment(self.lung_pickle_array[image_id], self.lung_mask_pickle_array[image_id])
        else:
            for i,image_id in enumerate(image_list):
                images[i][0] = self.lung_pickle_array[image_id]
                if self.dataloader_type != 'test':
                    labels[i] = self.lung_mask_pickle_array[image_id]
        
        return images, labels

    def augment(self, image, label, hflip_prob=0.75, vflip_prob=0.75, rotation_angle_list=np.arange(-90,90)):

        if np.random.rand(1)[0]>1-hflip_prob:
            image = np.flip(image, axis=1)
            label = np.flip(label, axis=1)
         
        if np.random.rand(1)[0]>1-vflip_prob:
            image = np.flip(image, axis=0)
            label = np.flip(label, axis=0)

        rotation_angle = np.random.choice(rotation_angle_list)
        image = rotate(image, angle=rotation_angle, order=1, mode='constant', reshape=False)
        label = rotate(label, angle=rotation_angle, order=1, mode='constant', reshape=False)

        return image, label

    def __getitem__(self, idx):

        if self.dataloader_type=='train':
            if self.batchsize >= len(self.data_list):
                temp_data_list = self.data_list
            else:  
                if (idx+1)*self.batchsize >= len(self.data_list):
                    temp_data_list = self.data_list[idx*self.batchsize : ]
                    np.random.shuffle(self.data_list)
                else:
                    temp_data_list = self.data_list[idx*self.batchsize : (idx+1)*self.batchsize]
                    if len(self.data_list)%self.batchsize == 0:
                        if idx == len(self.data_list)//self.batchsize:
                            np.random.shuffle(self.data_list)
        else:
            if self.batchsize >= len(self.data_list):
                temp_data_list = self.data_list
            else: 
                if (idx+1)*self.batchsize >= len(self.data_list):
                    temp_data_list = self.data_list[idx*self.batchsize : ]
                else:
                    temp_data_list = self.data_list[idx*self.batchsize : (idx+1)*self.batchsize]

        batch_images, batch_label = self.load_image_and_label(np.array(temp_data_list))
        batch_images_tensor = torch.from_numpy(batch_images).double().to(self.device)

        if self.dataloader_type != 'test':
            batch_label_tensor = torch.tensor(batch_label, dtype=torch.long, device=self.device)
            return batch_images_tensor, batch_label_tensor  
        else: 
            return batch_images_tensor
