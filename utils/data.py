__author__ = "Dohyun Kim <donny8.kim@gmail.com>"
__all__ = ['make_dataset_item', 'load_dataloaders']

import glob
from .tools import *

from sklearn.model_selection import train_test_split
from monai.data import Dataset, DataLoader, decollate_batch
from monai.inferers import sliding_window_inference
from monai.transforms import (AsDiscrete, ToTensord, Activations, Compose, EnsureTyped, LoadImaged, ScaleIntensityd,
                              RandCropByPosNegLabeld, RandAffined, RandFlipd, RandShiftIntensityd)
from monai.networks.layers.factories import Act, Norm
from monai.networks.nets import SwinUNETR, UNet, BasicUNetPlusPlus

def make_dataset_item(dataset_dir, ids):
    train_items = list()
    val_items = list()
    test_items = list() 
    supp_items = list() 
    
    for file in glob.glob(dataset_dir + '/*'):
        case_id = file.split('_')[1].split('img/')[1]
        image_path = file
        aug_path = file.replace('img', 'img_max').replace('MEAN', 'MAX')
        label_path = dataset_dir.replace('img', 'label') + f'/{case_id}.png'
        file_checker(image_path)
        file_checker(aug_path)
        file_checker(label_path)

        item = {'case_id':case_id,
                'image':image_path,
                'imagemax': aug_path,
                'label':label_path
                }

        if(case_id in ids[0]):
            train_items.append(item)
        elif(case_id in ids[1]):
            val_items.append(item)
        elif(case_id in ids[2]):
            test_items.append(item)
        elif(case_id in ids[3]):
            supp_items.append(item)

    return train_items, val_items, test_items, supp_items


def load_dataloaders(case_ids:list, dataset_dir:str, batch_size:int, num_workers:int, support:str, useMAXAug=True):
    if not( os.path.exists('./data/train.txt') and 
            os.path.exists('./data/val.txt') and 
            os.path.exists('./data/test.txt')  ):     # make split 8:1:1
        train_set, val_test_set = train_test_split(case_ids, test_size=0.2, shuffle=False)
        val_set, test_set =  train_test_split(val_test_set, test_size=0.5, shuffle=False)
    else:

        with open('./data/train.txt') as f:
            train_set = list(map(strip ,f.readlines()))
        with open('./data/val.txt') as f:
            val_set = list(map(strip ,f.readlines()))
        with open('./data/test.txt') as f:
            test_set = list(map(strip ,f.readlines()))
        with open(f'./data/support_{support}.txt') as f:
            supp_set = list(map(strip ,f.readlines()))

    # make dataset item
    train_items, val_items, test_items, supp_items = make_dataset_item(dataset_dir, [train_set, val_set, test_set, supp_set])


    train_key_img = ['image']
    if(useMAXAug):
        train_key_img.append('imagemax')
    train_key_all = train_key_img + ['label']

    eval_key_img = ['image']
    eval_key_all = ['image', 'label']

        

    train_transform = Compose([
                                LoadImaged(keys=train_key_all, 
                                        ensure_channel_first=True),
                                ScaleIntensityd(keys=train_key_img,
                                            minv=0, maxv=1),
                                RandFlipd(keys=train_key_all,
                                        prob=0.5,
                                        spatial_axis=[1]),
                                RandAffined(keys=train_key_all,
                                            scale_range=[0.5, 1.5],
                                            rotate_range=[10,10],
                                            prob=0.5),
                                RandShiftIntensityd(keys=train_key_img,
                                                    offsets=0.1,
                                                    prob=0.5),
                                EnsureTyped(keys=train_key_all, track_meta=False),
                                ToTensord(keys=train_key_all),
                            ])

    val_transform = Compose([
                            LoadImaged(keys=eval_key_all, 
                                        ensure_channel_first=True),
                            ScaleIntensityd(keys=eval_key_img,
                                            minv=0, maxv=1),
                            EnsureTyped(keys=eval_key_all, track_meta=False),
                            ToTensord(keys=eval_key_all),
                            ])

    # make data loader
    train_dataset =  Dataset(train_items, transform=train_transform)
    val_dataset =  Dataset(val_items, transform=val_transform)
    supp_dataset =  Dataset(supp_items, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, num_workers=num_workers, pin_memory=True, shuffle=False)
    supp_loader = DataLoader(supp_dataset, batch_size=len(supp_items), num_workers=num_workers, pin_memory=True, shuffle=False) 

    loaders = [train_loader, val_loader, test_items, supp_loader]
    return loaders 