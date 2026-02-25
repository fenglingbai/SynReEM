#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
import os
from os.path import join
import sys
# 将当前路径添加到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
import json
import pickle
from collections import OrderedDict
import cv2
import SimpleITK as sitk
import numpy as np
from tqdm import tqdm

from nnunet.paths import nnUNet_raw_data, preprocessing_output_dir
from skimage import io

def save_json(obj, file: str, indent: int = 4, sort_keys: bool = True) -> None:
    with open(file, 'w') as f:
        json.dump(obj, f, sort_keys=sort_keys, indent=indent)

def write_pickle(obj, file: str, mode: str = 'wb') -> None:
    with open(file, mode) as f:
        pickle.dump(obj, f)

def writeTrainData(raw_stack, label_stack, resolution, stack_id):
    img_tr_itk = sitk.GetImageFromArray(raw_stack.astype(np.float32))
    lab_tr_itk = sitk.GetImageFromArray(label_stack) # synapse are foreground, cells background

    img_tr_itk.SetSpacing(resolution)
    lab_tr_itk.SetSpacing(resolution)

    sitk.WriteImage(img_tr_itk, join(imagestr, "training%d_0000.nii.gz"%stack_id))
    sitk.WriteImage(lab_tr_itk, join(labelstr, "training%d.nii.gz"%stack_id))

def writeTestData(raw_stack, label_stack, resolution, stack_id):
    img_te_itk = sitk.GetImageFromArray(raw_stack.astype(np.float32))
    lab_te_itk = sitk.GetImageFromArray(label_stack) # synapse are foreground, cells background

    img_te_itk.SetSpacing(resolution)
    lab_te_itk.SetSpacing(resolution)

    sitk.WriteImage(img_te_itk, join(imagests, "testing%d_0000.nii.gz"%stack_id))
    sitk.WriteImage(lab_te_itk, join(labelsts, "testing%d.nii.gz"%stack_id))


if __name__ == "__main__":

    task_id = 603
    task_name = "synapse178synins"
    foldername = "Task%03.0d_%s" % (task_id, task_name)

    out_base = join(nnUNet_raw_data, foldername)
    imagestr = join(out_base, "imagesTr")
    imagests = join(out_base, "imagesTs")
    labelstr = join(out_base, "labelsTr")
    labelsts = join(out_base, "labelsTs")
    os.makedirs(imagestr, exist_ok=True)
    os.makedirs(imagests, exist_ok=True)
    os.makedirs(labelstr, exist_ok=True)
    os.makedirs(labelsts, exist_ok=True)

    root_path = "/data2/share/for_gjy/SynReEM/datasets/Synapse178/OriData/"

    resolution = (4, 4, 50)

    raw_stack1 = io.imread(os.path.join(root_path, 'synapse178_images_000.tif'))
    raw_stack2 = io.imread(os.path.join(root_path, 'synapse178_images_001.tif'))
    raw_stack3 = io.imread(os.path.join(root_path, 'synapse178_images_002.tif'))
    raw_stack4 = io.imread(os.path.join(root_path, 'synapse178_images_003.tif'))
    raw_stack5 = io.imread(os.path.join(root_path, 'synapse178_images_004.tif'))
    raw_stack6 = io.imread(os.path.join(root_path, 'synapse178_images_005.tif'))

    label_stack1 = io.imread(os.path.join(root_path, 'synapse178_labels_000_encode.tif'))
    # label_stack1[label_stack1>0] = 1
    label_stack2 = io.imread(os.path.join(root_path, 'synapse178_labels_001_encode.tif'))
    # label_stack2[label_stack2>0] = 1
    label_stack3 = io.imread(os.path.join(root_path, 'synapse178_labels_002_encode.tif'))
    # label_stack3[label_stack3>0] = 1
    label_stack4 = io.imread(os.path.join(root_path, 'synapse178_labels_003_encode.tif'))
    # label_stack4[label_stack4>0] = 1
    label_stack5 = io.imread(os.path.join(root_path, 'synapse178_labels_004_encode.tif'))
    # label_stack5[label_stack5>0] = 1
    label_stack6 = io.imread(os.path.join(root_path, 'synapse178_labels_005_encode.tif'))
    # label_stack6[label_stack6>0] = 1

    writeTrainData(raw_stack=raw_stack1, label_stack=label_stack1, resolution=resolution, stack_id=0)
    writeTrainData(raw_stack=raw_stack2, label_stack=label_stack2, resolution=resolution, stack_id=1)
    writeTrainData(raw_stack=raw_stack3, label_stack=label_stack3, resolution=resolution, stack_id=2)
    writeTrainData(raw_stack=raw_stack4, label_stack=label_stack4, resolution=resolution, stack_id=3)
    writeTrainData(raw_stack=raw_stack5, label_stack=label_stack5, resolution=resolution, stack_id=4)
    writeTestData(raw_stack=raw_stack6, label_stack=label_stack6, resolution=resolution, stack_id=5)

    json_dict = OrderedDict()
    json_dict['name'] = task_name
    json_dict['description'] = task_name
    json_dict['tensorImageSize'] = "4D"
    json_dict['reference'] = "see challenge website"
    json_dict['licence'] = "see challenge website"
    json_dict['release'] = "0.0"
    json_dict['modality'] = {
        "0": "EM",
    }
    json_dict['labels'] = {i: str(i) for i in range(5)}

    json_dict['numTraining'] = 5
    json_dict['numTest'] = 1
    json_dict['training'] = [{'image': "./imagesTr/training%d.nii.gz" % i, "label": "./labelsTr/training%d.nii.gz" % i} for i in
                             range(json_dict['numTraining'])]
    json_dict['test'] = ["./imagesTs/testing%d_0000.nii.gz"%5]

    save_json(json_dict, os.path.join(out_base, "dataset.json"))


    # set splits manually
    out_preprocessed = join(preprocessing_output_dir, foldername)
    os.makedirs(out_preprocessed, exist_ok=True)
    # manual splits. we train 5 models on all three datasets

    splits = [{'train': ["training1", "training2", "training3", "training4"], 'val': ["training0",]},
              {'train': ["training0", "training2", "training3", "training4"], 'val': ["training1",]},
              {'train': ["training0", "training1", "training3", "training4"], 'val': ["training2",]},
              {'train': ["training0", "training1", "training2", "training4"], 'val': ["training3",]},
              {'train': ["training0", "training1", "training2", "training3"], 'val': ["training4",]},]
    write_pickle(splits, join(out_preprocessed, "splits_final.pkl"))
    print("OK")

    