
import json
import numpy as np
from functools import reduce
from .pose2d_lib import convert_pifpaf, filter_joints_2d, NB_JOINTS, normalize_2d
import torch
import torchvision
from ..utils import get_keypoints, pixel_to_camera

"""process functions for images and PifPaf outputs"""

def image_transform(image):
    """Normalize the image for pipaf"""
    normalize = torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), normalize, ])
    return transforms(image)

def preprocess_pifpaf(annotations, body_props, face_pos, kk, arms = False):
    """all the prepocessing of pifpaf for getting pose3d inputs"""

    joints2keep = ['left hip', 'right hip', 'nose', 'right shoulder','left shoulder', 'right eye','left eye',
               'right ear','left ear','center shoulder','center hip','center back','head']
    if arms:
        joints2keep = ['left hip', 'right hip', 'nose', 'right shoulder','left shoulder', 'right eye','left eye',
               'right ear','left ear','center shoulder','center hip','center back','head', 'right elbow',
              'right wrist', 'left elbow', 'left wrist']

    props = [x[1] for x in body_props.items()]+reduce(lambda x, y: x+y, [x[1] for x in face_pos.items()])

    inputs = []

    for dic in annotations:

        keyp = [x for x in dic['keypoints']]

        #new['confidence'] = keyp[2::3] ## if you need the pifpaf confidence as part of the algorithm

        del keyp[2::3]

        keyp = np.array(keyp).reshape((int(len(keyp)/2),2)).transpose()

        keyp = convert_pifpaf(keyp, kk)

        keyp = filter_joints_2d(keyp.reshape((NB_JOINTS,2)).transpose(), joints2keep, 0)
        keyp = normalize_2d(keyp)
        keyp = filter_joints_2d(keyp, joints2keep, -10).transpose().flatten()

        keyp = np.concatenate((keyp, props), axis = 0)

        inputs.append(keyp)
    
    return np.array(inputs)

