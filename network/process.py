
import json
import numpy as np
from functools import reduce

#import .pose2d_lib as lib_2d
from .pose2d_lib import convert_pifpaf, filter_joints_2d, NB_JOINTS, normalize_2d


import torch
import torchvision

from ..utils import get_keypoints, pixel_to_camera

"""Monoloco functions"""
###########################################################################################################################
def preprocess_monoloco(keypoints, kk):

    """ Preprocess batches of inputs
    keypoints = torch tensors of (m, 3, 17)  or list [3,17]
    Outputs =  torch tensors of (m, 34) in meters normalized (z=1) and zero-centered using the center of the box
    """
    if isinstance(keypoints, list):
        keypoints = torch.tensor(keypoints)
    if isinstance(kk, list):
        kk = torch.tensor(kk)
    # Projection in normalized image coordinates and zero-center with the center of the bounding box
    uv_center = get_keypoints(keypoints, mode='center')
    xy1_center = pixel_to_camera(uv_center, kk, 10)
    xy1_all = pixel_to_camera(keypoints[:, 0:2, :], kk, 10)
    kps_norm = xy1_all - xy1_center.unsqueeze(1)  # (m, 17, 3) - (m, 1, 3)
    kps_out = kps_norm[:, :, 0:2].reshape(kps_norm.size()[0], -1)  # no contiguous for view
    return kps_out

def laplace_sampling(outputs, n_samples):

    # np.random.seed(1)
    mu = outputs[:, 0]
    bi = torch.abs(outputs[:, 1])

    # Analytical
    # uu = np.random.uniform(low=-0.5, high=0.5, size=mu.shape[0])
    # xx = mu - bi * np.sign(uu) * np.log(1 - 2 * np.abs(uu))

    # Sampling
    cuda_check = outputs.is_cuda
    if cuda_check:
        get_device = outputs.get_device()
        device = torch.device(type="cuda", index=get_device)
    else:
        device = torch.device("cpu")

    laplace = torch.distributions.Laplace(mu, bi)
    xx = laplace.sample((n_samples,)).to(device)

    return xx


def unnormalize_bi(outputs):
    """Unnormalize relative bi of a nunmpy array"""

    outputs[:, 1] = torch.exp(outputs[:, 1]) * outputs[:, 0]
    return outputs


def preprocess_pifpaf_1(annotations, im_size=None):
    """
    Preprocess pif annotations:
    1. enlarge the box of 10%
    2. Constraint it inside the image (if image_size provided)
    """

    boxes = []
    keypoints = []

    for dic in annotations:
        box = dic['bbox']
        if box[3] < 0.5:  # Check for no detections (boxes 0,0,0,0)
            return [], []

        kps = prepare_pif_kps(dic['keypoints'])
        conf = float(np.sort(np.array(kps[2]))[-3])  # The confidence is the 3rd highest value for the keypoints

        # Add 15% for y and 20% for x
        delta_h = (box[3] - box[1]) / 7
        delta_w = (box[2] - box[0]) / 3.5
        assert delta_h > -5 and delta_w > -5, "Bounding box <=0"
        box[0] -= delta_w
        box[1] -= delta_h
        box[2] += delta_w
        box[3] += delta_h

        # Put the box inside the image
        if im_size is not None:
            box[0] = max(0, box[0])
            box[1] = max(0, box[1])
            box[2] = min(box[2], im_size[0])
            box[3] = min(box[3], im_size[1])

        box.append(conf)
        boxes.append(box)
        keypoints.append(kps)

    return boxes, keypoints


def prepare_pif_kps(kps_in):
    """Convert from a list of 51 to a list of 3, 17"""

    assert len(kps_in) % 3 == 0, "keypoints expected as a multiple of 3"
    xxs = kps_in[0:][::3]
    yys = kps_in[1:][::3]  # from offset 1 every 3
    ccs = kps_in[2:][::3]

    return [xxs, yys, ccs]


def image_transform(image):

    normalize = torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), normalize, ])
    return transforms(image)

"""pose3d Functions"""
###########################################################################################################################

"""all the prepocessing of pifpaf for getting pose3d inputs"""
def preprocess_pifpaf_2(annotations, body_props, face_pos, kk, arms = False):

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

        #new['confidence'] = keyp[2::3]

        del keyp[2::3]

        keyp = np.array(keyp).reshape((int(len(keyp)/2),2)).transpose()

        keyp = convert_pifpaf(keyp, kk)

        keyp = filter_joints_2d(keyp.reshape((NB_JOINTS,2)).transpose(), joints2keep, 0)
        keyp = normalize_2d(keyp)
        keyp = filter_joints_2d(keyp, joints2keep, -10).transpose().flatten()

        keyp = np.concatenate((keyp, props), axis = 0)

        inputs.append(keyp)
    
    return np.array(inputs)

