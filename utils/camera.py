
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def pixel_to_camera(uv_tensor, kk, z_met):
    """
    Convert a tensor in pixel coordinate to absolute camera coordinates
    It accepts lists or torch/numpy tensors of (m, 2) or (m, x, 2)
    where x is the number of keypoints
    """
    if isinstance(uv_tensor, (list, np.ndarray)):
        uv_tensor = torch.tensor(uv_tensor)
    if isinstance(kk, list):
        kk = torch.tensor(kk)
    if uv_tensor.size()[-1] != 2:
        uv_tensor = uv_tensor.permute(0, 2, 1)  # permute to have 2 as last dim to be padded
        assert uv_tensor.size()[-1] == 2, "Tensor size not recognized"
    uv_padded = F.pad(uv_tensor, pad=(0, 1), mode="constant", value=1)  # pad only last-dim below with value 1

    kk_1 = torch.inverse(kk)
    xyz_met_norm = torch.matmul(uv_padded, kk_1.t())  # More general than torch.mm
    xyz_met = xyz_met_norm * z_met

    return xyz_met

def get_keypoints(keypoints, mode):
    """
    Extract center, shoulder or hip points of a keypoint
    Input --> list or torch/numpy tensor [(m, 3, 17) or (3, 17)]
    Output --> torch.tensor [(m, 2)]
    """
    if isinstance(keypoints, (list, np.ndarray)):
        keypoints = torch.tensor(keypoints)
    if len(keypoints.size()) == 2:  # add batch dim
        keypoints = keypoints.unsqueeze(0)
    assert len(keypoints.size()) == 3 and keypoints.size()[1] == 3, "tensor dimensions not recognized"
    assert mode in ['center', 'bottom', 'head', 'shoulder', 'hip', 'ankle']

    kps_in = keypoints[:, 0:2, :]  # (m, 2, 17)
    if mode == 'center':
        kps_max, _ = kps_in.max(2)  # returns value, indices
        kps_min, _ = kps_in.min(2)
        kps_out = (kps_max - kps_min) / 2 + kps_min   # (m, 2) as keepdims is False

    elif mode == 'bottom':  # bottom center for kitti evaluation
        kps_max, _ = kps_in.max(2)
        kps_min, _ = kps_in.min(2)
        kps_out_x = (kps_max[:, 0:1] - kps_min[:, 0:1]) / 2 + kps_min[:, 0:1]
        kps_out_y = kps_max[:, 1:2]
        kps_out = torch.cat((kps_out_x, kps_out_y), -1)

    elif mode == 'head':
        kps_out = kps_in[:, :, 0:5].mean(2)

    elif mode == 'shoulder':
        kps_out = kps_in[:, :, 5:7].mean(2)

    elif mode == 'hip':
        kps_out = kps_in[:, :, 11:13].mean(2)

    elif mode == 'ankle':
        kps_out = kps_in[:, :, 15:17].mean(2)

    return kps_out 

def xyz_from_distance(distances, xy_centers):
    """
    From distances and normalized image coordinates (z=1), extract the real world position xyz
    distances --> tensor (m,1) or (m) or float
    xy_centers --> tensor(m,3) or (3)
    """

    if isinstance(distances, float):
        distances = torch.tensor(distances).unsqueeze(0)
    if len(distances.size()) == 1:
        distances = distances.unsqueeze(1)
    if len(xy_centers.size()) == 1:
        xy_centers = xy_centers.unsqueeze(0)

    assert xy_centers.size()[-1] == 3 and distances.size()[-1] == 1, "Size of tensor not recognized"

    return xy_centers * distances / torch.sqrt(1 + xy_centers[:, 0:1].pow(2) + xy_centers[:, 1:2].pow(2))


# def open_image(path_image):
#     with open(path_image, 'rb') as f:
#         pil_image = Image.open(f).convert('RGB')
#         return pil_image
