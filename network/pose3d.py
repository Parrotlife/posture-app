import logging
from collections import defaultdict

import torch
from .pose3d_architecture import LinearModel

from .pose3d_lib import full_pose_rotation
from .pose3d_lib import project_pose_2cam
from .pose2d_lib import generate_3D_model
import numpy as np
import math


class Pose3d:
    """Pose3d class. From 2D joints to 3D joints"""
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    def __init__(self, model, input_size = 59, output_size = 9, device=None):

        if not device:
            self.device = torch.device('cpu')
        else:
            self.device = device

        # if the path is provided load the model parameters
        if isinstance(model, str):
            model_path = model
            if int(model_path[-5])>8:
                self.model = LinearModel(42, output_size, num_stage=5)
            elif int(model_path[-5])>3:
                self.model = LinearModel(input_size, output_size+9, num_stage=5)
            else:    
                self.model = LinearModel(input_size, output_size, num_stage=3)
            
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))

        # if the model is directly provided
        else:
            self.model = model
        self.model.eval()  # Default is train
        self.model.to(self.device)


    def forward(self, keypoints):
        """forward pass of pose3d network"""
  
        with torch.no_grad():
            inputs = keypoints
            inputs = torch.from_numpy(inputs).float().to(self.device)
            
            #  Don't use dropout for the mean prediction
            outputs = self.model(inputs)

        angles = outputs
        angles.cpu().detach().numpy()
        
        return angles

    @staticmethod
    def post_process(outputs, body_props, face_pos, width=192, height=108):

        LOGI_KK = [[0.7477614068838029, -0.0008111695246478873,  0.5134789751408451],
           [0.0,                      1.33178042553125, 0.45684016428124996],
           [0.                ,                     0.,                  1.]]


        kk = [[x*width  for x in LOGI_KK[0]],
            [x*height for x in LOGI_KK[1]],
            LOGI_KK[2]]

        dic_out = defaultdict(list)
        if outputs is None:
            return dic_out
        
        base_3d_pose = generate_3D_model(body_props, face_pos)

        for output in outputs:
            if len(output)==9:
                (a0,a1,a2,a3,a4,a5,a6,a7,a8) = output
                angle_dic = {'head':[a0,a1,a2], 'back':[a3,a4,a5], 'x':[a6],'y':[a7], 'z':[a8]}
                dic_out['angles'].append(angle_dic)
                dic_out['pose'].append(full_pose_rotation(base_3d_pose.reshape((21,3)).transpose(), angle_dic).transpose().flatten())
            else:

                joints_2_keep = ['left hip', 'right hip', 'nose', 'right shoulder','left shoulder', 'right eye','left eye',\
                               'right ear','left ear','center shoulder','center hip','center back','head', 'right elbow',\
                               'right wrist', 'left elbow', 'left wrist']

                (a0,a1,a2,a3,a4,a5,a6,a7,a8, a9, a10, a11, a12, a13, a14, a15, a16, a17) = output
                angle_dic = {'head':[a0,a1,a2], 'back':[a3,a4,a5], 'x':[a6],'y':[a7], 'z':[a8], 'left arm':[a9, a10, a11, a12], 'right arm':[a13, a14, a15, a16]}
                dic_out['angles'].append(angle_dic)
                moved_pose = full_pose_rotation(base_3d_pose.reshape((21,3)).transpose(), angle_dic)
                dic_out['pose'].append(moved_pose.transpose().flatten())
                dic_out['2d pose'].append(project_pose_2cam(moved_pose, kk, a17, joints_2_keep))

        return dic_out

    