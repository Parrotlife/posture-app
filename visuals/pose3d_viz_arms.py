from panda3d.core import *
import math
from direct.showbase.ShowBase import ShowBase
import numpy as np
from direct.task import Task
from .segment import Segment, Create3DPoint
import torch
import cv2
import gltf
import copy as cp
import os
import time
from ..network import PifPaf, Pose3d
from ..network.pose2d_lib import get_front_prop
from ..network.process import preprocess_pifpaf, image_transform
from ..visuals.client import *
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from pyquaternion import Quaternion
from direct.gui.OnscreenText import OnscreenText
import keyboard
import json
import pickle as pkl

###avatar props###
AVATAR_BODY_PROPS = {'neck': 0.07729765064733189,
            'shoulder': 0.13887964509275452,
            'biceps': 0.1903499496025787,
            'forearm': 0.1856855142453098,
            'quad': 0.2250013372092681,
            'tibias': 0.2199742522543633,
            'hip': 0.16409048728199763,
            'back': 0.22711310836812693}

AVATAR_FACE_POS = {'nose': [0.0, 0.0, 0.07331183671724711],
            'eye': [0.02339558783308983, 0.02517361111111116, 0.058635183635287635],
            'ear': [0.05771397604574112, 0.0, 0.0]}

AVATAR_SCALE_FACTOR = 100

### mean props ###
body_props = {'neck': 0.127,
              'shoulder': 0.22,
              'biceps': 0.3,
              'forearm': 0.27,
              'quad': 0.37,
              'tibias': 0.35,
              'hip': 0.338,
              'back': 0.339
             }
face_pos =   {'nose': [0.0, 0.0, 0.119],
              'ear': [0.0293, 0.036, 0.06],
              'eye': [0.0805, 0.0, 0.0]
             }


LOGI_KK = [[0.7477614068838029, -0.0008111695246478873,  0.5134789751408451],
           [0.0,                      1.33178042553125, 0.45684016428124996],
           [0.                ,                     0.,                  1.]]

class Pose3dViz(ShowBase):
    """ Initialize the ShowBase class from which we inherit, which will
        create a window and set up everything we need for rendering into it."""
    
    def __init__(self, body_props, face_pos, cam, args, pifpaf, pose3d, with_azure = False):

        ShowBase.__init__(self)

        # Set up Panda3d environment
        self.disableMouse()
        self.setBackgroundColor(0.5, 0.6, 0.9)
        self.setupLights()

        ### initialize all the required variables ###

        # set face and body proportions to construct the avatar
        self.body_props = body_props
        self.face_pos = face_pos

        # set the input video capture
        self.cam = cam

        # initialise the predictive models
        self.pifpaf = pifpaf
        self.pose3d = pose3d

        self.args = args

        # moving average variables and parameters
        self.buffer_counter = 0
        self.buffer_size = 5
        self.buffer = np.zeros((17,self.buffer_size))
        
        #kinect azure boolean
        self.with_azure = with_azure

        # variables to estimate frame rate
        self.iter_count = 0
        self.frame_rate = 0
        self.previous_time = time.time()
        
        # error text initializing
        if self.with_azure:
            self.text = OnscreenText(text='my text string', pos=(-0.5, 0.02), scale=0.07)
        
        # recording attributes
        self.stored_vals = {"azure":{"head":[], "back":[], "orient":[]},
                            "pose3d":{"head":[], "back":[], "orient":[]}}
        self.stored_kinect_data = []
        self.stored_pifpaf = []
        self.record_id = str(int(time.time()))

        # create the folders to store the recorded results of the session
        os.mkdir("posture-app/results/"+self.record_id)
        os.mkdir("posture-app/results/"+self.record_id+"/images")
        self.record_count = 0
        self.is_recording = False

        # initialize the angle dicts
        angles = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, math.pi/2, math.pi/2, math.pi/2, 0, 0, 0, 0, 0, 0, 0, 0)
        
        self.angle_dic = {'head':angles[:3], 'back':angles[3:6], 'x':[angles[6]],'y':[angles[7]], 'z':[angles[8]],
                          'left arm':angles[9:13], 'right arm':angles[13:]}
        self.azure_angle_dic = {}

        # create the pose skeleton
        self.create_pose(self.body_props, self.face_pos, self.angle_dic, self.with_azure)
        
        # necessary to read blender 3D objects
        gltf.patch_loader(self.loader)

        # adds the body part blender models to the skeleton
        self.create_avatar()
        
        # creates 3D points for each joints (used only for debugging angle conversions)
        self.points = [Create3DPoint() for _ in range(21)]

        # adds the update function as a thread to be called at every step of the game engine
        self.taskMgr.add(self.update, "change_angles")
        
    def setupLights(self):
        """setup the lighting"""

        ambientLight = AmbientLight("ambientLight")
        ambientLight.setColor((.3, .3, .3, 1))
        directionalLight = DirectionalLight("directionalLight")
        directionalLight.setDirection(LVector3(0, 0, -2.5))
        directionalLight.setColor((0.9, 0.8, 0.9, 1))
        render.setLight(render.attachNewNode(ambientLight))
        render.setLight(render.attachNewNode(directionalLight))
     
    def create_pose(self, body_props, face_pos, angle_dic, with_azure = True):
        """creates the skeleton of the avatar with given proportions and initial angles"""
        
        #check if azure is connected
        segments_dic = {}
        if with_azure:
            segments_azure_dic = {}
        

        #define the eye distance and rotation
        len_eye = np.linalg.norm(face_pos['eye'])
        eye_z_rot = (180/np.pi)*math.acos(face_pos['eye'][1]/len_eye)
        eye_y_rot = (180/np.pi)*math.acos(face_pos['eye'][0]/len_eye)

        #define all the skeleton segments data as follow: 

        #   (segment_name,      segment_length,          parent_name,      distance_from_parent,    rotation_angles)
        seg_data = [
            ('left_hip',        0.5*body_props['hip'],   render,            0,                      [0, 0, 270]), 
            ('right_hip',       0.5*body_props['hip'],   'left_hip',        0,                      [0, 180, 0]), 
            ('lower_back',      body_props['back'],      'left_hip',        0,                      [0, 90, 0]), 
            ('upper_back',      body_props['back'],      'lower_back',      body_props['back'],     [0, 0, 0]), 
            ('left_shoulder',   body_props['shoulder'],  'upper_back',      body_props['back'],     [0, 90, 0]), 
            ('left_biceps',     body_props['biceps'],    'left_shoulder',   body_props['shoulder'], [0, 90, 0]), 
            ('left_forearm',    body_props['forearm'],   'left_biceps',     body_props['biceps'],   [0, 0, 90]),
            ('right_shoulder',  body_props['shoulder'],  'upper_back',      body_props['back'],     [0, 270, 0]), 
            ('right_biceps',    body_props['biceps'],    'right_shoulder',  body_props['shoulder'], [0, 270, 0]), 
            ('right_forearm',   body_props['forearm'],   'right_biceps',    body_props['biceps'],   [0, 0, 90]), 
            ('lower_neck',      body_props['neck'],      'upper_back',      body_props['back'],     [0, 0, 0]), 
            ('upper_neck',      body_props['neck'],      'lower_neck',      body_props['neck'],     [0, 0, 0]),
            ('nose',           face_pos['nose'][2],     'upper_neck',      body_props['neck'],     [0, 0, 90]),
            ('r_eye',           len_eye,                 'upper_neck',      body_props['neck'],     [0, eye_y_rot, eye_z_rot]),
            ('l_eye',           len_eye,                 'upper_neck',      body_props['neck'],     [0, -eye_y_rot, eye_z_rot]),
            ('r_ear',           face_pos['ear'][0],      'upper_neck',      body_props['neck'],     [0, 90, 0]),
            ('l_ear',           face_pos['ear'][0],      'upper_neck',      body_props['neck'],     [0, -90, 0])
            ]
        
        # generate the skeleton for both pose3d and azure if azure is connected
        for name, length, parent, z_displacement, rots in seg_data:
            segments_dic[name] = Segment(length=AVATAR_SCALE_FACTOR*length).draw()
            if with_azure:
                segments_azure_dic[name] = Segment(length=AVATAR_SCALE_FACTOR*length).draw()
            
            if type(parent)==str:
                parent_obj = segments_dic[parent]
                if with_azure:
                    parent_azure_obj = segments_azure_dic[parent]
            else:
                parent_obj = parent
                if with_azure:
                    parent_azure_obj = parent
            
            segments_dic[name].reparentTo(parent_obj)
            segments_dic[name].setPosHprScale(0, 0, AVATAR_SCALE_FACTOR*z_displacement, rots[0], rots[1], rots[2], 1., 1., 1.)

            if with_azure:
                segments_azure_dic[name].reparentTo(parent_azure_obj)
                segments_azure_dic[name].setPosHprScale(0, 0, AVATAR_SCALE_FACTOR*z_displacement, rots[0], rots[1], rots[2], 1., 1., 1.)
        

        self.segments_dic = segments_dic
        if with_azure:
            self.segments_azure_dic = segments_azure_dic
        else:
            self.segments_azure_dic = None

    def create_avatar(self):
        """connect the different body part models to the skeleton"""

        ##  (part name,         parent name,        [angles, scaling])
        avatar_data = [
            ('head',            'upper_neck',       [270,0+180,180,20,20,20]),
            ('neck',            'lower_neck',       [270,30+180,180,15,15,15]),
            ('torso',           'upper_back',       [270,0+180,180,25,15,25]),
            ('rightarm',        'right_biceps',     [270,0,90,20,15,15]),
            ('rightforearm',    'right_forearm',    [270,180,90,20,15,15]),
            ('leftarm',         'left_biceps',      [270,180,90,20,15,15]),
            ('leftforearm',     'left_forearm',     [270,0,90,20,15,15]),
            ('spine',           'lower_back',       [270,0+180,180,15,15,45]),
            ('pelvis',          'lower_back',       [270,0+180,180,25,15,15])
        ]

        # attach each model part to the corresponding skeleton for pose3d and azure
        avatar_dic = {}
        for name, parent, rs in avatar_data:
            avatar_dic[name] = self.loader.loadModel('visuals/models3d/man3D.'+name+'.glb')
            avatar_dic[name].reparentTo(self.segments_dic[parent])
            avatar_dic[name].setHprScale(rs[0],rs[1],rs[2],rs[3],rs[4],rs[5])
        self.avatar_dic = avatar_dic

        if self.segments_azure_dic:
            
            # change the skeleton color for azure and move the skeleton and camera
            self.segments_azure_dic['left_hip'].setColor(0.0, 0.9, 0.9, 0.0)
            self.segments_azure_dic['left_hip'].setPos(50, 0, 0)
            self.segments_dic['left_hip'].setPos(-50, 0, 0)
            camera.setPosHpr(25, -20, 200, 0, -90, 180)

            avatar_azure_dic = {}
            for name, parent, rs in avatar_data:
                avatar_azure_dic[name] = self.loader.loadModel('visuals/models3d/man3D.'+name+'.glb')
                avatar_azure_dic[name].reparentTo(self.segments_azure_dic[parent])
                avatar_azure_dic[name].setHprScale(rs[0],rs[1],rs[2],rs[3],rs[4],rs[5])
            self.avatar_azure_dic = avatar_azure_dic
   
    def process_azure_data(self, azure_data):
        """process the azure data to convert it into a corresponding format for the avatar
            (works well for the head, back and absolute orientation but can still not working perfectly for arms)"""
        
        def angle_from_3pts(a, b, c):
            """get the acute angle between 3 3D points (used for the elbow angle)"""
            ba = a - b
            bc = c - b

            cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
            angle = np.arccos(cosine_angle)

            return np.degrees(angle)

        azure_angles_dic = {}

        #find absolute xyz
        xyz = azure_data[0,3:7]
        xyz = R.from_quat(xyz).as_euler('xyz', degrees=True)+np.array([90,0,0])

        azure_angles_dic['x'] = -xyz[0]
        azure_angles_dic['y'] = xyz[1]
        azure_angles_dic['z'] = xyz[2]

       
        #find the back angles
        key_rots = R.from_quat(azure_data[1:3,3:7])
        key_times = [0, 1]
        back_slerp = Slerp(key_times, key_rots)
        back_azure = back_slerp([0.5])[0].as_euler('xyz', degrees=True)+np.array([90,0,90])
        back_azure = np.array([-back_azure[2], back_azure[0], -back_azure[1]])

        azure_angles_dic['back'] = back_azure

        #find the head angles
        head_azure = azure_data[26,3:7]
        head_azure = R.from_quat(head_azure).as_euler('xyz', degrees=True)+np.array([90,45,90])
        head_azure = [-head_azure[2], -head_azure[0], -head_azure[1]]

        azure_angles_dic['head'] = head_azure

        #find the left shoulder angles

        left_shoulder_azure = Quaternion(azure_data[5,3:7])#*Quaternion(azure_data[4,3:7]).inverse
        left_shoulder_azure = R.from_quat(left_shoulder_azure.elements).as_euler('zyx', degrees=True)+np.array([0,0,180])
        left_shoulder_azure = [left_shoulder_azure[0], -left_shoulder_azure[2], left_shoulder_azure[1]]
        azure_angles_dic['left arm'] = left_shoulder_azure

        #find the right shoulder angles

        right_shoulder_azure = Quaternion(azure_data[12,3:7])#*Quaternion(azure_data[11,3:7]).inverse
        right_shoulder_azure = R.from_quat(right_shoulder_azure.elements).as_euler('zyx', degrees=True)+np.array([0,0,180])
        right_shoulder_azure = [-right_shoulder_azure[0], -right_shoulder_azure[2], -right_shoulder_azure[1]]
        azure_angles_dic['right arm'] = right_shoulder_azure

        #find the left elbow angles
        left_elbow_azure = -angle_from_3pts(azure_data[5,:3],azure_data[6,:3], azure_data[7,:3])+180
        azure_angles_dic['left arm'].append(left_elbow_azure)

        #find the right elbow angles
        right_elbow_azure = angle_from_3pts(azure_data[12,:3],azure_data[13,:3], azure_data[14,:3])+180
        azure_angles_dic['right arm'].append(right_elbow_azure)

        #change the reference 
        azure_angles_dic['back'] = np.array(back_azure)+np.array([azure_angles_dic['z'], azure_angles_dic['x'], azure_angles_dic['y']])+np.array([90,0,0])
        azure_angles_dic['head'] = np.array(head_azure)-np.array([back_azure[0], -back_azure[1], back_azure[2]])


        return azure_angles_dic

    def process_pose3d_data(self, pose3d_outputs):
        """convert from pose3d angles format to avatar angles"""

        a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17 = pose3d_outputs[0]

        a0, a1, a2, a3, a4, a5, a6, a7, a8 = ( pose3d_outputs[0][0], -pose3d_outputs[0][1], -pose3d_outputs[0][2], pose3d_outputs[0][3], -pose3d_outputs[0][4],\
            -pose3d_outputs[0][5], -pose3d_outputs[0][8], pose3d_outputs[0][7]+math.pi/2, -pose3d_outputs[0][6])

        a9, a10, a11, a12, a13, a14, a15, a16 = (pose3d_outputs[0][9], pose3d_outputs[0][10], pose3d_outputs[0][11], pose3d_outputs[0][12], pose3d_outputs[0][13],\
            pose3d_outputs[0][14], pose3d_outputs[0][15], pose3d_outputs[0][16])


        self.buffer[:,self.buffer_counter%self.buffer_size] = (a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16)
        self.buffer_counter+=1
        
        a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16 = np.mean(self.buffer, axis = 1)

        self.angle_dic = {'head':[a0,a1,a2], 'back':[a3,a4,a5], 'x':[a6],'y':[a7], 'z':[a8], 'left arm':[a9, a10, a11, a12], 'right arm':[a13, a14, a15, a16]}


        self.angle_dic['head'][0], self.angle_dic['head'][1], self.angle_dic['head'][2] = (180/np.pi)*self.angle_dic['head'][1],\
            -(180/np.pi)*self.angle_dic['head'][2], -(180/np.pi)*self.angle_dic['head'][0]
        self.angle_dic['back'][0], self.angle_dic['back'][1], self.angle_dic['back'][2] = (180/np.pi)*self.angle_dic['back'][1],\
            -(180/np.pi)*self.angle_dic['back'][2], -(180/np.pi)*self.angle_dic['back'][0]
        self.angle_dic['x'], self.angle_dic['y'], self.angle_dic['z'] = -(180/np.pi)*self.angle_dic['x'][0], -(180/np.pi)*self.angle_dic['z'][0]+20,\
                 -(180/np.pi)*self.angle_dic['y'][0]
        self.angle_dic['left arm'][0], self.angle_dic['left arm'][1], self.angle_dic['left arm'][2], self.angle_dic['left arm'][3] =\
             ((180/np.pi)*self.angle_dic['left arm'][0])-90,(180/np.pi)*self.angle_dic['left arm'][1],\
             (180/np.pi)*self.angle_dic['left arm'][2], (180/np.pi)*self.angle_dic['left arm'][3]
        self.angle_dic['right arm'][0], self.angle_dic['right arm'][1], self.angle_dic['right arm'][2], self.angle_dic['right arm'][3] = \
             ((180/np.pi)*self.angle_dic['right arm'][0])-90, (180/np.pi)*self.angle_dic['right arm'][1],\
                  (180/np.pi)*self.angle_dic['right arm'][2], (180/np.pi)*self.angle_dic['right arm'][3]

    def keyboard_action_check(self):
        """Check for actions and update the object accordingly"""
        ### recording ###
        # for start and stop recording you press a then r
        if keyboard.is_pressed('a'):
            keyboard.wait('r')
            self.recording_text = OnscreenText(text='', pos=( 0.0, -0.4), scale=0.07)
            if self.is_recording:
                self.recording_text.destroy()
                saving_text = OnscreenText(text='saving', pos=( 0.0, -0.4), scale=0.07)

                # if previous state was recording then we store the data in json tables and reset the variables
                json.dump(self.stored_vals, open('posture-app/results/'+self.record_id+'/pose3d_data.txt', 'w'))
                json.dump(np.array(self.stored_kinect_data).tolist(), open('posture-app/results/'+self.record_id+'/kinect_data.txt', 'w'))
                json.dump(self.stored_pifpaf, open('posture-app/results/'+self.record_id+'/pifpaf_data.txt', 'w'))


                self.stored_vals = {"azure":{"head":[], "back":[], "orient":[]},
                            "pose3d":{"head":[], "back":[], "orient":[]}}
                self.stored_kinect_data = []
                self.stored_pifpaf = []
                self.record_id = str(int(time.time()))
                self.record_count = 0
                saving_text.destroy()
            else:
                self.recording_text = OnscreenText(text='recording', pos=( 0.0, -0.4), scale=0.07)

            self.is_recording = not self.is_recording

    def get_frame_rate(self):
        """gives back the frame rate of the model"""
        if time.time() > (self.previous_time+1):
            self.frame_rate = self.iter_count
            self.iter_count = 0
            self.previous_time = time.time()
            
        self.iter_count+=1

    def update_segment_dic(self, segments_dic, angles_dic):
        """Update the angles of a segment dictionary with new angles"""
        
        segments_dic['upper_neck'].setPosHprScale(0, 0, AVATAR_SCALE_FACTOR*(self.body_props['neck']), angles_dic['head'][0],\
        angles_dic['head'][1], angles_dic['head'][2], 1., 1., 1.)

        segments_dic['upper_back'].setPosHprScale(0, 0, AVATAR_SCALE_FACTOR*(self.body_props['back']), angles_dic['back'][0],\
        angles_dic['back'][1], angles_dic['back'][2], 1., 1., 1.)

        segments_dic['left_hip'].setHpr( angles_dic['x'], angles_dic['y'], angles_dic['z'])

        segments_dic['left_biceps'].setPosHprScale(0, 0, AVATAR_SCALE_FACTOR*(self.body_props['shoulder']),\
            angles_dic['left arm'][0], angles_dic['left arm'][1], angles_dic['left arm'][2], 1., 1., 1.)

        segments_dic['left_forearm'].setPosHprScale(0, 0, AVATAR_SCALE_FACTOR*(self.body_props['biceps']), 0, 0,\
                angles_dic['left arm'][3], 1., 1., 1.)
        
        segments_dic['right_biceps'].setPosHprScale(0, 0, AVATAR_SCALE_FACTOR*(self.body_props['shoulder']),\
            angles_dic['right arm'][0], angles_dic['right arm'][1], angles_dic['right arm'][2], 1., 1., 1.)

        segments_dic['right_forearm'].setPosHprScale(0, 0, AVATAR_SCALE_FACTOR*(self.body_props['biceps']), 0, 0,\
                angles_dic['right arm'][3], 1., 1., 1.)

    def get_pifpaf_values(self, args, pifpaf):
        """Process the cam image and get the main pifpaf detection and the intrinsic matrix"""
        # get the camera image and process it
        _, frame = self.cam.read()

        image = cv2.resize(frame, None, fx=args.scale, fy=args.scale)
        height, width, _ = image.shape

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        processed_image_cpu = image_transform(image.copy())
        processed_image = processed_image_cpu.contiguous().to(args.device, non_blocking=True)

        # run the pifpaf model
        fields = pifpaf.fields(torch.unsqueeze(processed_image, 0))[0]
        _, _, pifpaf_out = pifpaf.forward(image, processed_image_cpu, fields)

        if pifpaf_out:
            # will only leave the largest person detected in the frame
            pifpaf_out = [pifpaf_out[np.argmax(list(map(lambda pp: max(pp['keypoints'][1::3]) - min(pp['keypoints'][1::3]), pifpaf_out)))]]

        # define the intrinsic camera parameters    
        kk = [[x*width  for x in LOGI_KK[0]],
              [x*height for x in LOGI_KK[1]],
               LOGI_KK[2]]

        return pifpaf_out, kk, frame

    def store_data(self, azure_data, pifpaf_out, frame):
        """Calculate the error and store the values for the data recording"""
        if self.with_azure:
            azure_angles_dic = self.azure_angles_dic
            head_error = abs(self.angle_dic['head'] - azure_angles_dic['head'])
            back_error = abs(self.angle_dic['back'] - azure_angles_dic['back'])
            orient_error = np.array([abs(self.angle_dic['x'] - azure_angles_dic['x']),
                                    abs(self.angle_dic['y'] - azure_angles_dic['y']),
                                    abs(self.angle_dic['z'] - azure_angles_dic['z'])])

        if self.is_recording:
            self.stored_vals['pose3d']['head'].append(list(self.angle_dic['head']))
            self.stored_vals['pose3d']['back'].append(list(self.angle_dic['back']))
            self.stored_vals['pose3d']['orient'].append([self.angle_dic['x'],
                                                            self.angle_dic['y'],
                                                            self.angle_dic['z']])
            self.record_count += 1
            print("saving")
            cv2.imwrite("posture-app/results/"+self.record_id+"/images/frame{}.jpg".format(self.record_count), frame)

            if self.with_azure:
                self.stored_vals['azure']['head'].append(list(azure_angles_dic['head']))
                self.stored_vals['azure']['back'].append(list(azure_angles_dic['back']))
                self.stored_vals['azure']['orient'].append([azure_angles_dic['x'],
                                                            azure_angles_dic['y'],
                                                            azure_angles_dic['z']])
                self.stored_kinect_data.append(azure_data)
                self.stored_pifpaf.append(pifpaf_out[0])
            

        if self.with_azure:
            self.text.destroy()

            text_2_show = 'head: '+str(head_error.astype(int))+\
                        '\nback: '+str(back_error.astype(int))+\
                        '\norient: '+str(orient_error.astype(int))

            self.text = OnscreenText(text= text_2_show , pos=(0., -0.2), scale=0.07)

    def update(self, task):
        """Function to update the visualization and run the model"""
        

        self.keyboard_action_check()
        
        self.get_frame_rate()
        print("frame rate:", self.frame_rate)

        #############get Azure Values###########################################    
        if self.with_azure:
            azure_data = np.array(get_data()).reshape((32,8))
            
            self.azure_angles_dic = self.process_azure_data(azure_data)

            self.update_segment_dic(self.segments_azure_dic, self.azure_angles_dic)

        # get the models and parameters
        args = self.args
        pifpaf = self.pifpaf
        pose3d = self.pose3d

        # get the pifpaf values and the intrinsic matrix
        pifpaf_out, kk, frame = self.get_pifpaf_values(args, pifpaf)
        

        if pifpaf_out:

            # set camera position if kinect model is available
            if self.with_azure==False:
                camera.setPosHpr(0, -30, 300, 0, -90, 180)
            
            # preprocess the pifpaf outputs for the pose3d model
            pose3d_inputs = preprocess_pifpaf(pifpaf_out, self.body_props, self.face_pos, kk, True)

            # run the pose3d model
            pose3d_outputs = pose3d.forward(pose3d_inputs).cpu().detach().numpy()

            ########################process and store the angles for comparaison###################################

            self.process_pose3d_data(pose3d_outputs)

            self.segments_dic['left_hip'].setPos(0, 0, 0)

            self.update_segment_dic(self.segments_dic, self.angle_dic)

            self.store_data(azure_data, pifpaf_out, frame)

        return Task.cont
        

### This runs all the processes for the webcam inference
def webcam(args):

    # add args.device
    args.device = torch.device('cpu')
    if torch.cuda.is_available():
        args.device = torch.device('cuda')
    
    # load models
    args.camera = True
    pifpaf = PifPaf(args)
    pose3d = Pose3d(model=args.pose3d_model, device=args.device)

    # # # Start recording
    cam = cv2.VideoCapture(0)

    # Check if the kinect azure is sending info
    with_azure = True
    try:
        get_data()
    except:
        print("Azure not connected")
        with_azure = False

    # initialize the Graphic engine and sets up the model
    demo = Pose3dViz(AVATAR_BODY_PROPS, AVATAR_FACE_POS, cam, args, pifpaf, pose3d, with_azure)
    demo.run()
    print('done')
