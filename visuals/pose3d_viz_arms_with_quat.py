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
from ..network import PifPaf, MonoLoco, Pose3d
from ..network.process import preprocess_pifpaf_1, preprocess_pifpaf_2, image_transform
from ..visuals.client import *
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from pyquaternion import Quaternion
from direct.gui.OnscreenText import OnscreenText


body_props = {'neck': 0.07729765064733189,
            'shoulder': 0.13887964509275452,
            'biceps': 0.1903499496025787,
            'forearm': 0.1856855142453098,
            'quad': 0.2250013372092681,
            'tibias': 0.2199742522543633,
            'hip': 0.16409048728199763,
            'back': 0.22711310836812693}

face_pos = {'nose': [0.0, 0.0, 0.07331183671724711],
            'eye': [0.02339558783308983, 0.02517361111111116, 0.058635183635287635],
            'ear': [0.05771397604574112, 0.0, 0.0]}

LOGI_KK = [[0.7477614068838029, -0.0008111695246478873,  0.5134789751408451],
           [0.0,                      1.33178042553125, 0.45684016428124996],
           [0.                ,                     0.,                  1.]]

class Pose3dViz(ShowBase):
    def __init__(self, body_props, face_pos, cam, args, pifpaf, monoloco, pose3d, with_azure = False):
        # Initialize the ShowBase class from which we inherit, which will
        # create a window and set up everything we need for rendering into it.
        ShowBase.__init__(self)

        #self.disableMouse()

        self.setBackgroundColor(0.5, 0.6, 0.9)

        # Add lighting so that the objects are not drawn flat
        self.setupLights()

        self.body_props = body_props
        self.face_pos = face_pos
        self.cam = cam
        self.pifpaf = pifpaf
        self.monoloco = monoloco
        self.pose3d = pose3d
        self.args = args
        self.buffer_counter = 0
        self.buffer_size = 4
        self.buffer = np.zeros((17,self.buffer_size))
        self.with_azure = with_azure
        

        #straight_pose = [-0.09089972, -0.00305391, 0.00309062, -0.6947644, 0.01137601, 0.0478842, 0.7534785, -0.00854983, -0.0391763]

        #straight_pose = [ straight_pose[0], straight_pose[1], straight_pose[2], straight_pose[3], straight_pose[4],\
        #     straight_pose[5], -straight_pose[8], straight_pose[7]+math.pi/2, -straight_pose[6]]

        #self.straight_pose_dic = {'head':[straight_pose[0], straight_pose[1], straight_pose[2]],\
        #                          'back':[straight_pose[3], straight_pose[4], straight_pose[5]],\
        #                          'x':[straight_pose[6]],'y':[straight_pose[7]], 'z':[straight_pose[8]]}

        a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16 = ( 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, math.pi/2, math.pi/2, math.pi/2, 0, 0, 0, 0, 0, 0, 0, 0)
        
        self.angle_dic = {'head':[a0,a1,a2], 'back':[a3,a4,a5], 'x':[a6],'y':[a7], 'z':[a8], 'left arm':[a9, a10, a11, a12], 'right arm':[a13, a14, a15, a16]}

        self.angle_init = np.array([[a0,a1,a2,a3,a4,a5,a6,a7,a8]])

        self.create_pose(self.body_props, self.face_pos, self.angle_dic, self.with_azure)
        
        gltf.patch_loader(self.loader)

        self.create_avatar()
        
        self.points = [Create3DPoint() for _ in range(21)]

        self.azure_points = [Create3DPoint() for _ in range(32)]

        self.taskMgr.add(self.update, "change_angles")
        
    # This function sets up the lighting
    def setupLights(self):
        ambientLight = AmbientLight("ambientLight")
        ambientLight.setColor((.3, .3, .3, 1))
        directionalLight = DirectionalLight("directionalLight")
        directionalLight.setDirection(LVector3(0, 0, -2.5))
        directionalLight.setColor((0.9, 0.8, 0.9, 1))
        render.setLight(render.attachNewNode(ambientLight))
        render.setLight(render.attachNewNode(directionalLight))
   
    def create_pose(self, body_props, face_pos, angle_dic, with_azure = True):
        
        segments_dic = {}
        if with_azure:
            segments_azure_dic = {}
        scale_factor = 100

        len_eye = np.linalg.norm(face_pos['eye'])
        eye_z_rot = (180/np.pi)*math.acos(face_pos['eye'][1]/len_eye)
        eye_y_rot = (180/np.pi)*math.acos(face_pos['eye'][0]/len_eye)

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
        for name, length, parent, z_displacement, rots in seg_data:
            segments_dic[name] = Segment(length=scale_factor*length).draw()
            if with_azure:
                segments_azure_dic[name] = Segment(length=scale_factor*length).draw()
            
            if type(parent)==str:
                parent_obj = segments_dic[parent]
                if with_azure:
                    parent_azure_obj = segments_azure_dic[parent]
            else:
                parent_obj = parent
                if with_azure:
                    parent_azure_obj = parent
            
            segments_dic[name].reparentTo(parent_obj)
            segments_dic[name].setPosHprScale(0, 0, scale_factor*z_displacement, rots[0], rots[1], rots[2], 1., 1., 1.)

            if with_azure:
                if name in []:#['right_hip', 'left_hip', 'lower_back']:
                    segments_azure_dic[name].setPosHprScale(0, 0, scale_factor*z_displacement, rots[0], rots[1], rots[2], 1., 1., 1.)
                else:
                    segments_azure_dic[name].reparentTo(render)
        

        # for segment in segments_dic.values():
        #     segment.setRenderMode(RenderModeAttrib.M_wireframe, 1)

        self.segments_dic = segments_dic
        if with_azure:
            self.segments_azure_dic = segments_azure_dic
        else:
            self.segments_azure_dic = None

    def create_avatar(self):
        avatar_data = [
            ('head',            'upper_neck',       [270,0+180,180,15,15,15]),
            ('neck',            'lower_neck',       [270,30+180,180,15,15,15]),
            ('torso',           'upper_back',       [270,0+180,180,25,15,25]),
            ('rightarm',        'right_biceps',     [270,0,90,20,15,15]),
            ('rightforearm',    'right_forearm',    [270,180,90,20,15,15]),
            ('leftarm',         'left_biceps',      [270,180,90,20,15,15]),
            ('leftforearm',     'left_forearm',     [270,0,90,20,15,15]),
            ('spine',           'lower_back',       [270,0+180,180,15,15,45]),
            ('pelvis',          'lower_back',       [270,0+180,180,25,15,15])
        ]

        #self.segments_dic['left_hip'].hide()

        avatar_dic = {}
        for name, parent, rs in avatar_data:
            avatar_dic[name] = self.loader.loadModel('visuals/models3d/man3D.'+name+'.glb')
            avatar_dic[name].reparentTo(self.segments_dic[parent])
            avatar_dic[name].setHprScale(rs[0],rs[1],rs[2],rs[3],rs[4],rs[5])
        self.avatar_dic = avatar_dic

        if self.segments_azure_dic:
            
            self.segments_azure_dic['left_hip'].setColor(0.0, 0.9, 0.9, 0.0)
            self.segments_dic['left_hip'].setPos(-50, 0, 0)
            camera.setPosHpr(25, -20, 200, 0, -90, 180)

            avatar_azure_dic = {}
            # for name, parent, rs in avatar_data:
            #     avatar_azure_dic[name] = self.loader.loadModel('visuals/models3d/man3D.'+name+'.glb')
            #     avatar_azure_dic[name].reparentTo(self.segments_azure_dic[parent])
            #     avatar_azure_dic[name].setHprScale(rs[0],rs[1],rs[2],rs[3],rs[4],rs[5])
            self.avatar_azure_dic = avatar_azure_dic

        
       
    def process_azure_data(self, azure_data):

        azure_angles_dic = {}
        azure_pos_dic = {}

        #find for hip
        hip_angles = R.from_quat(azure_data[0,3:7]).as_euler('xyz', degrees=True)
        hip_pos = azure_data[0,:3]

        azure_angles_dic['hip'] = np.array([-hip_angles[0], -hip_angles[1], hip_angles[2]])
        azure_pos_dic['hip'] = hip_pos

        #find the back
        key_rots = R.from_quat(azure_data[1:3,3:7])
        key_times = [0, 1]
        back_slerp = Slerp(key_times, key_rots)
        back_angles = back_slerp([0.5])[0].as_euler('xyz', degrees=True)
        azure_angles_dic['back'] = back_angles

        azure_pos_dic['back'] = np.mean(azure_data[1:3,:3], axis = 0)

        #find the head
        head_angles = azure_data[26,3:7]
        head_angles = R.from_quat(head_angles).as_euler('xyz', degrees=True)
        azure_angles_dic['head'] = head_angles

        azure_pos_dic['head'] = azure_data[26,:3]

        #find the left shoulder

        left_shoulder_angles = azure_data[5,3:7]
        left_shoulder_angles = R.from_quat(left_shoulder_angles).as_euler('xyz', degrees=True)
        azure_angles_dic['left shoulder'] = left_shoulder_angles

        azure_pos_dic['left shoulder'] = azure_data[5,:3]

        #find the left elbow
        left_elbow_angles = azure_data[6,3:7]
        left_elbow_angles = R.from_quat(left_elbow_angles).as_euler('xyz', degrees=True)
        azure_angles_dic['left elbow'] = left_elbow_angles

        azure_pos_dic['left elbow'] = azure_data[6,:3]

        #find the right shoulder angles

        right_shoulder_angles = azure_data[12,3:7]
        right_shoulder_angles = R.from_quat(right_shoulder_angles).as_euler('xyz', degrees=True)
        azure_angles_dic['right shoulder'] = right_shoulder_angles

        azure_pos_dic['right shoulder'] = azure_data[12,:3]

        #find the right elbow angles
        right_elbow_angles = azure_data[13,3:7]
        right_elbow_angles = R.from_quat(right_elbow_angles).as_euler('xyz', degrees=True)
        azure_angles_dic['right elbow'] = right_elbow_angles

        azure_pos_dic['right elbow'] = azure_data[13,:3]

        return azure_angles_dic, azure_pos_dic

    
    def update(self, task):
        scale_factor = 100
        if self.with_azure:
            azure_data = np.array(get_data()).reshape((32,8))
            azure_angles_dic, azure_pos_dic = self.process_azure_data(azure_data)

            ratio, shift = 20, -0

            print(azure_pos_dic['hip'][0], azure_pos_dic['hip'][1], azure_pos_dic['hip'][2])

            # self.segments_azure_dic['upper_neck'].setPosHprScale(azure_pos_dic['head'][0]/ratio + shift,
            #                                                      azure_pos_dic['head'][1]/ratio + shift,
            #                                                      azure_pos_dic['head'][2]/ratio + shift,
            #                                                      azure_angles_dic['head'][0],
            #                                                      azure_angles_dic['head'][1], 
            #                                                      azure_angles_dic['head'][2], 
            #                                                      1., 1., 1.)
            self.segments_azure_dic['left_hip'].setPosHprScale(  azure_pos_dic['hip'][0]/ratio + shift,
                                                                 azure_pos_dic['hip'][1]/ratio + shift,
                                                                 azure_pos_dic['hip'][2]/ratio + shift,
                                                                 azure_angles_dic['hip'][0],
                                                                 azure_angles_dic['hip'][1], 
                                                                 azure_angles_dic['hip'][2], 
                                                                 1., 1., 1.)
            self.segments_azure_dic['right_hip'].setPosHprScale(  azure_pos_dic['hip'][0]/ratio + shift,
                                                                 azure_pos_dic['hip'][1]/ratio + shift,
                                                                 azure_pos_dic['hip'][2]/ratio + shift,
                                                                 azure_angles_dic['hip'][0]+90,
                                                                 azure_angles_dic['hip'][1], 
                                                                 azure_angles_dic['hip'][2], 
                                                                 1., 1., 1.)
            self.segments_azure_dic['lower_back'].setPosHprScale(  azure_pos_dic['hip'][0]/ratio + shift,
                                                                 azure_pos_dic['hip'][1]/ratio + shift,
                                                                 azure_pos_dic['hip'][2]/ratio + shift,
                                                                 azure_angles_dic['hip'][0],
                                                                 azure_angles_dic['hip'][1], 
                                                                 azure_angles_dic['hip'][2], 
                                                                 1., 1., 1.)
            self.segments_azure_dic['upper_back'].setPosHprScale(  azure_pos_dic['back'][0]/ratio + shift,
                                                                 azure_pos_dic['back'][1]/ratio + shift,
                                                                 azure_pos_dic['back'][2]/ratio + shift,
                                                                 azure_angles_dic['back'][0]+180,
                                                                 azure_angles_dic['back'][1], 
                                                                 azure_angles_dic['back'][2], 
                                                                 1., 1., 1.)

            # self.segments_azure_dic['upper_back'].setPosHprScale(0, 0, scale_factor*(self.body_props['back']), azure_angles_dic['back'][0],\
            # azure_angles_dic['back'][1], azure_angles_dic['back'][2], 1., 1., 1.)

            # self.segments_azure_dic['left_hip'].setHpr( azure_angles_dic['x'], azure_angles_dic['y'], azure_angles_dic['z'])

            # self.segments_azure_dic['left_biceps'].setPosHprScale(0, 0, scale_factor*(self.body_props['shoulder']),\
            #  azure_angles_dic['left arm'][0], azure_angles_dic['left arm'][1], azure_angles_dic['left arm'][2], 1., 1., 1.)

            # self.segments_azure_dic['left_forearm'].setPosHprScale(0, 0, scale_factor*(self.body_props['biceps']), 0, 0,\
            #      azure_angles_dic['left arm'][3], 1., 1., 1.)
            
            # self.segments_azure_dic['right_biceps'].setPosHprScale(0, 0, scale_factor*(self.body_props['shoulder']),\
            #  azure_angles_dic['right arm'][0], azure_angles_dic['right arm'][1], azure_angles_dic['right arm'][2], 1., 1., 1.)

            # self.segments_azure_dic['right_forearm'].setPosHprScale(0, 0, scale_factor*(self.body_props['biceps']), 0, 0,\
            #      azure_angles_dic['right arm'][3], 1., 1., 1.)
        variation = math.sin(0.1*task.time)

        args = self.args
        pifpaf = self.pifpaf
        monoloco = self.monoloco
        pose3d = self.pose3d

        ret, frame = self.cam.read()

        image = cv2.resize(frame, None, fx=args.scale, fy=args.scale)
        height, width, _ = image.shape

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        processed_image_cpu = image_transform(image.copy())
        processed_image = processed_image_cpu.contiguous().to(args.device, non_blocking=True)

        fields = pifpaf.fields(torch.unsqueeze(processed_image, 0))[0]
        _, _, pifpaf_out = pifpaf.forward(image, processed_image_cpu, fields)

        #print('w,h', width, height)

    
        kk = [[x*width  for x in LOGI_KK[0]],
              [x*height for x in LOGI_KK[1]],
               LOGI_KK[2]]
        
        

        if pifpaf_out:
            if self.with_azure==False:
                camera.setPosHpr(0, -20, 200, 0, -90, 180)

            mono_boxes, mono_keypoints = preprocess_pifpaf_1(pifpaf_out, (width, height))

            pose3d_inputs = preprocess_pifpaf_2(pifpaf_out, body_props, face_pos, kk, True)


            mono_outputs, mono_varss = monoloco.forward(mono_keypoints, kk)

            pose3d_outputs = pose3d.forward(pose3d_inputs).cpu().detach().numpy()

            a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17 = pose3d_outputs[0]

            #print(pose3d_outputs)

            a0, a1, a2, a3, a4, a5, a6, a7, a8 = ( pose3d_outputs[0][0], -pose3d_outputs[0][1], -pose3d_outputs[0][2], pose3d_outputs[0][3], -pose3d_outputs[0][4],\
             -pose3d_outputs[0][5], -pose3d_outputs[0][8], pose3d_outputs[0][7]+math.pi/2, -pose3d_outputs[0][6])

            a9, a10, a11, a12, a13, a14, a15, a16 = (pose3d_outputs[0][9], pose3d_outputs[0][10], pose3d_outputs[0][11], pose3d_outputs[0][12], pose3d_outputs[0][13],\
                pose3d_outputs[0][14], pose3d_outputs[0][15], pose3d_outputs[0][16])


            self.buffer[:,self.buffer_counter%self.buffer_size] = (a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16)
            self.buffer_counter+=1
            
            a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16 = np.mean(self.buffer, axis = 1)

            #a0, a1, a2, a3, a4, a5, a6, a7, a8 = (0,0,0,0,0,0,0,0,0)

            self.angle_dic = {'head':[a0,a1,a2], 'back':[a3,a4,a5], 'x':[a6],'y':[a7], 'z':[a8], 'left arm':[a9, a10, a11, a12], 'right arm':[a13, a14, a15, a16]}
            

            #print('best',self.straight_pose_dic['head'][0])
            # print('current',self.angle_dic['head'][0])

            # diff = (self.straight_pose_dic['head'][0]-self.angle_dic['head'][0])*180/np.pi

            # if diff>20:
            #     self.segments_dic['left_hip'].setColor(0.5, 0.6, 0.9, 0.0)
            #     self.segments_dic['left_hip']
            # else:
            #     self.segments_dic['left_hip'].setColor(0.5, 0.6, 0.9, 0.0)
            #     self.segments_dic['left_hip']

            out_points = np.array(pose3d.post_process( pose3d_outputs, self.body_props, self.face_pos)['pose'])[0].reshape((21, 3))

            azure_points = azure_data[:,:3]

            joints = list(range(21))

            joints2keep = ['left hip', 'right hip', 'nose', 'right shoulder','left shoulder', 'right eye','left eye',
               'right ear','left ear','center shoulder','center hip','center back','head', 'right elbow',
              'right wrist', 'left elbow', 'left wrist']


            joint_names = ['nose', 'right shoulder','right elbow', 'right wrist', 'left shoulder', 'left elbow', 'left wrist', 
               'right hip', 'right knee', 'right ankle', 'left hip', 'left knee', 'left ankle','right eye','left eye',
               'right ear','left ear','center shoulder','center hip','center back','head']

            joint_dict = dict(zip(joint_names, joints))
            
            joints2keep = [joint_dict[x] for x in joints2keep]

            # for i in range(21):
            #     if i in joints2keep:
            #         self.points[i].setPos(scale_factor*out_points[i][0], scale_factor*out_points[i][1], scale_factor*out_points[i][2])

            for i, point in enumerate(azure_points):
                self.azure_points[i].setPos(point[0]/20, point[1]/20, point[2]/20)


            self.segments_dic['upper_neck'].setPosHprScale(0, 0, scale_factor*(self.body_props['neck']), (180/np.pi)*self.angle_dic['head'][1],\
            -(180/np.pi)*self.angle_dic['head'][2], -(180/np.pi)*self.angle_dic['head'][0], 1., 1., 1.)

            self.segments_dic['upper_back'].setPosHprScale(0, 0, scale_factor*(self.body_props['back']), (180/np.pi)*self.angle_dic['back'][1],\
            -(180/np.pi)*self.angle_dic['back'][2], -(180/np.pi)*self.angle_dic['back'][0], 1., 1., 1.)

            # self.segments_dic['upper_back'].setPosHprScale(0, 0, scale_factor*(self.body_props['back']), (180/np.pi)*self.angle_dic['back'][1],\
            # -(180/np.pi)*self.angle_dic['back'][2], -(180/np.pi)*self.angle_dic['back'][0], 1., 1., 1.)

            self.segments_dic['left_hip'].setPosHprScale(0, 0, 0, (180/np.pi)*self.angle_dic['x'][0], -(180/np.pi)*self.angle_dic['z'][0]+20,\
                 -(180/np.pi)*self.angle_dic['y'][0], 1., 1., 1.)
            
            self.segments_dic['left_biceps'].setPosHprScale(0, 0, scale_factor*(self.body_props['shoulder']),\
             ((180/np.pi)*self.angle_dic['left arm'][0])-90, (180/np.pi)*self.angle_dic['left arm'][1], (180/np.pi)*self.angle_dic['left arm'][2], 1., 1., 1.)
            
            self.segments_dic['left_forearm'].setPosHprScale(0, 0, scale_factor*(self.body_props['biceps']), 0, 0, (180/np.pi)*self.angle_dic['left arm'][3], 1., 1., 1.)

            self.segments_dic['right_biceps'].setPosHprScale(0, 0, scale_factor*(self.body_props['shoulder']),\
             ((180/np.pi)*self.angle_dic['right arm'][0])-90, (180/np.pi)*self.angle_dic['right arm'][1], (180/np.pi)*self.angle_dic['right arm'][2], 1., 1., 1.)
            
            self.segments_dic['right_forearm'].setPosHprScale(0, 0, scale_factor*(self.body_props['biceps']), 0, 0, (180/np.pi)*self.angle_dic['right arm'][3], 1., 1., 1.)
            
            

        return Task.cont
        


def webcam(args):
    # add args.device
    args.device = torch.device('cpu')
    if torch.cuda.is_available():
        args.device = torch.device('cuda')
    
    # load models
    args.camera = True
    pifpaf = PifPaf(args)
    monoloco = MonoLoco(model=args.mono_model, device=args.device)
    pose3d = Pose3d(model=args.pose3d_model, device=args.device)

    # # # Start recording
    cam = cv2.VideoCapture(0)

    with_azure = True

    try:
        get_data()
    except:
        print("Azure not connected")
        with_azure = False

    demo = Pose3dViz(body_props, face_pos, cam, args, pifpaf, monoloco, pose3d, with_azure)
    demo.run()
    print('done')
