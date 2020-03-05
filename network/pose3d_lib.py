import os
import numpy as np
import json
import math

"""define limbs connections and joints indices"""
limbs = [(17,20),(17,1),(1,2),(2,3),(17,4),(4,5),(5,6),(7,8),(8,9),(10,11),(11,12),(7,10),(17,19),(19,18)]
joints = list(range(21))

limb_names = ["neck", "right shoulder", "right biceps", "right forearm", "left shoulder", "left biceps",
              "left forearm", "right quad", "right tibias", "left quad", "left tibias","hip", "upper back", "lower back"]

joint_names = ['nose', 'right shoulder','right elbow', 'right wrist', 'left shoulder', 'left elbow', 'left wrist', 
               'right hip', 'right knee', 'right ankle', 'left hip', 'left knee', 'left ankle','right eye','left eye',
               'right ear','left ear','center shoulder','center hip','center back','head']

limb_dict = dict(zip(limb_names, limbs))
joint_dict = dict(zip(joint_names, joints))

"""define pifpaf body proportions"""

BODY_PROPORTIONS = {'neck': 0.0,
                    'shoulder': 0.0,
                    'biceps': 0.0,
                    'forearm': 0.0,
                    'quad': 0.0,
                    'tibias': 0.0,
                    'hip': 0.0,
                    'back': 0.0
                    }
FACE_POSITIONS = {'nose': [0.0, 0.0, 0.0],
                  'ears': [0.0, 0.0, 0.0],
                  'eyes': [0.0, 0.0, 0.0]
                 }

NB_JOINTS = 21

def normalize_3d(data):
    """normalize 3d data so we have the center hip at the axes origins and ymax-ymin is 1 """

    #we find the shift to center around the hip
    shift = (data[:,joint_dict['right hip']] + data[:,joint_dict['left hip']]).reshape((3,1))/2
    
    #we find the ratio to scale down
    ratio = (np.max(data[1,:]-np.min(data[1,:])))
    
    # we center and scale the joints
    data = (data[:,:]-shift)/ratio

    return data

def normalize_2d(data, shift=None):
    """normalize 2d data so we have the center hip at the axes origins and ymax-ymin is 1 """
    
    #we find the shift to center around the hip
    if shift is None:
        shift = (data[:,joint_dict['right hip']] + data[:,joint_dict['left hip']]).reshape((2,1))/2
    
    #we find the ratio to scale down
    ratio = (np.max(data[1,:]-np.min(data[1,:])))
    
    # we center and scale the joints
    data = (data[:,:]-shift)/ratio

    return data

def find_limb_length(data, limb_name):
    """find the length of a limb from its name"""

    keypoints = data

    return np.linalg.norm(keypoints[:,limb_dict[limb_name][0]]-keypoints[:,limb_dict[limb_name][1]])

def change_3d_joint_pos(data, joint_name, pos_3d):
    """ Change the position of a joint in 3d"""
    keypoints = data
    keypoints[:, joint_dict[joint_name]] = pos_3d

    return keypoints#.transpose().flatten()

def get_3d_joint_pos(data, joint_name):
    """ get the position of a joint in 3d"""
    keypoints = data

    return keypoints[:, joint_dict[joint_name]].reshape((3,1))
 
def move_member(data, member_name, a0, a1, a2, a3):
    """ Forward kinematics function""" 
    joints_data = data.copy()
    
    #### get all the useful sin and cos
    c0 = math.cos(a0)
    c1 = math.cos(a1)
    c2 = math.cos(a2)
    c3 = math.cos(a3)
    c23 = math.cos(a2+a3)

    s0 = math.sin(a0)
    s1 = math.sin(a1)
    s2 = math.sin(a2)
    s3 = math.sin(a3)
    s23 = math.sin(a2+a3)

    #### get what member we are moving and w
    body_side = member_name.split(maxsplit=1)[0]
    member = member_name.split(maxsplit=1)[1]

    #### define the correct limbs joints and rotation matrices
    if member == 'arm':
        member_limbs = ['biceps', 'forearm']
        member_joints = ['shoulder', 'elbow', 'wrist']

        r_2 = np.array([[1,    0,   0],
                        [0,   c0, -s0],
                        [0,   s0,  c0]])

        r_1 = np.array([[ c1,  0,   s1],
                        [  0,  1,    0],
                        [-s1,  0,   c1]])

    else:
        member_limbs = ['quad', 'tibias']
        member_joints = ['hip', 'knee', 'ankle']

        r_2 = np.array([[ c0,  0,   s0],
                        [  0,  1,    0],
                        [-s0,  0,   c0]])

        r_1 = np.array([[ c1, -s1, 0],
                        [ s1,  c1, 0],
                        [ 0,   0, 1]])


    #### get the limbs lengths
    l2 = find_limb_length(joints_data, body_side+' '+member_limbs[0])
    l3 = find_limb_length(joints_data, body_side+' '+member_limbs[1])

    #### get the member origin
    origin_pos = get_3d_joint_pos(joints_data, body_side+' '+member_joints[0])

    ### find the 2d positions of all the joints relative to the origin
    if member == 'arm':            
        factor = 1         
        if body_side == 'right':
            factor = -1

        x_joints = dict(zip(member_joints, [0, factor*(l2*c2), factor*(l2*c2 + l3*c23)]))
        y_joints = dict(zip(member_joints, [0, factor*(l2*s2), factor*(l2*s2 + l3*s23)]))
        z_joints = dict(zip(member_joints, [0, 0, 0]))
    else:
        y_joints = dict(zip(member_joints, [0, l2*c2, l2*c2 + l3*c23]))
        z_joints = dict(zip(member_joints, [0, l2*s2, l2*s2 + l3*s23]))
        x_joints = dict(zip(member_joints, [0, 0, 0]))

    #### find the new position of each joints and move it
    for joint in member_joints[1:]:
        pos = np.array([x_joints[joint], y_joints[joint], z_joints[joint]]).reshape((3,1))
        pos = np.dot(r_2, np.dot(r_1, pos))
        pos = pos + origin_pos

        joints_data = change_3d_joint_pos(joints_data, body_side+' '+joint, pos.reshape((3)))
    
    return joints_data

def rotate_pose(data, axis, angle):
    """Rotate the pose according to an angle and axis"""
        
    joints_data = data.copy()
    
    c = math.cos(angle)
    s = math.sin(angle)
    
    r = {'x': np.array([[1, 0,  0],
                        [0, c, -s],
                        [0, s,  c]]),
            'y': np.array([[ c, 0, s],
                        [ 0, 1, 0],
                        [-s, 0, c]]),
            'z': np.array([[ c, -s, 0],
                        [ s,  c, 0],
                        [ 0,  0, 1]])
        }
    for joint in joint_names:
        new_pos = np.dot(r[axis], get_3d_joint_pos(joints_data, joint))
        joints_data = change_3d_joint_pos(joints_data, joint, new_pos.reshape((3)))
    
    return joints_data
        
def rotate_point(point, rot_axis, angle):
    """Rotate the point according to an axis and an angle"""
    
    ux, uy, uz = rot_axis
    
    c = math.cos(angle)
    s = math.sin(angle)
    
    rot_matrix = np.array([[   c+ux**2*(1-c), ux*uy*(1-c)-uz*s, ux*uz*(1-c)+uy*s],
                           [uy*ux*(1-c)+uz*s,    c+uy**2*(1-c), uy*uz*(1-c)-ux*s],
                           [uz*ux*(1-c)-uy*s, uz*uy*(1-c)+ux*s,    c+uz**2*(1-c)]])
    
    return np.dot(rot_matrix, point)
    
def rotate_backOrHead(data, member_name, a0, a1, a2):
    """Rotate the back or head according to 3 angles"""
        
    joints_data = data.copy()
    
    joints2move = {'head': ['nose','head','right eye','left eye','right ear','left ear'],
                    'back': ['nose','head','right eye','left eye','right ear','left ear',
                            'right shoulder','right elbow', 'right wrist', 'left shoulder',
                            'left elbow', 'left wrist','center shoulder']}
    origins = {'head': 'center shoulder',
                'back': 'center back'}
    
    x = np.array([1,0,0])
    y = np.array([0,1,0])
    z = np.array([0,0,1])
    
    #rotate around x axis with angle a0
    for joint in joints2move[member_name]:
        pos = get_3d_joint_pos(joints_data, joint) - get_3d_joint_pos(joints_data, origins[member_name])
        
        pos = rotate_point(pos, x, a0) + get_3d_joint_pos(joints_data, origins[member_name])
        
        joints_data = change_3d_joint_pos(joints_data, joint, pos.reshape((3)))
    
    #rotate the coordinate system
    y = rotate_point(y, x, a0)
    z = rotate_point(z, x, a0)

    #rotate around y axis with angle a1
    for joint in joints2move[member_name]:
        pos = get_3d_joint_pos(joints_data, joint) - get_3d_joint_pos(joints_data, origins[member_name])
        
        pos = rotate_point(pos, y, a1) + get_3d_joint_pos(joints_data, origins[member_name])
        
        joints_data = change_3d_joint_pos(joints_data, joint, pos.reshape((3)))
    
    #rotate the coordinate system
    z = rotate_point(z, y, a1)

    #rotate around z axis with angle a2
    for joint in joints2move[member_name]:
        pos = get_3d_joint_pos(joints_data, joint) - get_3d_joint_pos(joints_data, origins[member_name])
        
        pos = rotate_point(pos, z, a2) + get_3d_joint_pos(joints_data, origins[member_name])
        
        joints_data = change_3d_joint_pos(joints_data, joint, pos.reshape((3)))
        
    return joints_data
    
def full_pose_rotation(data, angle_dic):
    """Rotates a pose according to angles in a dictionary"""
    
    members = ['right arm', 'left arm', 'right leg', 'left leg']
    head_or_back = ['head','back']
    axis = ['x', 'y', 'z']
    
    final_pose = data
    
    for key in angle_dic.keys():
        a = angle_dic[key]
        if key in members:
            final_pose = move_member(final_pose, key, a[0], a[1], a[2], a[3])
        if key in head_or_back:
            final_pose = rotate_backOrHead(final_pose, key, a[0], a[1], a[2])
        if key in axis:
            final_pose = rotate_pose(final_pose, key, a[0])
    return final_pose

def project_to_pixels(xyz, kk):
    """Project a single point in space into the image"""
    xx, yy, zz = np.dot(kk, xyz)
    uu = int(xx / zz)
    vv = int(yy / zz)

    return uu, vv

def project_pose_2cam(pose, kk, z_shift, joints2keep):
    """project 3d pose to 2d lens"""
    
    pose_moved = pose.copy()
    pose_moved[2] += z_shift
    projected_pose = []
    
    for point in pose_moved.transpose():
        projected_pose.append(list(project_to_pixels(point, kk)))
    
    projected_pose = np.array(projected_pose).transpose().astype(float)
    joints_indices = [joint_dict[x] for x in joints2keep]
    shift = (projected_pose[:,joint_dict['right hip']] + 
            projected_pose[:,joint_dict['left hip']]).reshape((2,1))/2
    projected_pose[:,joints_indices] = normalize_2d(projected_pose[:,joints_indices], shift)
    projected_pose = filter_joints_2d(projected_pose, joints2keep, -10).transpose().flatten()

    return projected_pose

def filter_joints(data, list_joints, value):
    """Filter joints according to a list of them"""
    
    null_pos = np.array([value,value,value])
    
    joints_data = data.copy()
    
    for joint in list(set(joint_names)-set(list_joints)):
        joints_data = change_3d_joint_pos(joints_data, joint, null_pos)
        
    return joints_data

def change_2d_joint_pos(data, joint_name, pos_2d):
    """ Change the position of a joint in 2d"""
    pedestrian = data
    pedestrian[:, joint_dict[joint_name]] = pos_2d
        
    return pedestrian

def filter_joints_2d(data, list_joints, value):
    """Filter joints according to a list of them"""
    null_pos = np.array([value,value])
    
    joints_data = data.copy()
    
    for joint in list(set(joint_names)-set(list_joints)):
        joints_data = change_2d_joint_pos(joints_data, joint, null_pos)
        
    return joints_data