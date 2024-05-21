#!/usr/bin/env python3

import numpy as np
import pandas as pd
import bagpy
import re
from bagpy import bagreader
from pathlib import Path
import inspect
import glob
import rospy
import os
import matplotlib.pyplot as plt
from scipy.integrate import simps
from scipy.interpolate import interp1d
from tf.transformations import euler_from_quaternion, quaternion_from_euler, quaternion_matrix, euler_from_matrix


def interpolate(data, time, time_match):
    time_match_limited = time_match[(time_match >= time.min()) & (time_match <= time.max())]
    if data.ndim == 1:
        data_interpolated = interp1d(time, data)(time_match_limited)
    elif data.ndim == 2:
        data_interpolated = np.array([interp1d(time, data[i])(time_match_limited) 
                                      for i in range(data.shape[0])])
    elif data.ndim == 3:
        data_interpolated = np.array([[[interp1d(time, data[:, i, j])(time_match_limited) 
                                        for j in range(data.shape[2])] for i in range(data.shape[1])]]).T
        data_interpolated = np.moveaxis(data_interpolated, 1, 2).squeeze()
    return time_match_limited, data_interpolated

def parse_position_orientation(row):
    position_matches = re.findall(r'position: \n  x: (.*?)\n  y: (.*?)\n  z: (.*?)\n', row['poses'])
    orientation_matches = re.findall(r'orientation: \n  x: (.*?)\n  y: (.*?)\n  z: (.*?)\n  w: (.*?)[,\]]', row['poses'])
    if position_matches and orientation_matches:
        positions = [tuple(float(i) for i in match) for match in position_matches]
        orientations = [tuple(float(i) for i in match) for match in orientation_matches]
        return np.array(positions), np.array(orientations)
    else:
        return None, None

def get_skl_data(data):
    skl_positions, skl_orientations = np.zeros((len(data), 51, 3)), np.zeros((len(data), 51, 4))
    skl_time = np.array(data['Time']) 
    for t in range(len(skl_time)):
        skl_positions[t], skl_orientations[t] = parse_position_orientation(data.iloc[t])
    return skl_time, skl_positions, skl_orientations

def get_us_pose(data):
    us_mocap_time = np.array(data['Time']) 
    us_mocap_pos = np.array([data['pose.position.x'], data['pose.position.y'], data['pose.position.z']])
    us_mocap_rot = np.array([data['pose.orientation.x'], data['pose.orientation.y'], data['pose.orientation.z'], data['pose.orientation.w']])
    return us_mocap_time, us_mocap_pos, us_mocap_rot

def get_ultrasound_image_data(data):
    ultrasound_time = np.array(data['Time']) 
    ultrasound_image = np.array(data['data'])
    return ultrasound_time, ultrasound_image


##########################################################################################################
##########################################################################################################
##########################################################################################################
##########################################################################################################
# Read bag file
test = 'nobag_1'
# bag_name = str(Path(inspect.getsourcefile(lambda:0)).parent.parent/'0426'/(test+'_repub.bag'))
bag_name = str(Path(inspect.getsourcefile(lambda:0)).parent.parent/'0426'/(test+'.bag'))
b = bagreader(bag_name)
print(b.topic_table)
# bag_names = glob.glob(str(Path(inspect.getsourcefile(lambda:0)).parent.parent/'data'/'repub_bags'/"*.bag"))
# for bag_name in bag_names:
#     b = bagreader(bag_name)
#     print(b.topic_table)

skl_data = pd.read_csv(b.message_by_topic('/natnet_ros/fullbody/pose'))
print('/natnet_ros/fullbody/pose  ', skl_data.columns)
skl_time, skl_positions, skl_orientations = get_skl_data(skl_data)

# us_mocap_data = pd.read_csv(b.message_by_topic('/natnet_ros/ultrasound/pose'))
# us_mocap_time, us_mocap_pos, us_mocap_rot = get_us_pose(us_mocap_data)

ultrasound_data = pd.read_csv(b.message_by_topic('/ultrasound'))
print('/ultrasound  ', ultrasound_data.columns)
ultrasound_time, ultrasound_image = get_ultrasound_image_data(ultrasound_data)
# print("ultrasound_time: ", ultrasound_time.shape, "ultrasound_image: ", ultrasound_image.shape)

# Interpolate the mocap data to the ultrasound image time 
_, skl_positions = interpolate(skl_positions, skl_time, ultrasound_time)
skl_time, skl_orientations = interpolate(skl_orientations, skl_time, ultrasound_time)
time = skl_time - skl_time[0]


# Extract skl angles
'''
Mocap poses (51 bones)
Upper body chains:
- Hip to Head: 0 (hip), 1 (abdomen), 2 (chest), 22 (neck), 23 (head)
- Right Arm: 0 (hip), 1 (abdomen), 2 (chest), 3 (r_shoulder), 4 (r_u_arm), 5 (r_f_arm), 6 (r_hand)
- Left Arm: 0 (hip), 1 (abdomen), 2 (chest), 24 (l_shoulder), 25 (l_u_arm), 26 (l_f_arm), 27 (l_hand)
'''
skl_bones_num = 51
bone_rot = np.zeros((skl_bones_num,3))
right_arm = np.zeros((len(time), 7, 3))
left_arm = np.zeros((len(time), 7, 3))

for t in range(len(time)):
    # # Hip to Head
    # R_prox = np.identity(3)
    # for idx in [0, 1, 2, 22, 23]:
    #     R_global = quaternion_matrix(skl_orientations[t,idx])[:3,:3]
    #     R_rel = R_prox.T @ R_global
    #     R_prox = R_global
    #     (ang1, ang2, ang3) = euler_from_matrix(R_rel, axes='rxyz')
    #     bone_rot[idx,0] = -ang2 # ang3 # ###????
    #     bone_rot[idx,1] = ang1
    #     bone_rot[idx,2] = ang3

    # Hip to Right Arm
    right_side = []
    R_prox = np.identity(3)
    for idx in [0, 1, 2, 3, 4, 5, 6]:
        # print("bone_quat idx: ", idx, skl_orientations[t,idx])
        R_global = quaternion_matrix(skl_orientations[t,idx])[:3,:3]
        R_rel = R_prox.T @ R_global
        R_prox = R_global
        (ang1, ang2, ang3) = euler_from_matrix(R_rel, axes='rzxy')
        bone_rot[idx,0] = ang1 #-ang2
        bone_rot[idx,1] = ang2
        bone_rot[idx,2] = ang3
        right_side.append(bone_rot[idx,:])
    right_arm[t] = np.array(right_side)
    
    # Hip to Left Arm
    left_side = []
    R_prox = np.identity(3)
    for idx in [0, 1, 2, 24, 25, 26, 27]:
        R_global = quaternion_matrix(skl_orientations[t,idx])[:3,:3]
        R_rel = R_prox.T @ R_global
        R_prox = R_global
        (ang1, ang2, ang3) = euler_from_matrix(R_rel, axes='rzxy')
        bone_rot[idx,0] = ang1 #-ang2
        bone_rot[idx,1] = ang2
        bone_rot[idx,2] = ang3
        left_side.append(bone_rot[idx,:])
    left_arm[t] = np.array(left_side)
    # print("sh: ", [round(180/np.pi*element, 2) for element in bone_rot[24,:]] , \
    #       "el: ", [round(180/np.pi*element, 2) for element in bone_rot[25,:]] , \
    #       "wr: ", [round(180/np.pi*element, 2) for element in bone_rot[26,:]] )
    
left_elbow = left_arm[:,5,0]
right_elbow = right_arm[:,5,0]
print("left_elbow: ", left_elbow.shape, "right_elbow: ", right_elbow.shape)

# Plot the angles
fig = plt.figure(num='left_angles', figsize=(9, 10), tight_layout=True)
ax = fig.add_subplot(111)
ax.set_xlim([0, time[-1]])
# ax.set_ylim([-180, 180])
ax.set_title(test + ': Left Arm Angles')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Angle (deg)')
for i in range(3):
    ax.plot(time, 180/np.pi*left_arm[:,4,i], label=f'Left Shoulder {i}')
for i in range(3):
    ax.plot(time, 180/np.pi*left_arm[:,5,i], label=f'Left Elbow {i}')
for i in range(3):
    ax.plot(time, 180/np.pi*left_arm[:,6,i], label=f'Left Wrist {i}')
ax.legend(loc='upper right')

fig = plt.figure(num='right_angles', figsize=(9, 10), tight_layout=True)
ax = fig.add_subplot(111)
ax.set_xlim([0, time[-1]])
ax.set_title(test + ': Right Arm Angles')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Angle (deg)')
for i in range(3):
    ax.plot(time, 180/np.pi*right_arm[:,4,i], label=f'Right Shoulder {i}')
for i in range(3):
    ax.plot(time, 180/np.pi*right_arm[:,5,i], label=f'Right Elbow {i}')
for i in range(3):
    ax.plot(time, 180/np.pi*right_arm[:,6,i], label=f'Right Wrist {i}')
ax.legend(loc='upper right')

plt.show()

