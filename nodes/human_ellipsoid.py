#!/usr/bin/env python3

import os
import numpy as np
import rospy
# from math import cos, sin, pi
# import matplotlib.pyplot as plt
# import geometry_msgs
from std_msgs.msg import Header, Float32, Float64MultiArray, MultiArrayDimension
import sensor_msgs.msg
import visualization_msgs.msg
from visualization_msgs.msg import Marker
from scipy.spatial.transform import Rotation
# import tf
# from tf.transformations import quaternion_from_matrix
from std_srvs.srv import Empty, EmptyResponse
# import torch
import pytorch_kinematics as pk
# import xacro
from urdf_parser_py.urdf import URDF
# import PyKDL

# BEFORE RUNNING THIS:
# go to urdf directory
# rosrun xacro xacro -o human.urdf human.urdf.xacro

USE_MOCAP = True

urdf_path = os.path.join(os.path.abspath(os.path.join(os.path.abspath(__file__), '../..')), "urdf")

# xacro_model = xacro.process_file(os.path.join(urdf_path, "human.urdf.xacro"), in_order=True)
# xml_model = xacro_model.toxml()
# urdf_model = URDF.from_xml_string(xml_model)
# # urdf_model = urdf_model.replace('robot name="human" version="1.0"', 'robot name="human"')
# print(urdf_model)
# with open(os.path.join(urdf_path, "human.urdf"), 'w') as urdf_file:
#         urdf_file.write(urdf_model.to_xml_string())

chain_right = pk.build_serial_chain_from_urdf(open(os.path.join(urdf_path, "human.urdf")).read(), "RightHandCOM","hip")
chain_left  = pk.build_serial_chain_from_urdf(open(os.path.join(urdf_path, "human.urdf")).read(), "LeftHandCOM","hip")
chain_right_joints=  chain_right.get_joint_parameter_names(exclude_fixed=True)
chain_left_joints=   chain_left.get_joint_parameter_names(exclude_fixed=True)
# chain_right.ee = "RightHandCOM"
# chain_left.ee = "LeftHandCOM"

# print("chain_right=  ",chain_right)
# print( "chain_left=  ", chain_left)
# print("chain_right=  ",chain_right.get_joint_parameter_names(exclude_fixed=True))
# print( "chain_left=  ", chain_left.get_joint_parameter_names(exclude_fixed=True))

class Ellipsoid:

    def __init__(self, dt) -> None:
        self.dt = dt
        self.time_start = rospy.Time.now()
        self.time_last_update = rospy.Time.now()
        self.counter = 10
        self.counter_save = 10
        # self.last_iteration_time == None

        self.pelvis_pos = np.zeros((3,))
        self.right_joint_states = np.zeros(len(chain_right_joints))
        self.left_joint_states = np.zeros(len(chain_left_joints))
        self.human_ellipsoid_matrix = np.zeros((6,6))

        if USE_MOCAP:
            self.human_joint_state_sub = rospy.Subscriber('/human/joint_states', sensor_msgs.msg.JointState, self.joint_states_callback, tcp_nodelay=True)
        else:
            self.human_joint_state_sub = rospy.Subscriber('/human/joint_states', sensor_msgs.msg.JointState, self.fixed_joint_states_callback, tcp_nodelay=True)

        self.right_ellipsoid_pub = rospy.Publisher('/human/vis/ellipsoid_r', visualization_msgs.msg.Marker, queue_size=10)
        self.left_ellipsoid_pub =  rospy.Publisher('/human/vis/ellipsoid_l', visualization_msgs.msg.Marker, queue_size=10)
        self.right_force_ellipsoid_pub = rospy.Publisher('/human/vis/force_ellipsoid_r', visualization_msgs.msg.Marker, queue_size=10)
        self.left_force_ellipsoid_pub =  rospy.Publisher('/human/vis/force_ellipsoid_l', visualization_msgs.msg.Marker, queue_size=10)
        self.human_right_ellipsoid_matrix_pub = rospy.Publisher('/human/right_ellip_matrix', Float64MultiArray, queue_size=10)
        self.human_left_ellipsoid_matrix_pub = rospy.Publisher('/human/left_ellip_matrix', Float64MultiArray, queue_size=10)

    def fixed_joint_states_callback(self,data):
        if len(data.position) == len(data.name):
            self.right_joint_states = [data.position[i] for i, joint in enumerate(data.name) if joint in chain_right_joints]
            self.left_joint_states = [data.position[i] for i, joint in enumerate(data.name) if joint in chain_left_joints]
        else:
            rospy.logwarn("Mismatched lengths in joint_states data.")    

    def joint_states_callback(self,data):
        if len(data.position) == len(data.name):
            self.right_joint_states = [data.position[i] for i, joint in enumerate(data.name) if joint in chain_right_joints]
            self.left_joint_states = [data.position[i] for i, joint in enumerate(data.name) if joint in chain_left_joints]
        else:
            rospy.logwarn("Mismatched lengths in joint_states data.")
        
    def get_hand_position(self,chain):
        if chain==chain_right:
            tg = chain.forward_kinematics(self.right_joint_states)
        elif chain==chain_left:
            tg = chain.forward_kinematics(self.left_joint_states)
        else:
            print("chain not specified")
        ee_T = tg.get_matrix()
        ee_T = np.array(ee_T[0, :, :])
        ee_pos = ee_T[:3, 3]
        return ee_pos, ee_T
    
    def calc_manipullability_ellipsoid(self,chain): 
        # print(self.right_joint_states)
        # print(self.left_joint_states)
        if chain==chain_right:
            J = chain.jacobian(self.right_joint_states)
        elif chain==chain_left:
            J = chain.jacobian(self.left_joint_states)
        else:
            print("chain not specified")
        J_pos = J[0,:3,:]
        M = J_pos @ J_pos.T
        return M

    def show_manipulability_ellipsoid(self,chain):
        
        matrix_msg = Float64MultiArray()
        marker = visualization_msgs.msg.Marker()
        marker.header.frame_id = "human/hip" 
        marker.header.stamp = rospy.Time.now()
        marker.type = visualization_msgs.msg.Marker.SPHERE
        marker.action = visualization_msgs.msg.Marker.ADD

        ee_pos, ee_T = self.get_hand_position(chain)
        marker.pose.position.x = ee_pos[0] 
        marker.pose.position.y = ee_pos[1] 
        marker.pose.position.z = ee_pos[2] 

        M = self.calc_manipullability_ellipsoid(chain)

        #### Method 1
        # eigenvalues, eigenvectors = np.linalg.eig(M)
        # order = np.argsort(eigenvalues)[::-1]
        # eigenvalues = eigenvalues[order]
        # eigenvectors = eigenvectors[:, order]
        # axes_len = eigenvalues
        # axes_len = np.sqrt(eigenvalues)
        # R = np.vstack((np.hstack((eigenvectors, np.zeros((3,1)))), \
                        # np.array([0.0, 0.0, 0.0, 1.0])))
        # quat = quaternion_from_matrix(R)
        # # quat = quat / np.linalg.norm(q)

        #### Method 2
        # U, S, Vh = np.linalg.svd(M)
        # axes_len = np.sqrt(S)
        # R = np.vstack((np.hstack(((U * axes_len[..., None, :]) @ Vh, np.zeros((3,1)))), \
        #                 np.array([0.0, 0.0, 0.0, 1.0])))
        # R = ee_T @ R          
        # quat = quaternion_from_matrix(R)

        #### Method 3
        (eigValues,eigVectors) = np.linalg.eig (M)
        axes_len = np.sqrt(eigValues)

        # eigx_n =-PyKDL.Vector(eigVectors[0,0],eigVectors[1,0],eigVectors[2,0])
        # eigy_n =-PyKDL.Vector(eigVectors[0,1],eigVectors[1,1],eigVectors[2,1])
        # eigz_n =-PyKDL.Vector(eigVectors[0,2],eigVectors[1,2],eigVectors[2,2])        
        # R = PyKDL.Rotation (eigx_n,eigy_n,eigz_n)
        # quat = R.GetQuaternion ()
        eigx_n = -eigVectors[:, 0]
        eigy_n = -eigVectors[:, 1]
        eigz_n = -eigVectors[:, 2]
        R_matrix = np.column_stack((eigx_n, eigy_n, eigz_n))
        rotation = Rotation.from_matrix(R_matrix)
        quat = rotation.as_quat()
        
        marker.pose.orientation.x = quat[0]
        marker.pose.orientation.y = quat[1]
        marker.pose.orientation.z = quat[2]
        marker.pose.orientation.w = quat[3]

        scl = 0.5       
        marker.scale.x = axes_len[0] * scl
        marker.scale.y = axes_len[1] * scl
        marker.scale.z = axes_len[2] * scl

        marker.color.a = 0.5
        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 1.0

        # Publish human_ellipsoid_matrix(6,6):  top-left(3,3) for left hand and bottom-right(3,3) for right hand
        if chain==chain_right:
            self.right_ellipsoid_pub.publish(marker)
            matrix_msg.data = np.vstack((np.hstack((M, ee_pos.reshape(-1,1))), np.array([0.0, 0.0, 0.0, 1.0]))).flatten().tolist()
            self.human_right_ellipsoid_matrix_pub.publish(matrix_msg)
        elif chain==chain_left:
            self.left_ellipsoid_pub.publish(marker)
            matrix_msg.data = np.vstack((np.hstack((M, ee_pos.reshape(-1,1))), np.array([0.0, 0.0, 0.0, 1.0]))).flatten().tolist()
            self.human_left_ellipsoid_matrix_pub.publish(matrix_msg)
        else:
            print("chain not specified")
        

    def show_force_ellipsoid(self,chain):
        
        marker = visualization_msgs.msg.Marker()
        marker.header.frame_id = "human/hip" 
        marker.header.stamp = rospy.Time.now()
        marker.type = visualization_msgs.msg.Marker.SPHERE
        marker.action = visualization_msgs.msg.Marker.ADD

        ee_pos, ee_T = self.get_hand_position(chain)
        marker.pose.position.x = ee_pos[0] 
        marker.pose.position.y = ee_pos[1] 
        marker.pose.position.z = ee_pos[2]   

        M = self.calc_manipullability_ellipsoid(chain)
        
        (eigValues,eigVectors) = np.linalg.eig (np.linalg.inv(M))
        axes_len = np.sqrt(eigValues)

        # eigx_n =-PyKDL.Vector(eigVectors[0,0],eigVectors[1,0],eigVectors[2,0])
        # eigy_n =-PyKDL.Vector(eigVectors[0,1],eigVectors[1,1],eigVectors[2,1])
        # eigz_n =-PyKDL.Vector(eigVectors[0,2],eigVectors[1,2],eigVectors[2,2])        
        # R = PyKDL.Rotation (eigx_n,eigy_n,eigz_n)
        # quat = R.GetQuaternion ()
        eigx_n = -eigVectors[:, 0]
        eigy_n = -eigVectors[:, 1]
        eigz_n = -eigVectors[:, 2]
        R_matrix = np.column_stack((eigx_n, eigy_n, eigz_n))
        rotation = Rotation.from_matrix(R_matrix)
        quat = rotation.as_quat()
        
        marker.pose.orientation.x = quat[0]
        marker.pose.orientation.y = quat[1]
        marker.pose.orientation.z = quat[2]
        marker.pose.orientation.w = quat[3]
        
        scl = 0.5        
        marker.scale.x = axes_len[0] * scl
        marker.scale.y = axes_len[1] * scl
        marker.scale.z = axes_len[2] * scl

        marker.color.a = 0.5
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0

        if chain==chain_right:
            self.right_force_ellipsoid_pub.publish(marker)
        elif chain==chain_left:
            self.left_force_ellipsoid_pub.publish(marker)
        else:
            print("chain not specified")        

def main():
    rospy.init_node("ellipsoid_publisher")
    freq=10
    r = rospy.Rate(freq)
    ellipsoid = Ellipsoid(1/freq)

    while not rospy.is_shutdown():
        ellipsoid.show_manipulability_ellipsoid(chain_right)
        ellipsoid.show_manipulability_ellipsoid(chain_left)
        ellipsoid.show_force_ellipsoid(chain_right)
        ellipsoid.show_force_ellipsoid(chain_left)
        r.sleep()
    rospy.spin()    
         

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass





