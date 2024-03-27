#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import copy
from pfilter import ParticleFilter, t_noise
from plot_util import generate_surface, plot_sample
from functools import partial
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import PointCloud
from geometry_msgs.msg import Point32
import math
import time
import rospy
import tf
from PFDS import PFDS
import rospkg

from std_msgs.msg import Header, Float32, Float64MultiArray
from geometry_msgs.msg import WrenchStamped, Pose, Twist, TwistStamped
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion, quaternion_from_euler, quaternion_matrix
from std_srvs.srv import Empty, EmptyResponse

from visualization_msgs.msg import Marker

from dynamic_reconfigure.server import Server
import dynamic_reconfigure.client
from intent_capability_hri.cfg import IntentCapabilityHRIConfig
import config.controller_config as cc

from game_theory_controller import NegotiationController
from objects.Desk import Desk2D, Pot3D ###SHOULD NOT USE !!! USE MOCAP!!!
from pfds_msgs.msg import MatrixVec
from helper_fun import pred_goal_from_hist, A_only_est

DESIRED_ROLL = np.pi
DESIRED_PITCH = np.pi/2
USE_MOCAP = False
DESIRED_HEIGHT = 0.5
A_Z = -3.0

def make_text(text, attractor):
    marker = Marker()
    marker.header.frame_id = "world"
    marker.header.stamp = rospy.Time.now()
    marker.type = marker.TEXT_VIEW_FACING
    marker.action = marker.ADD
    marker.pose.position.x = attractor[0]
    marker.pose.position.y = attractor[1]
    marker.pose.position.z = attractor[2] + 0.1 # yaw
    marker.scale.z = 0.05
    marker.text = text
    marker.color.a = 1.0  # Alpha
    # white text  
    marker.color.r = 1.0  # Red
    marker.color.g = 1.0  # Green
    marker.color.b = 1.0  # Blue
    return marker
def make_sphere(x,y, z, radius, color = [1.0, 0.0, 0.0]):
    marker = Marker()
    marker.header.frame_id = "world"
    marker.header.stamp = rospy.Time.now()

    marker.type = marker.SPHERE
    marker.action = marker.ADD

    marker.pose.position.x = x
    marker.pose.position.y = y
    marker.pose.position.z = z

    marker.pose.orientation.x = 0.0
    marker.pose.orientation.y = 0.0
    marker.pose.orientation.z = 0.0
    marker.pose.orientation.w = 1.0

    marker.scale.x = radius  # Radius
    marker.scale.y = radius  # Radius
    marker.scale.z = radius  # Radius

    marker.color.a = 1.0  # Alpha
    marker.color.r = color[0]  # Red
    marker.color.g = color[1]  # Green
    marker.color.b = color[2]  # Blue
    return marker

class Move:
    def __init__(self, dt) -> None:
        self.dt = dt
        self.desk = Desk2D(pos=np.array([0.5, 0.0, 0.0]), vel=np.array([0.0, 0.0, 0.0]), dt=dt)
        # self.desk = Pot3D(pos=np.array([0.5, 0.0, 0.0, 0.0, 0.2, 0.3]), vel=np.array([0.0, 0.0, 0.0]), dt=dt)
        # self.desk.plot()
        # exit()
        self.pfds = PFDS(n_particles=400, dt=dt, object_init_pos=[0.7, 0.0, 0.0], \
                    object_init_vel=[0.0, 0.0, 0.0], \
                        save_in_folder=False, real_time_vis=False, verbose=False, sim_bool=False, 
                        object=self.desk)
        self.time_start = rospy.Time.now()
        self.receive_force_time = rospy.Time.now()
        self.counter = 10
        self.counter_save = 10

        self.vel_angvel_err = np.zeros(6)

        self.force_sub = rospy.Subscriber("/robotiq_force_torque_wrench_filtered", WrenchStamped, self.force_callback)

        if USE_MOCAP:
            self.object_sub = rospy.Subscriber("/object_pos", Pose, self.object_callback)
        self.robot_state_sub = rospy.Subscriber("/iiwa/task_states", Pose, self.robot_callback, tcp_nodelay=True)
        self.robot_vel_sub = rospy.Subscriber("/iiwa/ee_vel_cartimp", Pose, self.robot_vel_callback, tcp_nodelay=True)
        self.robot_angVel_sub = rospy.Subscriber("/iiwa/ee_angvel_cartimp", Pose, self.robot_ang_vel_callback, tcp_nodelay=True)
        self.vel_error_sub = rospy.Subscriber("/iiwa/ee_vel_angvel_error", Twist, self.vel_error_callback, tcp_nodelay=True)
        self.human_right_ellipsoid_matrix_sub = rospy.Subscriber('/human/right_ellip_matrix', Float64MultiArray, self.right_ellipsoid_callback, tcp_nodelay=True)
        self.human_left_ellipsoid_matrix_sub = rospy.Subscriber('/human/left_ellip_matrix', Float64MultiArray, self.left_ellipsoid_callback, tcp_nodelay=True)
        # self.force_filter_sub = rospy.Subscriber("/admittance/human_wrench", Wrench, force_filter_callback)
        # self.tank_state_sub = rospy.Subscriber("/admittance/tank_state_percentage", Float32, self.tank_state_callback, tcp_nodelay=True)
        self.tank_state = 0.0
        self.past_error_norm = np.zeros(100)


        self.force_frame_rotation = np.zeros((6,6))
        self.force_arr = np.zeros(6)
        self.object_pos = np.zeros(7)
        self.ee_pos = np.array([0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        self.ee_vel = np.zeros(6)
        self.right_M = np.eye(3)
        self.left_M = np.eye(3)

        self.force_torque_compensate_pub = rospy.Publisher("/robotiq_force_torque_compensate", WrenchStamped, queue_size=1)
        self.assist_pub = rospy.Publisher("/iiwa/desired_twist", TwistStamped, queue_size=10)
        self.odom_pub   = rospy.Publisher("/iiwa/desired_odom" , Odometry, queue_size=10)
        self.ds_pub     = rospy.Publisher("/iiwa/desired_ds"   , MatrixVec, queue_size=10)
        #following DS convention, pose is the attractor, velocity is the current desired velocity

        #Using pointcloud to represent particle location, for visualization
        self.particles_pub = rospy.Publisher("/particles", PointCloud, queue_size=10)
        self.particles_eig_pub = rospy.Publisher("/particles_eig", PointCloud, queue_size=10)
        # self.tracking_point_pub = rospy.Publisher("/tracked_point", Point, queue_size=10)
        self.attractor_red_pub = rospy.Publisher('/attractor_tracked', Marker, queue_size=10)
        self.attractor_green_pub = rospy.Publisher('/attractor', Marker, queue_size=10)
        self.attractor_ideal_pub = rospy.Publisher('/attractor_ideal', Marker, queue_size=10)

        self.text_ideal_Aonly_pub = rospy.Publisher('/A_ideal_delta', Marker, queue_size=10)
        self.text_pub = rospy.Publisher('/A_and_tank_state_text', Marker, queue_size=10)
        self.text_ideal_pub = rospy.Publisher('/A_ideal_text', Marker, queue_size=10)

        self.velocity_error_pub = rospy.Publisher('/vel_angvel_error_smoothed', Twist, queue_size=10)

        self.tank_state_pub = rospy.Publisher('/tank_state', Float32, queue_size=10)

        self.listener = tf.TransformListener()

        self.start = False
        self.finish_srv = rospy.Service("finish_calibration", Empty, self.finish_callback)
        self.reset_srv = rospy.Service("reset_move", Empty, self.reset_callback)


        self.curr_t = 0.0

        self.controller_type = 0 # 0 for direct output, 1 for game theory filter
        
        ctrlx = NegotiationController(dt=self.dt,  target_human=0.7, target_robot=0.7)
        ctrly = NegotiationController(dt=self.dt,  target_human=0.0, target_robot=-0.2)
        ctrltheta = NegotiationController(dt=self.dt,  target_human=0.0, target_robot=0.1)
        self.controllers = [ctrlx, ctrly, ctrltheta]


        #the string should be node name, not config type, config auto use dictionary, so no need to know the type
        self.controller_config_client = dynamic_reconfigure.client.Client("/iiwa_cartesian_impedance_bringup", timeout=0.5)
        # , config_callback=controller_config_callback) # don't print anything
        self.move_config_client = None 
        self.v_control_error_norm_prev = 0.0
        self.loop_last_time = rospy.Time.now()
    def vel_error_callback(self, data):
        alpha = 0.05
        #alpha * data.position.x + (1-alpha) * self.ee_vel[0]
        self.vel_angvel_err[0] = alpha * data.linear.x + (1-alpha) * self.vel_angvel_err[0]
        self.vel_angvel_err[1] = alpha * data.linear.y + (1-alpha) * self.vel_angvel_err[1]
        self.vel_angvel_err[2] = alpha * data.linear.z + (1-alpha) * self.vel_angvel_err[2]
        ang_vel_err = np.array([data.angular.x, data.angular.y, data.angular.z])
        rot_mat_ee = quaternion_matrix(self.object_pos[3:])[0:3,0:3]
        rot_v_ee_frame = rot_mat_ee.transpose() @ ang_vel_err.reshape(-1,1)
        self.vel_angvel_err[3] = alpha * rot_v_ee_frame[0] + (1-alpha) * self.vel_angvel_err[3]
        self.vel_angvel_err[4] = alpha * rot_v_ee_frame[1] + (1-alpha) * self.vel_angvel_err[4]
        self.vel_angvel_err[5] = alpha * rot_v_ee_frame[2] + (1-alpha) * self.vel_angvel_err[5]
    def tank_state_callback(self, data):
        # self.tank_state = data.data
        pass

    def finish_callback(self, emp):
        self.start = True
        return EmptyResponse()

    def reset_callback(self, emp):
        # reset to current location particles
        # print(self.pfds.pf.particles.shape)
        self.pfds.pf.particles = np.tile(np.array([self.desk.pos[0], self.desk.pos[1], self.desk.pos[2], 
                                                     -0.5, 0, 0,
                                                     0, -0.5, 0,
                                                     0, 0, -0.5]), (self.pfds.pf.particles.shape[0], 1))
        # print(self.pfds.pf.particles.shape)
        self.tank_state = 0.0
        return EmptyResponse()

    def get_optitrack_pos(self):
        try:
            (trans, rot) = self.listener.lookupTransform('Trash', 'Kuka', rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.logerr_throttle(3.0, "No tf from Trash to Kuka")
            return


        # print(trans)

    def right_ellipsoid_callback(self,data):
        self.right_M = data[:3,:3]
        self.human_right_ee = data[:3,3]

    def left_ellipsoid_callback(self,data):
        self.left_M = data[:3,:3]
        self.human_left_ee = data[:3,3]


    def get_wrench_rotation(self):
        try:
            (trans, rot) = self.listener.lookupTransform('world', 'robotiq_force_torque_frame_id', rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.logerr_throttle(3.0, "No tf from world to robotiq")
            return
        
        rotation_mat = quaternion_matrix(rot)[:3,:3]

        self.force_frame_rotation[:3,:3] = rotation_mat
        self.force_frame_rotation[-3:,-3:] = rotation_mat #no need to transform torque, people are looking at the arm and know its axis

    def force_callback(self, data):
        self.receive_force_time = data.header.stamp
        self.force_arr[0] = data.wrench.force.x
        self.force_arr[1] = data.wrench.force.y
        self.force_arr[2] = data.wrench.force.z
        self.force_arr[3] = data.wrench.torque.x
        self.force_arr[4] = data.wrench.torque.y
        self.force_arr[5] = data.wrench.torque.z

        self.get_wrench_rotation()
        F_extra  = self.force_frame_rotation[:3,:3].transpose() @ np.array([0.0, 0.0 , (-4.444)*9.81]).reshape(-1,1)
        rospy.loginfo_throttle(1.0, "F_extra = \n{:.0f}, {:.0f}, {:.0f}".format(F_extra.flatten()[0], F_extra.flatten()[1], F_extra.flatten()[2]))
        r_load = np.array([0,-0.04, 0.155]).reshape(-1,1) #compensate for clamps
        torque_extra = np.cross(r_load.flatten(), F_extra.flatten()).reshape(-1,1)
        F_extra_world = (self.force_frame_rotation[:3,:3] @ F_extra).flatten()
        torque_extra_world =(self.force_frame_rotation[:3,:3] @ torque_extra).flatten()
        
        rospy.loginfo_throttle(1.0, "F_extra_world = \n{:.0f}, {:.0f}, {:.0f}, torque_extra = \n{:.1f}, {:.1f}, {:.1f}".format(F_extra_world[0], F_extra_world[1], F_extra_world[2], torque_extra_world[0], torque_extra_world[1], torque_extra_world[2]))
        # self.force_arr[0:3] -= F_extra.reshape(-1)
        # self.force_arr[3:6] -= torque_extra.reshape(-1) # torque needs more thoughts
        # should the full torque when rotating be compensated, or just half??
        
        #republish the force sensor msg with load compensated
        data.wrench.force.x = self.force_arr[0]
        data.wrench.force.y = self.force_arr[1]
        data.wrench.force.z = self.force_arr[2]
        data.wrench.torque.x = self.force_arr[3]
        data.wrench.torque.y = self.force_arr[4]
        data.wrench.torque.z = self.force_arr[5]
        self.force_torque_compensate_pub.publish(data)

        self.force_arr = (self.force_frame_rotation @ self.force_arr.reshape(-1,1)).flatten()
        rospy.loginfo_throttle(1.0, "F_compensated = \n{:.0f}, {:.0f}, {:.0f}, torque_compensated = \n{:.1f}, {:.1f}, {:.1f}".format(self.force_arr[0], self.force_arr[1], self.force_arr[2], self.force_arr[3], self.force_arr[4], self.force_arr[5]))
    def robot_vel_callback(self, data):
        alpha = 0.05
        self.ee_vel[0] = alpha * data.position.x + (1-alpha) * self.ee_vel[0]
        self.ee_vel[1] = alpha * data.position.y + (1-alpha) * self.ee_vel[1]
        self.ee_vel[2] = alpha * data.position.z + (1-alpha) * self.ee_vel[2]
    
    def robot_ang_vel_callback(self, data):
        alpha = 0.4
        self.ee_vel[3] = alpha * data.position.x + (1-alpha) * self.ee_vel[3]
        self.ee_vel[4] = alpha * data.position.y + (1-alpha) * self.ee_vel[4]
        self.ee_vel[5] = alpha * data.position.z + (1-alpha) * self.ee_vel[5]

    def robot_callback(self, data):
        self.ee_pos[0] = data.position.x
        self.ee_pos[1] = data.position.y
        self.ee_pos[2] = data.position.z
        self.ee_pos[3] = data.orientation.x
        self.ee_pos[4] = data.orientation.y
        self.ee_pos[5] = data.orientation.z
        self.ee_pos[6] = data.orientation.w
        self.object_pos = copy.deepcopy(self.ee_pos)
        
        #self.get_optitrack_pos()
        self.counter -= 1
        if self.counter < 0:
            self.counter = 10
            (ang1, ang2, ang3) = euler_from_quaternion(self.object_pos[3:], axes='ryxz')
            rot_mat_ee = quaternion_matrix(self.object_pos[3:])[0:3,0:3]
            rot_v_ee_frame = rot_mat_ee.transpose() @ self.ee_vel[3:].reshape(-1,1)
            obj_pos = np.array([self.object_pos[0], self.object_pos[1], ang3])
            obj_vel = np.array([self.ee_vel[0], self.ee_vel[1], rot_v_ee_frame.flatten()[2]])
            # print("real_dt", real_dt)
            self.desk.pos = obj_pos
            self.desk.vel = obj_vel
            if self.pfds.xyt_history is None:
                self.pfds.xyt_history = np.hstack((obj_pos, obj_vel))
            else:
                self.pfds.xyt_history = np.vstack((self.pfds.xyt_history, np.hstack((obj_pos, obj_vel))))            
            if len(self.pfds.xyt_history) > 10: # 0.05 *10 = 0.5 sec
                self.pfds.xyt_history = self.pfds.xyt_history[1:]

    def object_callback(self, data):
        self.object_pos[0] = data.position.x
        self.object_pos[1] = data.position.y
        self.object_pos[2] = data.position.z
        self.object_pos[3] = data.orientation.x
        self.object_pos[4] = data.orientation.y
        self.object_pos[5] = data.orientation.z
        self.object_pos[6] = data.orientation.w
        # (roll, pitch, yaw) = euler_from_quaternion (orientation_list)


    def moving(self):
        
        # Modulation matrix from manipulability ellipsoid (A_full = M_full @ A_full)
        if np.linalg.norm(self.object_pos[:3]-self.human_left_ee) > np.linalg.norm(self.object_pos[:3]-self.human_right_ee):
            ellipsoid_matrix = self.left_M
        else:
            ellipsoid_matrix = self.right_M
        # M_full = np.vstack(( np.hstack((np.eye(3), np.zeros((3,3)))), \
        #                      np.hstack((np.zeros((3,3)), ellipsoid_matrix)) ))
        
        if self.pfds.xyt_history is not None  and self.pfds.xyt_history.shape[0] > 7:# it takes the size of the 6 element first thing
            F_human_sensed = np.array([self.force_arr[0], self.force_arr[1], self.force_arr[5]]).reshape(-1,1)
            # rospy.loginfo_throttle(1.0, "F_human_sensed = {}".format(self.force_arr))
            desired_v, attractor, A_est, alive_avg = self.pfds.run(F_human_sensed, self.desk.pos, self.desk.vel, ellipsoid_matrix)
            h = Header()
            h.stamp = rospy.Time.now()
            h.frame_id = 'world'

            desired_twist = TwistStamped()
            desired_twist.header = h
            desired_odom = Odometry()
            desired_odom.header = h

            theta_offset = 0.0
            # attractor, _ = self.desk.get_left_right_coordinate()
            # attractor = np.concatenate((attractor, self.desk.get_world_frame_theta_w()))
            # print("attractor x y theta w", attractor)
            ############## IMPEDEANCE GAIN ADJUSTMENT ##############
            dt_loop = (rospy.Time.now() - self.loop_last_time).to_sec()
            #usually 40hz 
            dt_standard = 1.0/40.0
            multiplier_loop = dt_loop / dt_standard
            self.loop_last_time = rospy.Time.now()
            #intropolate between HIGH and LOW depending on tank_state
            # set tank state to be norm of the force
            # rospy.loginfo_throttle(1.0, "desired_v = {}, actual_v = {}".format(desired_v[0:2], self.ee_vel[0:2]))
            v_control_error = np.array([self.vel_angvel_err[0], self.vel_angvel_err[1], self.vel_angvel_err[5]])
            rospy.loginfo_throttle(1.0, "v_control_error = {}".format(v_control_error))
            # ang_control_error = np.array([self.vel_angvel_err[3], self.vel_angvel_err[4], self.vel_angvel_err[5]])
            v_control_error_norm = np.linalg.norm(v_control_error)
            self.past_error_norm = np.roll(self.past_error_norm, -1)
            self.past_error_norm[-1] = v_control_error_norm
            num_past_points = int(0.5 // dt_loop) + 1
            past_error_norm_avg = np.mean(self.past_error_norm[-num_past_points:])

            # calculate dt and scale the error so that it is the same for different dt

            motion_force_activated = True # True for Motion, False for Force
            if motion_force_activated:
                if (v_control_error_norm - self.v_control_error_norm_prev) > 0.001:
                    delta_v_error = multiplier_loop* 0.1#13*(v_control_error_norm - self.v_control_error_norm_prev)
                else:
                    delta_v_error = multiplier_loop*-0.03
                self.v_control_error_norm_prev = v_control_error_norm

                # self.tank_state += delta_v_error
                self.tank_state = past_error_norm_avg / 0.2
                if self.tank_state < 0.0:
                    self.tank_state = 0.0
                elif self.tank_state > 1.0:
                    self.tank_state = 1.0

                velocity_error_pub_msg = Twist()
                velocity_error_pub_msg.linear.x = v_control_error[0]#x
                velocity_error_pub_msg.linear.y = v_control_error[1]#y
                velocity_error_pub_msg.linear.z = v_control_error[2]#ang z 
                velocity_error_pub_msg.angular.x = v_control_error_norm
                velocity_error_pub_msg.angular.y = delta_v_error*10
                velocity_error_pub_msg.angular.z = self.tank_state
                self.velocity_error_pub.publish(velocity_error_pub_msg)
            # rospy.loginfo_throttle(1.0, "delta error = {}".format(delta_v_error))
            
            else:
            #####FORCE Tank
                force_normalized = self.force_arr
                force_normalized[3:6] *= 6.0 # torque is too small
                force_mag = np.linalg.norm(force_normalized) 
                # rospy.loginfo_throttle(1.0, "force_mag = {}".format(force_mag))
                if force_mag < 0.0:
                    force_mag = 0.0
                else:
                    force_mag /= 35.0
                
                self.tank_state -= multiplier_loop * 0.04
                if self.tank_state < 0.0:
                    self.tank_state = 0.0
                self.tank_state += multiplier_loop*force_mag/3.0  
                if self.tank_state > 1.0:
                    self.tank_state = 1.0
                if force_mag > 1.0:
                    force_mag = 1.0
            ## Implement deadzone tank_state
            if self.tank_state < 0.15:
                self.tank_state = 0.0

            
            # self.tank_state -= 0.03
            # if self.tank_state < 0.0:
            #     self.tank_state = 0.0
            # self.tank_state += delta_v_error
            # if self.tank_state > 1.0:
            #     self.tank_state = 1.0


            # these numbers work well when robot is itself being controlled
            # self.tank_state = v_control_error_norm
            # # self.tank_state -= 0.15
            # # if self.tank_state < 0.0:
            # #     self.tank_state = 0.0
            # # self.tank_state += v_control_error
            # if self.tank_state > 1.0:
            #     self.tank_state = 1.0
            # rospy.loginfo_throttle(1.0, "tank_state = {}".format(self.tank_state))
            
            ##TEST
            #v_control_error range is 0 to 0.3, when error is large, tank state is large
            # self.tank_state = v_control_error / 0.2
            # if self.tank_state > 1.0: 
            #     self.tank_state = 1.0
            
            one_minus_tank_state = 1.0 - self.tank_state
            # delta_stiffness_rot = cc.STIFFNESS_ROT_HIGH - cc.STIFFNESS_ROT_LOW
            delta_stiffness_lin = cc.STIFFNESS_LIN_HIGH - cc.STIFFNESS_LIN_LOW
            delta_damping_lin = cc.DAMPING_LIN_HIGH - cc.DAMPING_LIN_LOW
            delta_damping_rot = cc.DAMPING_ROT_HIGH - cc.DAMPING_ROT_LOW
            # delta_resample = cc.RESAMPLE_PROPORTION_HIGH - cc.RESAMPLE_PROPORTION_LOW
            rot_stiff = cc.STIFFNESS_ROT_HIGH
            #cc.STIFFNESS_ROT_LOW + delta_stiffness_rot * one_minus_tank_state
            lin_stiff = cc.STIFFNESS_LIN_LOW + delta_stiffness_lin * one_minus_tank_state
            lin_damp = cc.DAMPING_LIN_LOW + delta_damping_lin * one_minus_tank_state
            rot_damp = cc.DAMPING_ROT_LOW + delta_damping_rot * one_minus_tank_state
            # resample_proportion = cc.RESAMPLE_PROPORTION_LOW + delta_resample * self.tank_state
            noise_xy = cc.NOISE_XY_LOW + (cc.NOISE_XY_HIGH - cc.NOISE_XY_LOW) * self.tank_state
            noise_theta = cc.NOISE_THETA_LOW + (cc.NOISE_THETA_HIGH - cc.NOISE_THETA_LOW) * self.tank_state

            if self.start:
                self.controller_config_client.update_configuration({"stiffness_rot": rot_stiff, 
                                                                    "stiffness_lin": lin_stiff, 
                                                                    "damping_lin": lin_damp, 
                                                                    "damping_rot": rot_damp,
                                                                    "stiffness_lin_z":cc.STIFFNESS_LIN_HIGH, 
                                                                    "damping_lin_z":cc.DAMPING_LIN_HIGH})
            
            self.move_config_client.update_configuration({"noise_x": noise_xy,
                                                            "noise_y": noise_xy,
                                                            "noise_theta": noise_theta,
                                                            "ellipsoid_matrix":ellipsoid_matrix})
                                                        #   "resample_proportion": resample_proportion})
                                                        #    "weight_velocity": 1.0,  # these will be fixed for now
                                                        #    "weight_A": 0.0,


            if self.controller_type == 0:
                #desired twist saves the position of the attractor
                desired_twist.twist.linear.x = attractor[0]
                desired_twist.twist.linear.y = attractor[1]
                desired_twist.twist.angular.z = np.arctan2(self.ee_pos[1], self.ee_pos[0]) #this is correct!

                desired_odom.pose.pose.position.x = attractor[0]
                desired_odom.pose.pose.position.y = attractor[1]
                # get quaternion from yaw
            elif self.controller_type == 1:
                x_negotiate,_,_ = self.controllers[0].update_x_negotiate(attractor[0])
                y_negotiate,_,_ = self.controllers[1].update_x_negotiate(attractor[1])
                theta_negotiate,_,_ = self.controllers[2].update_x_negotiate(attractor[2])
                desired_twist.twist.linear.x = x_negotiate[0,0]
                desired_twist.twist.linear.y = y_negotiate[0,0]
                desired_twist.twist.angular.z = theta_negotiate[0,0] + theta_offset
                
                desired_odom.pose.pose.position.x = x_negotiate[0,0]
                desired_odom.pose.pose.position.y = y_negotiate[0,0]
                # get quaternion from yaw
            #desired twist currently send the orientation of the attractor
            desired_odom.pose.pose.position.z = DESIRED_HEIGHT
            
            quat = quaternion_from_euler(DESIRED_PITCH, -desired_twist.twist.angular.z, attractor[2] ,axes='ryxz')

            desired_twist.twist.angular.x = 0.0#roll
            desired_twist.twist.angular.y = DESIRED_PITCH #pitch
            desired_twist.twist.linear.z = 0.0

            desired_odom.pose.pose.orientation.x = quat[0]
            desired_odom.pose.pose.orientation.y = quat[1]
            desired_odom.pose.pose.orientation.z = quat[2]
            desired_odom.pose.pose.orientation.w = quat[3]

            #for now publish zero twist
            
            desired_odom.twist.twist.linear.x = desired_v[0]
            desired_odom.twist.twist.linear.y = desired_v[1]
            desired_odom.twist.twist.linear.z = - 0.5 * (self.ee_pos[2] - desired_odom.pose.pose.position.z)
            desired_odom.twist.twist.angular.x = 0
            desired_odom.twist.twist.angular.y = 0
            desired_odom.twist.twist.angular.z = desired_v[2]

            # A_est = np.diag(A_only_est(self.pfds.xyt_history))
            # rospy.loginfo_throttle(3.0, "Overwrite A_est !!!!!!!!!!")

            A_full = np.diag([A_est[2,2], -1.0, -1.0, A_est[0,0], A_est[1,1], A_Z]) #order ang and then x y z
            # A_full = M_full @ A_full
            desired_Ab = self.get_desired_Ab(A_full, desired_odom.pose.pose)
            
            particles_viz = self.pfds.pf.particles
            particles_pub_msg = PointCloud()
            particles_pub_msg.header = h
            for i in range(particles_viz.shape[0]):
                particles_pub_msg.points.append(Point32(particles_viz[i,0], particles_viz[i,1], particles_viz[i,2]))
            self.particles_pub.publish(particles_pub_msg)
            # particles_pub_eig_msg = PointCloud()
            # particles_pub_eig_msg.header = h
            # for i in range(particles_viz.shape[0]):
            #     particles_pub_eig_msg.points.append(Point32(particles_viz[i,3], particles_viz[i,7], particles_viz[i,11]))
            # self.particles_eig_pub.publish(particles_pub_eig_msg)

            marker_red_instant = make_sphere(desired_twist.twist.linear.x, desired_twist.twist.linear.y, 0.0, 0.1)

            self.attractor_red_pub.publish(marker_red_instant)

            marker_green_long_term = make_sphere(attractor[0], attractor[1], attractor[2], 0.05, color=[0.0, 1.0, 0.0])
            
            self.attractor_green_pub.publish(marker_green_long_term)

            attractor_text = attractor.copy()
            attractor_text[2] = 0.0
            text_marker = make_text("A: [{:.2f}, {:.2f}, {:.2f}], state = {:.0f} %, {:.0f}Hz".format(A_est[0,0], A_est[1,1], A_est[2,2], self.tank_state*100, 1/dt_loop),
                        attractor_text)
            
            self.text_pub.publish(text_marker)
            tank_state_msg = Float32()
            tank_state_msg.data = self.tank_state
            self.tank_state_pub.publish(tank_state_msg)


            # goal_pred, eig_pred, A_est_only = pred_goal_from_hist(self.pfds.xyt_history)
            # marker_ideal_attractor = make_sphere(goal_pred[0], goal_pred[1], 0.0, 0.1, color=[0.0, 0.0, 1.0])
            # self.attractor_ideal_pub.publish(marker_ideal_attractor)

            # goal_pred[2] = 0.3
            # text_marker = make_text("A: [{:.2f}, {:.2f}, {:.2f}]".format(A_est_only[0], A_est_only[1], A_est_only[2]), goal_pred)
            # self.text_ideal_Aonly_pub.publish(text_marker)
            
            # goal_pred[2] = 0.2
            # text_marker = make_text("A: [{:.2f}, {:.2f}, {:.2f}]".format(eig_pred[0], eig_pred[1], eig_pred[2]), goal_pred)
            # self.text_ideal_pub.publish(text_marker)
            
            # save xyt_history with time stamp
            # np.save("xyt_history_{}.npy".format(time.time()), self.pfds.xyt_history)
            # if self.counter_save == 0:
            #     np.savetxt("/home/yifei/kuka_ws/data/xyt_history_{}.txt".format(time.time()), self.pfds.xyt_history)
            #     self.counter_save = 10
            # else:
            #     self.counter_save -= 1
        if not self.start:
            # self.tank_state = 1.1
            A_INIT_L = -0.5
            A_INIT_R = -0.3
            # desired_y = 5.0
            if int(self.curr_t) % 10 > 5:
                A = np.diag([A_INIT_L, A_INIT_L, A_Z]) # x, y, z not x, y, theta as in PFDS
                desired_y = -0.3
                desired_x = 0.9
            else:
                A = np.diag([A_INIT_R, A_INIT_R, A_Z]) # x, y, z not x, y, theta as in PFDS
                desired_y = 0.3
                desired_x = 0.9
            
            # if int(self.curr_t) % 20 < 10:
            b = np.array([desired_x, desired_y, DESIRED_HEIGHT]).reshape(-1,1)
            # Direct position control is a lot eaiser and don't involve two levels of stiffness
            # rospy.loginfo_throttle(1.0, self.ee_pos[:3])
            err = A @ (self.ee_pos[:3].reshape(-1,1) - b)
            
            h = Header()
            h.frame_id = 'world'
            h.stamp = rospy.Time.now()
            
            desired_twist = self.get_desired_twist(h, b)
            desired_odom = self.get_desired_odom(h, b, err, desired_twist.twist.angular.z)

            A_full = np.diag([-1.0, -1.0, -1.0, A[0,0], A[1,1], A[2,2]]) #order ang and then x y z
            # A_full = M_full @ A_full

            desired_Ab = self.get_desired_Ab(A_full, desired_odom.pose.pose)
            
            self.curr_t = (rospy.Time.now() - self.time_start).to_sec()
                            
            

        self.ds_pub.publish(desired_Ab)
        self.assist_pub.publish(desired_twist)
        self.odom_pub.publish(desired_odom)

    def get_desired_Ab(self, A, b):
        desired_Ab = MatrixVec()
        desired_Ab.mat = A.flatten("F")
        desired_Ab.pose = b
        return desired_Ab
    
    def get_desired_odom(self, header, attractor, lin_v, yaw ):
        desired_odom = Odometry()
        desired_odom.header = header
        # (angularx, angulary, angularz) = euler_from_quaternion (np.array([70710678118, 0.0, 70710678118, 0.0]))
        # print(angularx, angulary, angularz)
    
        desired_odom.pose.pose.position.x = attractor[0]
        desired_odom.pose.pose.position.y = attractor[1]
        desired_odom.pose.pose.position.z = attractor[2]
        # get quaternion from yaw

        #NOTICE negative sign on last x rotation # + 0.2*np.sin(self.curr_t/3)
        quat = quaternion_from_euler(DESIRED_PITCH , -yaw, -0.2*np.sin(self.curr_t), axes='ryxz')
        # quat = quaternion_from_euler(0, 0, 0, axes='rxyx')
        desired_odom.pose.pose.orientation.x = quat[0]
        desired_odom.pose.pose.orientation.y = quat[1]
        desired_odom.pose.pose.orientation.z = quat[2]
        desired_odom.pose.pose.orientation.w = quat[3]
        zero_twist = Twist()
        desired_odom.twist.twist = zero_twist
        desired_odom.twist.twist.linear.x = lin_v[0]
        desired_odom.twist.twist.linear.y = lin_v[1]
        desired_odom.twist.twist.linear.z = lin_v[2]
        desired_odom.twist.twist.angular.x = 0.0
        desired_odom.twist.twist.angular.y = 0.0
        desired_odom.twist.twist.angular.z = 0.0 # this is not used since we are not doing angular velocity control at the moment
        return desired_odom
    
    def get_desired_twist(self, header, attractor):
        desired_twist = TwistStamped()
        desired_twist.header = header
        desired_twist.twist.linear.x = attractor[0]
        desired_twist.twist.linear.y = attractor[1]
        desired_twist.twist.linear.z = attractor[2]
        desired_twist.twist.angular.x = 0.0# DESIRED_ROLL
        desired_twist.twist.angular.y = DESIRED_PITCH
        desired_twist.twist.angular.z = 0.0#-1.5*np.sin(self.curr_t/4)
        return desired_twist

    def config_callback(self, config, level):
        self.controller_type = config['controller_type']
        for i in self.controllers:
            i.L_human[0] = config['L_human_0']
            i.L_human[1] = config['L_human_1']
            i.L_robot[0] = config['L_robot_0']
            i.L_robot[1] = config['L_robot_1']
            i.update_lambda_robot(config['lambda_robot'])
        self.pfds.update_config(config)
        return config


def main():
    rospy.init_node('move')
    freq=10
    r = rospy.Rate(freq)
    move = Move(1/freq)
    srv = Server(IntentCapabilityHRIConfig, move.config_callback)
    move.move_config_client =  dynamic_reconfigure.client.Client("/move", timeout=0.5)
    # , config_callback=move_config_callback)
    while not rospy.is_shutdown():
        move.moving()
        r.sleep()



#print("Avg F_human / 2 over time = ", F_human_sum / 2 / (dt * max_idx))

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
