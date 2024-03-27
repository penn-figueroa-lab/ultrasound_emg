#!/usr/bin/env python3
import rospy
# from kr_planning_msgs.msg import SplineTrajectory
from kr_planning_msgs.msg import PlanTwoPointGoal, PlanTwoPoseHumanRobotAction, VoxelMap
import numpy as np
# import matplotlib.pyplot as plt
import pandas as pd
from copy import deepcopy
from geometry_msgs.msg import Point
from visualization_msgs.msg import MarkerArray, Marker
from actionlib import SimpleActionClient
import random 
import actionlib
# from std_msgs.msg import Bool
from std_msgs.msg import Int32
from tqdm import tqdm
import pickle
import yaml
import csv

#sudo apt install python3-pcl
from datetime import datetime


import rospy
import tf
from std_msgs.msg import Header, Float32
from geometry_msgs.msg import Wrench, Vector3, Point
from gazebo_msgs.srv import ApplyBodyWrench
from geometry_msgs.msg import WrenchStamped, Pose, Twist, TwistStamped
from tf.transformations import euler_from_quaternion, quaternion_from_euler, quaternion_matrix


import copy

planner_id = 3
# 0 no controller, 1 proposed(40hz), 2 admittance, 3 task_adaptation, 4 proposed with curobo(10hz)

use_odom_bool = False

path_prefix = "/home/yifei/kuka_ws/data/"

filename = '/home/yifei/kuka_ws/src/intent-capability-hri/ECI_goals_01-30.csv'
#load file as numpy array
goals = np.loadtxt(filename, delimiter=',')
print(goals.shape)
# exit()
## Helper Functions
def sample_points_on_circle_band(n, inner_radius, outer_radius, min_angle, max_angle, theta_limit):
    """
    Randomly samples n points on a circular band defined by inner and outer radii,
    and within a specified angular range.

    :param n: Number of points to sample.
    :param inner_radius: Inner radius of the circular band.
    :param outer_radius: Outer radius of the circular band.
    :param min_angle: Minimum angle in degrees.
    :param max_angle: Maximum angle in degrees.
    :return: Array of sampled points in Cartesian coordinates (x, y).
    """

    # Randomly sample radii and angles
    radii = np.random.uniform(inner_radius, outer_radius, n)
    angles = np.random.uniform(np.radians(min_angle), np.radians(max_angle), n)

    # Convert to Cartesian coordinates
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    theta = np.random.uniform(-theta_limit, theta_limit, n)
    states = np.column_stack((x, y, theta))
    return states

def list_of_pose_to_numpy(pose_list):
    pose_array = np.zeros((len(pose_list), 7))
    for i in range(len(pose_list)):
        pose_array[i,0] = pose_list[i].position.x
        pose_array[i,1] = pose_list[i].position.y
        pose_array[i,2] = pose_list[i].position.z
        pose_array[i,3] = pose_list[i].orientation.x
        pose_array[i,4] = pose_list[i].orientation.y
        pose_array[i,5] = pose_list[i].orientation.z
        pose_array[i,6] = pose_list[i].orientation.w
    return pose_array

def list_of_wrench_to_numpy(wrench_list):
    wrench_array = np.zeros((len(wrench_list), 6))
    for i in range(len(wrench_list)):
        wrench_array[i,0] = wrench_list[i].force.x
        wrench_array[i,1] = wrench_list[i].force.y
        wrench_array[i,2] = wrench_list[i].force.z
        wrench_array[i,3] = wrench_list[i].torque.x
        wrench_array[i,4] = wrench_list[i].torque.y
        wrench_array[i,5] = wrench_list[i].torque.z
    return wrench_array

def list_of_twist_to_numpy(twist_list):
    twist_array = np.zeros((len(twist_list), 6))
    for i in range(len(twist_list)):
        twist_array[i,0] = twist_list[i].linear.x
        twist_array[i,1] = twist_list[i].linear.y
        twist_array[i,2] = twist_list[i].linear.z
        twist_array[i,3] = twist_list[i].angular.x
        twist_array[i,4] = twist_list[i].angular.y
        twist_array[i,5] = twist_list[i].angular.z
    return twist_array


class Evaluater:
    def __init__(self):


        self.num_planners = 1
        self.num_trials = 10
        self.ee_vel = np.zeros(6)
        self.ee_pos = np.zeros(7)
        self.object_pos = np.zeros(7)
        self.obj_vel_3 = np.zeros(3)
        self.obj_pos_3 = np.zeros(3)
        self.counter = 0
        self.client_list = []
        
        self.map_x_min = 0.25
        self.map_x_max = 0.75
        self.map_y_min = -0.4
        self.map_y_max = 0.4
        self.map_roll_min = -1.4
        self.map_roll_max = 1.4

        print("VERIFY!!!!!!!! planner_id = ", planner_id)

        for i in range(self.num_planners): #  0, 1, 2, ... not gonna include the one with no suffix
            self.client_list.append(SimpleActionClient('/plan_local_trajectory', PlanTwoPoseHumanRobotAction))
            
        self.start_and_goal_pub = rospy.Publisher('/start_and_goal', MarkerArray, queue_size=10, latch=True)

        self.wait_for_things = True #rospy.get_param('/local_plan_server0/trajectory_planner/use_tracker_client')
   
        # this will still be useful for setting manipulator state #### Set state will happen in controller ######
        # self.set_state_pub      = rospy.Publisher( '/' + self.mav_name + '/set_state', PositionCommand, queue_size=1, latch=False)
        
        # get robot position
        self.robot_state_sub = rospy.Subscriber("/iiwa/task_states", Pose, self.robot_callback, tcp_nodelay=True)
        self.robot_vel_sub = rospy.Subscriber("/iiwa/ee_vel_cartimp", Pose, self.robot_vel_callback, tcp_nodelay=True)
        self.robot_angVel_sub = rospy.Subscriber("/iiwa/ee_angvel_cartimp", Pose, self.robot_ang_vel_callback, tcp_nodelay=True)
        self.effort_temp = 0.0 # these need to be defined before the callback
        self.effort_counter = 0


        # rospy.Subscriber("/local_plan_server0/trajectory", SplineTrajectory, self.callback)
        self.success = np.zeros((self.num_trials, self.num_planners), dtype=bool)
        self.success_detail = -2*np.ones((self.num_trials, self.num_planners), dtype=int)
        self.traj_time = np.zeros((self.num_trials, self.num_planners))
        self.traj_cost = np.zeros((self.num_trials, self.num_planners))
        self.traj_jerk = np.zeros((self.num_trials, self.num_planners))
        self.poly_compute_time = np.zeros((self.num_trials, self.num_planners))
        self.compute_time_front = np.zeros((self.num_trials, self.num_planners))
        self.compute_time_back = np.zeros((self.num_trials, self.num_planners))
        self.tracking_error = np.zeros((self.num_trials, self.num_planners))
        self.effort = np.zeros((self.num_trials, self.num_planners)) #unit in force integrated over time
        self.collision_front = np.zeros((self.num_trials, self.num_planners), dtype=bool)
        self.collision_cnt = np.zeros((self.num_trials, self.num_planners), dtype=bool)
        self.dist_to_goal = np.zeros((self.num_trials, self.num_planners))

        self.publisher()

    def robot_vel_callback(self, data):
        alpha = 0.2
        self.ee_vel[0] = alpha * data.position.x + (1-alpha) * self.ee_vel[0]
        self.ee_vel[1] = alpha * data.position.y + (1-alpha) * self.ee_vel[1]
        self.ee_vel[2] = alpha * data.position.z + (1-alpha) * self.ee_vel[2]
    
    def robot_ang_vel_callback(self, data):
        alpha = 0.95
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
            #print("orientation:", self.object_pos[3:])
            self.obj_pos_3 = np.array([self.object_pos[0], self.object_pos[1], ang3])
            self.obj_vel_3 = np.array([self.ee_vel[0], self.ee_vel[1], rot_v_ee_frame.flatten()[2]])

    def sample_in_map(self):

        curr_sample_idx = 0
        start_end_feasible = True
        max_iter = 200
        lower_limit = np.array([self.map_x_min, self.map_y_min, self.map_roll_min])
        upper_limit = np.array([self.map_x_max, self.map_y_max, self.map_roll_max])
        norm_dist = np.linalg.norm(upper_limit - lower_limit)
        while curr_sample_idx < max_iter:
        
            rand_start_x = 0.80#random.uniform(self.map_x_min, self.map_x_max)
            rand_start_y = 0.0#random.uniform(self.map_y_min, self.map_y_max)
            rand_start_z = 0.0#random.uniform(self.map_roll_min, self.map_roll_max)

            xyz =sample_points_on_circle_band(1, 0.78, 1.08 , -70, 70, 1.4) 
            #full range is 0.5 to 1.3
            rand_goal_x = xyz[0][0]
            rand_goal_y = xyz[0][1]
            rand_goal_z = xyz[0][2]

            start = np.array([rand_start_x, rand_start_y, rand_start_z])
            goal = np.array([rand_goal_x, rand_goal_y, rand_goal_z])
            
            curr_sample_idx += 1
            dis = np.linalg.norm(start - goal)
            
            #check dist is far and collision free
            if dis > 0.3 * norm_dist:
                break
            
        if curr_sample_idx >= max_iter:
            rospy.logerr("Failed to sample a start and goal pair far enough apart")
            start_end_feasible = False
        return start, goal, start_end_feasible


    def odom_callback(self, msg):
        self.odom_data = msg.pose.pose.position

    def sim_output_callback(self, msg): #ToDo: This also has odom inside, consider combine with previous callback 
        self.effort_temp += np.mean(msg.motor_rpm) #rpm
        self.effort_counter += 1

    def point_clouds_callback(self, msg):
        tqdm.write("point cloud CALLBACK received")
        points_list = []

        for data in pc2.read_points(msg, skip_nans=True):
            points_list.append([data[0], data[1], data[2]])

        self.pcl_data = pcl.PointCloud(np.array(points_list, dtype=np.float32))


        self.kdtree = self.pcl_data.make_kdtree_flann()

        return
    

    def evaluate_collision(self, pts, tol = -1.0):
        if tol < 0:
            tol = self.mav_radius
        min_sq_dist = 1000000
        min_idx = 0
        if len(pts) > 200:
            pts = pts[::10]
        # tqdm.write("odom length = " + str( len(pts)))
        if self.kdtree is None:
            rospy.sleep(0.1)
            rospy.logerr("No KD Tree, skipping collision check")
            return True # assume collision
        for pt_idx in range(len(pts)):
            pt = pts[pt_idx]
            sp = pcl.PointCloud()
            sps = np.zeros((1, 3), dtype=np.float32)
            sps[0][0] = pt.x
            sps[0][1] = pt.y
            sps[0][2] = pt.z
            sp.from_array(sps)
            [ind, sqdist] = self.kdtree.nearest_k_search_for_cloud(sp, 1) #which pointcloud pt has min dist
            if sqdist[0][0] < min_sq_dist:

                min_sq_dist = sqdist[0][0]
                min_idx = pt_idx

        # tqdm.write("min dist = " + str(np.sqrt(min_sq_dist)) + "@ traj percentage = " + str(min_idx/len(pts)))
        if np.sqrt(min_sq_dist) < tol:

            if tol == self.mav_radius:
                rospy.logwarn("Collision Detected")
                print("np.sqrt(min_sq_dist) is ", np.sqrt(min_sq_dist))
            return True
        else:
            # print("min dist is ", np.sqrt(min_sq_dist))
            return False            

    def computeJerk(self, traj):
        # creae empty array for time
        t_vec = np.array([])
        # create empty array for jerk norm sq
        jerk_sq = np.array([])
        # jerk = 0
        dt = .01
        for t in np.arange(0, traj.data[0].t_total, dt):
            t_vec = np.append(t_vec, t)
            jerk_sq = np.append(jerk_sq, (np.linalg.norm(evaluate(traj, t, 3)))**2 )
        return np.sqrt(np.trapz(jerk_sq, t_vec))/traj.data[0].t_total
        

    def computeCost(self, traj, rho):
        time = traj.data[0].t_total
        cost = rho*time + self.computeJerk(traj)
        return cost
    def send_start_goal_viz(self, msg):
        start_and_goal = MarkerArray()
        start = Marker()
        start.header.frame_id = "world"
        start.header.stamp = rospy.Time.now()
        start.pose.position = msg.p_init.position
        start.pose.orientation.w = 1
        start.color.g = 1
        start.color.a = 1
        start.type = 2
        start.scale.x = start.scale.y = start.scale.z = 0.05
        goal = deepcopy(start)
        goal.pose.position = msg.p_final.position
        goal.id = 1
        goal.color.r = 1
        goal.color.g = 0
        goal.color.b = 1

        goalflat = deepcopy(goal)
        goalflat.scale.x = goalflat.scale.y = goalflat.scale.z = 0.1
        goalflat.pose.position.z = 0.0
        
        # start_and_goal.markers.append(start)
        start_and_goal.markers.append(goal)
        start_and_goal.markers.append(goalflat)
        # self.path_pub.publish(msg)
        self.start_and_goal_pub.publish(start_and_goal) # viz
    def publisher(self):
        print("Running ", self.num_planners, "planner combinations for", self.num_trials, "trials")
        for i in range(self.num_planners):
            print("waiting for action server ", i)
            self.client_list[i].wait_for_server()
            
       
            
        print("All action server connected, number of planners = ", self.num_planners)
        now = datetime.now()
        file_name_save_time = now.strftime("%m-%d_%H-%M-%S")

        param_names = rospy.get_param_names()
        params = {}
        for name in param_names:
            params[name] = rospy.get_param(name)

        with open(path_prefix+'ECI_params_'+file_name_save_time+'.yaml', 'w') as f:
            yaml.dump(params, f)

        try:
            with open(path_prefix+'ECI_single_line_'+file_name_save_time+'.csv', 'w') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(['seed_val',
                                     'start_end_feasible', 'success', 'success_detail', 'traj_time(s)', 'traj_length(m)', 'traj_jerk', 'traj_effort(rpm)',
                                     'compute_time_poly(ms)', 'compute_time_frontend(ms)', 'compute_time_backend(ms)', 
                                     'tracking_error(m) avg', 'collision_frontend', 'collision_status','dist_to_goal(m)'])

                for i in tqdm(range(self.num_trials)):
                    if rospy.is_shutdown():
                        break
                    ######## CHANGE MAP ######
                    seed_val = i
                    # random.seed(seed_val+13)

                    start_end_feasible = True
                    ####### DEFINE START #####
                    # define start location not actually sending pos_msg since we using a tracker to get there
                    if not use_odom_bool: # hopefully this is always the case, we can specify the start
                        # pos_msg = PositionCommand() # change position in simulator
                        # pos_msg.header.frame_id = "world"
                        # pos_msg.header.stamp = rospy.Time.now()
                        pos_cmd_pos = Point()
                        # start, end, start_end_feasible = self.sample_in_map() #0.1 for fillMap_inflation
                        end = goals[seed_val]
                        pos_cmd_pos.x = 0.65#start[0]
                        pos_cmd_pos.y = 0.0#start[1]
                        pos_cmd_pos.z = 0.0#start[2]

                        print("end = ", end)
                        ##### GO TO START #####
                    if not use_odom_bool and self.wait_for_things:  #this needs to be done for every client 
                        pass
                        ##### SET GOAL #####
                    msg = PlanTwoPointGoal()
                    if use_odom_bool:
                        msg.p_init.position.x = self.obj_pos_3[0]
                        msg.p_init.position.y = self.obj_pos_3[1]
                        msg.p_init.position.z = self.obj_pos_3[2]
                    else:
                        msg.p_init.position = pos_cmd_pos # if starting from random position
                    # set goal to be random
                    msg.p_final.position.x = end[0]
                    msg.p_final.position.y = end[1]
                    msg.p_final.position.z = end[2] # this is not hardware, so set it to whatever
                    msg.check_vel = False

                    for client_idx in range(self.num_planners): # send goal to center!!!
                        msg_reset = PlanTwoPointGoal()
                        msg_reset.p_final.position = pos_cmd_pos
                        client = self.client_list[client_idx]
                        client.send_goal(msg_reset) #motion
                        self.send_start_goal_viz(msg_reset)
                        client.wait_for_result(rospy.Duration.from_sec(30.0)) # wait for 30 seconds
                        result = client.get_result() #discard result

                    for client_idx in range(self.num_planners):
                        client = self.client_list[client_idx]
                        client.send_goal(msg) #motion
                    for client_idx in range(self.num_planners):
                        client = self.client_list[client_idx]

                        
                        total_wait_time = 20.0

                        while not client.get_result():
                            client.wait_for_result(rospy.Duration.from_sec(0.1)) 
                            total_wait_time -= 0.1
                            self.send_start_goal_viz(msg)
                            if total_wait_time < 0:
                                break
                        # Waits for the server to finish performing the action.
                        
                
                        # stop accumulating the effort
                        result = client.get_result()
                        if not result:
                            tqdm.write("Server Failure: trial " + str(i) + " planner: " + str(planner_id))
                        else:
                            self.success[i,client_idx] = result.success
                            tqdm.write("Solve Status: trial "+ str(seed_val) + " planner: " + str(planner_id) + " status: "+ str(result.success))
                            success_bool = np.array(result.success, dtype=bool)
                            pose_traj = list_of_pose_to_numpy(result.pose)
                            twist_traj = list_of_twist_to_numpy(result.twist)
                            wrench_human_traj = list_of_wrench_to_numpy(result.human_sensed)
                            wrench_robot_traj = list_of_wrench_to_numpy(result.robot_sensed)
                            wrench_robot_effort_traj = list_of_wrench_to_numpy(result.robot_effort_world)
                            tank_state_traj = result.tank_level
                            time_traj = np.array(result.time_vec)
                            goal_pose = np.array([msg.p_final.position.x, msg.p_final.position.y, msg.p_final.position.z, msg.p_final.orientation.x, msg.p_final.orientation.y, msg.p_final.orientation.z, msg.p_final.orientation.w])
                            # write numpy to npy file
                            np.savez(path_prefix+'ECI_traj_trial_'+str(i)+'_planner_'+str(planner_id)+'.npz', 
                                     pose_traj=pose_traj, 
                                     twist_traj=twist_traj,
                                     wrench_human_traj=wrench_human_traj, 
                                     wrench_robot_traj=wrench_robot_traj, 
                                     wrench_robot_effort_traj=wrench_robot_effort_traj,
                                     time_traj=time_traj, 
                                     success_bool=success_bool, 
                                     goal_pose=goal_pose, 
                                     tank_state_traj=tank_state_traj)
                            if result.success:
                            
                                pass
                            else:
                                tqdm.write("Server Failure: trial " + str(i) + " planner: " + str(planner_id))
                                self.success_detail[i,client_idx] = -1
                        
                #dont have traj length, so just put 0
                            # csv_writer.writerow([seed_val,
                            #                     start_end_feasible, 
                            #                     self.success[i,client_idx], self.success_detail[i,client_idx], self.traj_time[i,client_idx], 0.0, self.traj_jerk[i,client_idx], self.effort[i,client_idx],
                            #                     self.poly_compute_time[i,client_idx], self.compute_time_front[i,client_idx], self.compute_time_back[i,client_idx],
                            #                     self.tracking_error[i,client_idx], self.collision_front[i,client_idx], self.collision_cnt[i,client_idx], self.dist_to_goal[i, client_idx]])

        except KeyboardInterrupt:
            tqdm.write("Keyboard Interrupt!")



        # data_all = {}

        # data_all['success'] = self.success
        # data_all['success_detail'] = self.success_detail
        # data_all['traj_time'] = self.traj_time
        # data_all['traj_cost'] = self.traj_cost
        # data_all['traj_jerk'] = self.traj_jerk
        # data_all['poly_compute_time'] = self.poly_compute_time
        # data_all['compute_time_front'] = self.compute_time_front
        # data_all['compute_time_back'] = self.compute_time_back
        # data_all['tracking_error'] = self.tracking_error
        # data_all['effort'] = self.effort
        # data_all['collision_front'] = self.collision_front
        # data_all['collision_cnt'] = self.collision_cnt
        # data_all['dist_to_goal'] = self.dist_to_goal

        # print(self.success)
        # print("Legend: -2: not run, -1: server failure, 0: front failure, 1: front success, 2: poly success, 3: back success")
        # print(self.success_detail)
        # print("Traj Time", self.traj_time)
        # print("Traj Cost",self.traj_cost)
        # print("Jerk", self.traj_jerk)
        # print("Compute Time Front", self.compute_time_front)
        # print("Compute Time Poly", self.poly_compute_time)
        # print("Compute Time Back", self.compute_time_back)
        # print("Tracking Error", self.tracking_error)
        # print("Effort", self.effort)
        # print("Is Collide Front", self.collision_front)
        # print("Is Collide", self.collision_cnt)
        # print("Distance To Goal", self.dist_to_goal)



        # #save pickle with all the data, use date time as name
        # # with open('ECI_eval_data_'+file_name_save_time+'.pkl', 'wb') as f:
        # #     pickle.dump([self.success, self.success_detail, self.traj_time, self.traj_cost, self.traj_jerk, self.traj_compute_time, self.compute_time_front, self.compute_time_back, self.tracking_error, self.effort], f)
        # with open('ECI_Result_' + file_name_save_time + '.pkl', 'wb') as f:
        #     pickle.dump(data_all, f)# result and config

        # #create variables to store the average values
        # success_front_rate = np.sum(self.success_detail >= 1, axis = 0)/self.success.shape[0]
        # success_rate_avg = np.sum(self.success,axis = 0)/self.success.shape[0]
        # traj_time_avg = np.sum(self.traj_time,axis = 0) / np.sum(self.success, axis = 0)
        # # traj_cost_avg = np.sum(self.traj_cost[self.success]) / np.sum(self.success)
        # traj_jerk_avg = np.sum(self.traj_jerk, axis = 0) / np.sum(self.success, axis = 0)
        # poly_compute_time_avg = np.sum(self.poly_compute_time, axis = 0) / np.sum(self.success, axis = 0)
        # compute_time_front_avg = np.sum(self.compute_time_front, axis = 0) / np.sum(self.success, axis = 0)
        # compute_time_back_avg = np.sum(self.compute_time_back, axis = 0) / np.sum(self.success, axis = 0)
        # tracking_error_avg = np.sum(self.tracking_error, axis = 0) / np.sum(self.success, axis = 0)
        # effort_avg = np.sum(self.effort, axis = 0) / np.sum(self.success, axis = 0)
        # collision_rate_avg = np.sum(self.collision_front, axis = 0) / np.sum(self.success, axis = 0)
        # collision_rate_avg = np.sum(self.collision_cnt, axis = 0) / np.sum(self.success, axis = 0)
        # dist_to_goal_avg = np.sum(self.dist_to_goal, axis = 0) / np.sum(self.success, axis = 0)
        # # rewrite the above section with defined avg variables
        # print("frontend success rate: " + str(success_front_rate)+ " out of " + str(self.success.shape[0]))
        # print("success rate: " + str(success_rate_avg)+ " out of " + str(self.success.shape[0]))
        # print("avg traj time(s): " + str(traj_time_avg))
        # # print("avg traj cost(time + jerk): " + str(traj_cost_avg))
        # print("avg traj jerk: " + str(traj_jerk_avg))
        # print("avg compute time front(ms): " + str(compute_time_front_avg))
        # print("avg compute time poly(ms): " + str(poly_compute_time_avg))
        # print("avg compute time back(ms): " + str(compute_time_back_avg))
        # print("avg tracking error(m): " + str(tracking_error_avg))
        # print("avg effort(rpm): " + str(effort_avg))# this is bugg!! need to consider success
        # print("avg dist to goal: " + str(dist_to_goal_avg))
        # print("collision rate: " + str(collision_rate_avg))


        
        # # save the avg values to a csv file by appending to the end of the file
        # csv_name = 'ECI_Summary_'+file_name_save_time+'.csv'

        # with open(csv_name, 'w') as f: #result summary
        #     writer = csv.writer(f)
        #     writer.writerow([' Run:' + str(self.num_trials),'success rate', 'frontend success','traj time', 'traj jerk', 'compute time(ms)', 'compute time front(ms)', 'compute time back(ms)', 'tracking error(m)', 'effort(rpm)', 'collision rate'])
        #     for i in range(self.num_planners):
        #         writer.writerow([success_rate_avg[i], success_front_rate[i], traj_time_avg[i], traj_jerk_avg[i], poly_compute_time_avg[i], compute_time_front_avg[i], compute_time_back_avg[i], tracking_error_avg[i], effort_avg[i], collision_rate_avg[i]])
           

def subscriber():
    rospy.init_node('evaluate_traj')
    Evaluater()

    # spin() simply keeps python from exiting until this node is stopped
    # rospy.spin()


if __name__ == '__main__':
    try:
        subscriber()
    except rospy.ROSInterruptException:
        pass
