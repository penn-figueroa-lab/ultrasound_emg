#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import rospy
import tf
import pygame
from std_msgs.msg import Header, Float32
from geometry_msgs.msg import Wrench, Vector3, Point
from gazebo_msgs.srv import ApplyBodyWrench, SetModelConfiguration
from geometry_msgs.msg import WrenchStamped, Pose, Twist, TwistStamped
from tf.transformations import euler_from_quaternion, quaternion_from_euler, quaternion_matrix
from visualization_msgs.msg import Marker
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty

import tf2_ros
import tf.transformations as tf_transform
from geometry_msgs.msg import TransformStamped
import random
import csv
import time
import copy

import actionlib
from kr_planning_msgs.msg import PlanTwoPoseHumanRobotAction, PlanTwoPoseHumanRobotGoal, PlanTwoPoseHumanRobotResult, PlanTwoPoseHumanRobotFeedback

    # the main job of this node is to 
    # 1. Use joystick as a test
    # 2. Compute force based on a pd controller with saturation with a desired goal
    # 3. Publish the force to the robotiq force torque sensor topic
    # 4. Receive feedback terms and write them down to a csv file

USE_JOYSTICK = False 
REAL_EXP = not USE_JOYSTICK

APPLICATION_LINK_NAME = "iiwa_link_ee"
if APPLICATION_LINK_NAME == "iiwa_link_human":
    use_extended_xyz = True
else:
    use_extended_xyz = False

def cleanup():
    if USE_JOYSTICK:
        print("Closing Joystick...")
        pygame.joystick.quit()
        pygame.quit()
    print("Node shutdown complete.")

def compute_force(obj_pos, obj_vel, goal_pos):
    # 2.1 Compute the force based on the pd controller with saturation
    pos_err = goal_pos - obj_pos
    vel_err = 0.0 - obj_vel
    force_vec = np.array([30,30, 2]) * pos_err + np.array([30,30,2]) * vel_err
    force_vec[0] = np.clip(force_vec[0], -30, 30)
    force_vec[1] = np.clip(force_vec[1], -30, 30)
    force_vec[2] = np.clip(force_vec[2], -4, 4)
    #
    # force_vec[0] = -force_vec[0]
    # force_vec[1] = -force_vec[1]
    # rospy.loginfo_throttle(0.8, "pos_err = {}, vel_err = {}, force_vec = {}".format(pos_err, vel_err, force_vec))
    return force_vec

class Joystick:
    def __init__(self, dt) -> None:
        self._feedback = PlanTwoPoseHumanRobotFeedback()
        self._result = PlanTwoPoseHumanRobotResult()
        self._action_name ="/plan_local_trajectory"
        self._as = actionlib.SimpleActionServer(self._action_name, PlanTwoPoseHumanRobotAction, execute_cb=self.action_execute_cb, auto_start = False)
        self._as.start()
        self.set_config = rospy.ServiceProxy('/gazebo/set_model_configuration', SetModelConfiguration)
        self.reset_move = rospy.ServiceProxy('/reset_move', Empty)
        if not REAL_EXP:
            rospy.loginfo("Waiting for service /gazebo/set_model_configuration")
            self.set_config.wait_for_service()
        self.dt = dt
        self.counter = 10
        self.force_arr = np.zeros(6)
        self.object_pos = np.zeros(7)
        self.ee_pos = np.array([0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        self.ee_vel = np.zeros(6)
        self.obj_pos_3 = np.zeros(3)
        self.obj_vel_3 = np.zeros(3)

        self.tank_state = 0.0

        self.force_received_arr = np.zeros(6)
        self.force_human_received_arr = np.zeros(6) 

        self.cur_pose = Pose()
        self.human_wrench_world_frame = Wrench()
        self.robot_wrench_world_frame = Wrench()

        if USE_JOYSTICK:
            pygame.init()
            pygame.joystick.init()
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()
            print("joystick: {} (axes: {}, buttons: {}, hats: {})".format(
            self.joystick.get_name(), self.joystick.get_numaxes(), self.joystick.get_numbuttons(), self.joystick.get_numhats()))
            self.clock = pygame.time.Clock()
        else:
            print("Using PD controller")
            self.horizontal = 0.0
            self.vertical = 0.0
            self.yaw = 0.0
            self.clock = None
        if REAL_EXP:
            self.force_sub = rospy.Subscriber("/robotiq_force_torque_wrench", WrenchStamped, self.force_callback, tcp_nodelay=True)
            self.force_human_sub = rospy.Subscriber("/bus0/ft_sensor0/ft_sensor_readings/wrench", WrenchStamped, self.force_human_callback, tcp_nodelay=True)
        else:
            self.js_pub = rospy.Publisher("/robotiq_force_torque_wrench_filtered", WrenchStamped, queue_size=10)
        self.ang_vel_ee_frame_pub = rospy.Publisher("/iiwa/ang_vel_ee_frame", Twist, queue_size=10)
        self.goal_gt_pub = rospy.Publisher("/goal_gt", Marker, queue_size=10)
        self.cur_roll_pub = rospy.Publisher("/ee_roll_ang", Float32, queue_size=10)
        self.end_effector_location_pub = rospy.Publisher("/end_effector_location", Marker, queue_size=10)
        
        self.robot_state_sub = rospy.Subscriber("/iiwa/task_states", Pose, self.robot_callback, tcp_nodelay=True)
        self.robot_vel_sub = rospy.Subscriber("/iiwa/ee_vel_cartimp", Pose, self.robot_vel_callback, tcp_nodelay=True)
        self.robot_angVel_sub = rospy.Subscriber("/iiwa/ee_angvel_cartimp", Pose, self.robot_ang_vel_callback, tcp_nodelay=True)
        self.robot_wrench_sub = rospy.Subscriber("/iiwa/ee_wrench", Wrench, self.robot_wrench_callback, tcp_nodelay=True)
        self.tank_state_sub = rospy.Subscriber("/tank_state", Float32, self.tank_state_callback, tcp_nodelay=True)
        self.tank_state_sub2 = rospy.Subscriber("/human_admittance/tank_state_percentage", Float32, self.tank_state_callback, tcp_nodelay=True)

        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)
        self.tfBuffer2 = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer2)
        # give some time to look up the transform
        end_time = 0
        while not end_time: # ensure that the clock is not zero
            end_time = rospy.Time.now()
        end_time = rospy.Time.now() + rospy.Duration(3.0)
        while rospy.Time.now() < end_time:
            try:
                trans = self.tfBuffer.lookup_transform('world', APPLICATION_LINK_NAME, rospy.Time(0))
                break  # If transform is found, exit the loop
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                rospy.sleep(0.1)
        if rospy.Time.now() >= end_time:
            print('Lookup transform failed')
            exit()
    def tank_state_callback(self, data):
        self.tank_state = data.data
    def force_callback(self, data):
        self.force_received_arr[0] = data.wrench.force.x
        self.force_received_arr[1] = data.wrench.force.y
        self.force_received_arr[2] = data.wrench.force.z
        self.force_received_arr[3] = data.wrench.torque.x
        self.force_received_arr[4] = data.wrench.torque.y
        self.force_received_arr[5] = data.wrench.torque.z
        # rospy.loginfo_throttle(0.8, "force_received_arr = {}".format(np.linalg.norm(self.force_received_arr)))
    def force_human_callback(self, data):
        self.force_human_received_arr[0] = data.wrench.force.x
        self.force_human_received_arr[1] = data.wrench.force.y
        self.force_human_received_arr[2] = data.wrench.force.z
        self.force_human_received_arr[3] = data.wrench.torque.x
        self.force_human_received_arr[4] = data.wrench.torque.y
        self.force_human_received_arr[5] = data.wrench.torque.z
    def robot_wrench_callback(self, data):
        self.robot_wrench_world_frame = data

    def reset_robot(self):
        self.set_config('iiwa', '', ['iiwa_joint_1','iiwa_joint_2','iiwa_joint_3','iiwa_joint_4','iiwa_joint_5','iiwa_joint_6','iiwa_joint_7'], 
               ## iiwa 7 null pose
            #    [0.044752691045324394, 0.6951627023357917, -0.01416978801753847, -1.0922311725109015, -0.0050429618456282, 1.1717338014778385, -0.015026300603056137],
               ## iiwa 14 null pose
               [0.        ,  1.50899694,  0.        , -1.3962634 ,  0.        , -1.35191731,  0.0]
               )
    def action_execute_cb(self, goal):
        # helper variables
        # append the seeds for the fibonacci sequence
        # self._feedback.sequence = []
        # self._feedback.sequence.append(0)
        # self._feedback.sequence.append(1)
        if not REAL_EXP:
            self.reset_robot()
            rospy.sleep(0.1)
            try:
                self.reset_move()
            except rospy.ServiceException as e:
                print("Service call failed: %s"%e)
            rospy.sleep(0.5)
        else:
            pass    
        rospy.loginfo('%s: Executing, creating plan from %s to %s' % (self._action_name, goal.p_init.position.y, goal.p_final.position.y))    
        self._result = PlanTwoPoseHumanRobotResult()
        self._result.p_final = goal.p_final
        # start executing the action
        # this step is not necessary, the sequence is computed at 1 Hz for demonstration purposes
        goal_pos = np.array([goal.p_final.position.x, goal.p_final.position.y, goal.p_final.position.z])
        start_time = rospy.Time.now()
        end_time = rospy.Time.now() + rospy.Duration(19.0)
        # while time < 8 seconds or the goal is not reached
        self._result.success = False
        while rospy.Time.now() < end_time:
            self._result.time_vec.append((rospy.Time.now()-start_time).to_sec())
            self._result.pose.append(self.cur_pose)
            cur_twist_world = Twist()
            cur_twist_world.linear.x = self.ee_vel[0]
            cur_twist_world.linear.y = self.ee_vel[1]
            cur_twist_world.linear.z = self.ee_vel[2]
            cur_twist_world.angular.x = self.ee_vel[3]
            cur_twist_world.angular.y = self.ee_vel[4]
            cur_twist_world.angular.z = self.ee_vel[5]
            append_wrench = Wrench()
            append_wrench_human = Wrench()
            self._result.tank_level.append(self.tank_state)
            if REAL_EXP:
                append_wrench.force.x = self.force_received_arr[0]
                append_wrench.force.y = self.force_received_arr[1]
                append_wrench.force.z = self.force_received_arr[2]
                append_wrench.torque.x = self.force_received_arr[3]
                append_wrench.torque.y = self.force_received_arr[4]
                append_wrench.torque.z = self.force_received_arr[5]
                append_wrench_human.force.x = self.force_human_received_arr[0]
                append_wrench_human.force.y = self.force_human_received_arr[1]
                append_wrench_human.force.z = self.force_human_received_arr[2]
                append_wrench_human.torque.x = self.force_human_received_arr[3]
                append_wrench_human.torque.y = self.force_human_received_arr[4]
                append_wrench_human.torque.z = self.force_human_received_arr[5]
                rospy.sleep(0.1) # 10 hz
            else:
                append_wrench.force.x = self.human_wrench_world_frame.force.x
                append_wrench.force.y = self.human_wrench_world_frame.force.y
                append_wrench.force.z = self.human_wrench_world_frame.force.z
                append_wrench.torque.x = self.human_wrench_world_frame.torque.x
                append_wrench.torque.y = self.human_wrench_world_frame.torque.y
                append_wrench.torque.z = self.human_wrench_world_frame.torque.z
            self._result.twist.append(cur_twist_world)
            self._result.robot_sensed.append(append_wrench)
            self._result.human_sensed.append(append_wrench_human)
            self._result.robot_effort_world.append(copy.deepcopy(self.robot_wrench_world_frame))

            self.moving(goal_pos)
            if np.linalg.norm(self.obj_pos_3 - goal_pos) < 0.15 and np.linalg.norm(self.obj_vel_3) < 0.05:
                self._result.success = True
                break
        self._result.total_time = (end_time - start_time).to_sec()
        rospy.loginfo('Action Done' )
        self._as.set_succeeded(self._result)

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
        self.cur_pose = copy.deepcopy(data)
        
        #self.get_optitrack_pos()
        self.counter -= 1
        if self.counter < 0:
            self.counter = 10
            (ang1, ang2, ang3) = euler_from_quaternion(self.object_pos[3:], axes='ryxz')
            rot_mat_ee = quaternion_matrix(self.object_pos[3:])[0:3,0:3]
            rot_v_ee_frame = rot_mat_ee.transpose() @ self.ee_vel[3:].reshape(-1,1)
            ang_vel_ee_frame = Twist()
            ang_vel_ee_frame.angular.x = rot_v_ee_frame.flatten()[0]
            ang_vel_ee_frame.angular.y = rot_v_ee_frame.flatten()[1]
            ang_vel_ee_frame.angular.z = rot_v_ee_frame.flatten()[2]
            self.ang_vel_ee_frame_pub.publish(ang_vel_ee_frame)
            #print("orientation:", self.object_pos[3:])
            self.obj_pos_3 = np.array([self.object_pos[0], self.object_pos[1], ang3])
            self.obj_vel_3 = np.array([self.ee_vel[0], self.ee_vel[1], rot_v_ee_frame.flatten()[2]])
            ang3_msg = Float32()
            ang3_msg.data = ang3
            self.cur_roll_pub.publish(ang3_msg)
            end_eff_pub_msg = Marker()
            end_eff_pub_msg.header.frame_id = "world"
            end_eff_pub_msg.header.stamp = rospy.Time.now()
            end_eff_pub_msg.type = end_eff_pub_msg.SPHERE
            end_eff_pub_msg.action = end_eff_pub_msg.ADD
            end_eff_pub_msg.pose.position.x = self.ee_pos[0]
            end_eff_pub_msg.pose.position.y = self.ee_pos[1]
            end_eff_pub_msg.pose.position.z = 0.0
            end_eff_pub_msg.pose.orientation.x = 0.0
            end_eff_pub_msg.pose.orientation.y = 0.0
            end_eff_pub_msg.pose.orientation.z = 0.0
            end_eff_pub_msg.pose.orientation.w = 1.0
            end_eff_pub_msg.scale.x = 0.05
            end_eff_pub_msg.scale.y = 0.05
            end_eff_pub_msg.scale.z = 0.05
            end_eff_pub_msg.color.a = 1.0
            end_eff_pub_msg.color.r = 1.0
            end_eff_pub_msg.color.g = 0.0
            end_eff_pub_msg.color.b = 0.0
            self.end_effector_location_pub.publish(end_eff_pub_msg)
    def moving(self, goal_pos):
        if USE_JOYSTICK:
            pygame.event.pump()
            horizontal = self.joystick.get_axis(0)
            vertical = self.joystick.get_axis(1)
            yaw = self.joystick.get_axis(2)
        else:
            # 2.2 Compute the force based on the pd controller with saturation
            force_vec = compute_force(self.obj_pos_3, self.obj_vel_3, goal_pos)
            self.vertical = force_vec[0]
            self.horizontal = force_vec[1]
            self.yaw = force_vec[2]
            horizontal = self.horizontal
            vertical = self.vertical
            yaw = self.yaw
        # rospy.loginfo_throttle(0.8, "horizontal = {}, vertical = {}, yaw = {}".format(horizontal, vertical, yaw))

        if abs(horizontal) < 0.01:
            horizontal = 0.0

        if abs(vertical) < 0.01:
            vertical = 0.0
        
        if abs(yaw) < 0.01:
            yaw = 0.0

        if USE_JOYSTICK:
            horizontal *= 60
            vertical *= 60
            yaw *= -20



        
        
        trans = self.tfBuffer.lookup_transform('world', APPLICATION_LINK_NAME, rospy.Time(0))
               
        rotation_quat = [trans.transform.rotation.x,
                             trans.transform.rotation.y,
                             trans.transform.rotation.z,
                             trans.transform.rotation.w]

        rotation_matrix = tf_transform.quaternion_matrix(rotation_quat)
        force_world_frame = np.array([vertical,horizontal, 0.0]).reshape(-1,1)
        force_ee_frame = (rotation_matrix[0:3,0:3].transpose() @ force_world_frame).flatten()
        torque_world_frame =  rotation_matrix[0:3,0:3] @ np.array([0.0, 0.0, yaw]).reshape(-1,1)
        xyz_human_robot =  rotation_matrix[0:3,0:3] @ np.array([0.0, 0.0, 0.4]).reshape(-1,1)
        trans = self.tfBuffer2.lookup_transform('world', 'robotiq_force_torque_frame_id', rospy.Time(0))
        rotation_quat = [trans.transform.rotation.x,
                             trans.transform.rotation.y,
                             trans.transform.rotation.z,
                             trans.transform.rotation.w]
        rotation_ft = tf_transform.quaternion_matrix(rotation_quat)
        load_world = np.array([0.0, 0.0 , (-4.444)*9.81])
        r_load_ft = np.array([0,-0.04, 0.155]).reshape(-1,1) #compensate for clamps
        r_load_world = (rotation_ft[:3,:3] @ r_load_ft).flatten()
        wrench_grav = Wrench()
        wrench_grav.force.x = load_world[0]
        wrench_grav.force.y = load_world[1]
        wrench_grav.force.z = load_world[2]

        # exit()
        
        if not REAL_EXP:

            wrench = WrenchStamped()
            h = Header()
            h.stamp = rospy.Time.now()
            h.frame_id = APPLICATION_LINK_NAME
            wrench.header = h
            wrench.wrench.force.x = force_ee_frame[0]
            wrench.wrench.force.y = force_ee_frame[1]
            wrench.wrench.force.z = force_ee_frame[2]
            wrench.wrench.torque.z = yaw
            # wrench.wrench.force.x = wrech_world_frame[0]
            # wrench.wrench.force.y = wrech_world_frame[1]
            # wrench.wrench.force.z = wrech_world_frame[2]
            self.js_pub.publish(wrench) #viz auto adjust frame


            bodywrench = Wrench()
            # bodywrench.force.x = wrech_ee_frame[0]
            # bodywrench.force.y = wrech_ee_frame[1]
            # bodywrench.force.z = wrech_ee_frame[2]
            # bodywrench.torque.x = yaw
            # we instead apply world frame wrench
            bodywrench.force.x = vertical
            bodywrench.force.y = horizontal
            bodywrench.torque.x = torque_world_frame[0]
            bodywrench.torque.y = torque_world_frame[1]
            bodywrench.torque.z = torque_world_frame[2]
            self.human_wrench_world_frame = bodywrench

            rospy.wait_for_service('/gazebo/apply_body_wrench')
            rospy.loginfo_throttle(0.5, "Here")
            if not use_extended_xyz:
                #set each element of xyz_human_robot to 0
                xyz_human_robot = np.array([0.0, 0.0, 0.0])
            try:
                # print('applying wrench')
                apply_wrench = rospy.ServiceProxy('/gazebo/apply_body_wrench', ApplyBodyWrench)
                resp1 = apply_wrench("iiwa::iiwa_link_7", 'world', Point(xyz_human_robot[0],xyz_human_robot[1],xyz_human_robot[2]) , bodywrench, rospy.Time().now(), rospy.Duration(0.1))
                resp2 = apply_wrench("iiwa::iiwa_link_7", 'world', Point(r_load_world[0] ,r_load_world[1],r_load_world[2]) , wrench_grav, rospy.Time().now(), rospy.Duration(0.1))
            except rospy.ServiceException as e:
                print("Service call failed: %s"%e) 
        else:
            pass
            # rospy.loginfo_throttle(0.5, "force_received_arr = {}".format(np.linalg.norm(self.force_received_arr)))
            


def main():
    rospy.init_node('joystick')
    rospy.on_shutdown(cleanup)
    freq = 10
    r = rospy.Rate(freq)
    js = Joystick(1/freq)

    # 1 completed

    # 2.1 Fix the random seed and give a new goal when the old goal is reached
    # random.seed(0)
    # goals are 3d vectors with x,y, roll, generate 10 goals
    # goals = np.zeros((10,3))
    # for i in range(10):
    #     goals[i,0] = random.uniform(0.2, 0.7) # x range
    #     goals[i,1] = random.uniform(-0.4,0.4)
    #     goals[i,2] = random.uniform(-1.5,1.5)
    rospy.sleep(1.0)
    while not rospy.is_shutdown():
        # find the goal based on time, every goal lasts for 8 seconds.
        # t = rospy.Time.now().to_sec()
        # goal_idx = int(t // 8)
        # goal_idx = goal_idx % 10
        # goal_pos = goals[goal_idx]
        # goal_pub = Marker()
        # goal_pub.header.frame_id = "world"
        # goal_pub.header.stamp = rospy.Time.now()
        # goal_pub.type = goal_pub.SPHERE
        # goal_pub.action = goal_pub.ADD
        # goal_pub.pose.position.x = goal_pos[0]
        # goal_pub.pose.position.y = goal_pos[1]
        # goal_pub.pose.position.z = 0.0
        # goal_pub.pose.orientation.x = 0.0
        # goal_pub.pose.orientation.y = 0.0
        # goal_pub.pose.orientation.z = 0.0
        # goal_pub.pose.orientation.w = 1.0
        # goal_pub.scale.x = 0.05
        # goal_pub.scale.y = 0.05
        # goal_pub.scale.z = 0.05
        # goal_pub.color.a = 1.0
        # goal_pub.color.r = 1.0
        # goal_pub.color.g = 0.0
        # goal_pub.color.b = 0.0

        # js.goal_gt_pub.publish(goal_pub)
        # goal_pos = np.array([0.2, 0.0, 0.0])
        # rospy.loginfo_throttle(0.8, "goal_pos = {}".format(goal_pos))
        if USE_JOYSTICK:
            js.moving(np.array([0.5, 0.0, 0.0]))
        r.sleep()



#print("Avg F_human / 2 over time = ", F_human_sum / 2 / (dt * max_idx))

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
    finally:
        cleanup()