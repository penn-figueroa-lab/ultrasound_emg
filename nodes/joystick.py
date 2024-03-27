#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import rospy
import tf
import pygame
from std_msgs.msg import Header
from geometry_msgs.msg import Wrench, WrenchStamped, Vector3, Point
from gazebo_msgs.srv import ApplyBodyWrench

import tf2_ros
import tf.transformations as tf_transform
from geometry_msgs.msg import TransformStamped
from pynput import keyboard

USE_JOYSTICK = False

class Joystick:
    def __init__(self, dt) -> None:

        self.dt = dt
        pygame.init()
        if USE_JOYSTICK:
            pygame.joystick.init()
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()
            print("joystick: {} (axes: {}, buttons: {}, hats: {})".format(
            self.joystick.get_name(), self.joystick.get_numaxes(), self.joystick.get_numbuttons(), self.joystick.get_numhats()))
        else:
            print("Using Keyboard Input")
            self.keyboard_lister = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
            self.keyboard_lister.start()
            self.horizontal = 0.0
            self.vertical = 0.0
            self.yaw = 0.0
        self.clock = pygame.time.Clock()

        self.js_pub = rospy.Publisher("/robotiq_force_torque_wrench_filtered", WrenchStamped, queue_size=10)

        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)
        end_time = 0
        while not end_time: # ensure that the clock is not zero
            end_time = rospy.Time.now()
        end_time = rospy.Time.now() + rospy.Duration(5)  
        while rospy.Time.now() < end_time:
            try:
                trans = self.tfBuffer.lookup_transform('world', 'iiwa_link_ee', rospy.Time(0))
                break  # If transform is found, exit the loop
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                pass
        if rospy.Time.now() >= end_time:
            print('Lookup transform failed')
            exit()
    def on_press(self, key):
        try:
            if key == keyboard.Key.left:
                self.horizontal = -1.0
            elif key == keyboard.Key.right:
                self.horizontal = 1.0

            if key == keyboard.Key.up:
                self.vertical = 1.0
            elif key == keyboard.Key.down:
                self.vertical = -1.0

            if key.char == 'a':
                self.yaw = -1.0
            elif key.char == 'd':
                self.yaw = 1.0
        except AttributeError:
            pass

    def on_release(self, key):
        try:
            if key == keyboard.Key.left or key == keyboard.Key.right:
                self.horizontal = 0.0
            if key == keyboard.Key.up or key == keyboard.Key.down:
                self.vertical = 0.0
            if key.char == 'a' or key.char == 'd':
                self.yaw = 0.0
        except AttributeError:
            pass
    def moving(self):
        if USE_JOYSTICK:
            pygame.event.pump()
            horizontal = self.joystick.get_axis(0)
            vertical = self.joystick.get_axis(1)
            yaw = self.joystick.get_axis(2)
        else:
            horizontal = self.horizontal
            vertical = self.vertical
            yaw = self.yaw

        if abs(horizontal) < 0.01:
            horizontal = 0.0

        if abs(vertical) < 0.01:
            vertical = 0.0
        
        if abs(yaw) < 0.01:
            yaw = 0.0

        horizontal *= 30
        vertical *= 30
        yaw *= -10


        print(horizontal, vertical, yaw)
        wrench = WrenchStamped()
        h = Header()
        h.stamp = rospy.Time.now()
        h.frame_id = 'robotiq_force_torque_frame_id'
        wrench.header = h
        wrench.wrench.force.x = horizontal
        wrench.wrench.force.y = 0.0
        wrench.wrench.force.z = vertical

        wrench.wrench.torque.x = yaw

        self.js_pub.publish(wrench)
        
        trans = self.tfBuffer.lookup_transform('world', 'iiwa_link_ee', rospy.Time(0))

        rospy.logerr("This has bug, plz fix first")
        exit()
        rotation_quat = [trans.transform.rotation.x,
                             trans.transform.rotation.y,
                             trans.transform.rotation.z,
                             trans.transform.rotation.w]
                             
        rotation_matrix = tf_transform.quaternion_matrix(rotation_quat)
        wrech_world_frame = np.array([0.0,horizontal, vertical]).reshape(-1,1)
        wrech_ee_frame = (rotation_matrix[0:3,0:3] @ wrech_world_frame).flatten()
        torque_ee_frame =  rotation_matrix[0:3,0:3] @ np.array([0.0, 0.0, yaw]).reshape(-1,1)
        
        bodywrench = Wrench()
        bodywrench.force.x = wrech_ee_frame[0]
        bodywrench.force.y = wrech_ee_frame[1]
        bodywrench.force.z = wrech_ee_frame[2]
        bodywrench.torque.x = torque_ee_frame[0]
        bodywrench.torque.y = torque_ee_frame[1]
        bodywrench.torque.z = torque_ee_frame[2]

        rospy.wait_for_service('/gazebo/apply_body_wrench')
        try:
            # print('applying wrench')
            apply_wrench = rospy.ServiceProxy('/gazebo/apply_body_wrench', ApplyBodyWrench)
            resp1 = apply_wrench('iiwa::iiwa_link_7', 'world', Point(0.0, 0.0, 0.0),bodywrench, rospy.Time().now(), rospy.Duration(0.01))
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e) 


def main():
    rospy.init_node('joystick')
    freq=50
    r = rospy.Rate(freq)
    js = Joystick(1/freq)

    while not rospy.is_shutdown():
        js.moving()
        r.sleep()



#print("Avg F_human / 2 over time = ", F_human_sum / 2 / (dt * max_idx))

if __name__ == '__main__':
    rospy.logerr("This is deprecated, plz use human_pd_controller.py instead")
    exit()
    try:
        main()
    except rospy.ROSInterruptException:
        pass