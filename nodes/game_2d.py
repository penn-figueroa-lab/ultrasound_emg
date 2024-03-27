#!/usr/bin/env python
import pygame
import rospy
import numpy as np
from pfilter import ParticleFilter, t_noise
from geometry_msgs.msg import WrenchStamped, Wrench
from std_msgs.msg import Float32
from dynamic_reconfigure.server import Server
from intent_capability_hri.cfg import IntentCapabilityHRIConfig

from functools import partial
from helper_fun import *
from objects.Desk import Desk2D
from plot_util import RealTimePlot
from PFDS import PFDS
USE_FT_FILTER = True
USE_JOYSTICK = True # False for force torque sensor
PRINT_DEBUG = False

GAME_THEORY_CONTROLLER = 2

tmp_force_arr = np.zeros((6,1))
tmp_force_filter_arr = np.zeros((6,1))
tank_state = 0.0
tank_delta = 0.0

def tank_state_callback(data):
    global tank_state
    tank_state = data.data
    # print("tank_state", tank_state)
def tank_delta_callback(data):
    global tank_delta
    tank_delta = data.data
    # print("tank_delta", tank_delta)
def force_callback(data):
    global tmp_force_arr
    tmp_force_arr = np.array([data.wrench.force.x, data.wrench.force.y, data.wrench.force.z,\
                                data.wrench.torque.x, data.wrench.torque.y, data.wrench.torque.z]).reshape(-1,1)
def force_filter_callback(data):
    global tmp_force_filter_arr
    tmp_force_filter_arr = np.array([data.force.x, data.force.y, data.force.z,\
                                data.torque.x, data.torque.y, data.torque.z]).reshape(-1,1)
if __name__ == '__main__':
    rospy.init_node('game_2d')
    freq=10
    r = rospy.Rate(freq)
    # srv = Server(IntentCapabilityHRIConfig, move.config_callback)

    pygame.init()
    FPS = 100 # dt = 0.02
    dt = 1 / FPS
    if USE_JOYSTICK:
        joystick = pygame.joystick.Joystick(0)
        joystick.init()
    else:
        force_sub = rospy.Subscriber("/bus0/ft_sensor0/ft_sensor_readings/wrench", WrenchStamped, force_callback)
        force_filter_sub = rospy.Subscriber("/admittance/human_wrench", Wrench, force_filter_callback)
        tank_state_sub = rospy.Subscriber("/admittance/tank_state", Float32, tank_state_callback)
        tank_delta_sub = rospy.Subscriber("/tank_delta", Float32, tank_delta_callback)
    # Run until the user asks to quit
    running = True
    clock = pygame.time.Clock()
    object_init_pos = [0.6-0.15, 0.0, 0.0]
    object_init_vel = [1e-3, 1e-3, 0.0]

    A_prior = np.array([[-0.9, 0, 0], [0, -0.9, 0], [0, 0, -0.9]])
    desk = Desk2D(np.array(object_init_pos), np.array(object_init_pos), dt)
    pfds = PFDS(n_particles=400, dt=dt, object_init_pos= object_init_pos, \
                object_init_vel= object_init_vel, A_prior=A_prior,\
                    object = desk, save_in_folder=False, real_time_vis=True, verbose = False, sim_bool = True)
    rt_plot = RealTimePlot(desk, pfds.pf)
    
    while not rospy.is_shutdown():
        clock.tick(FPS)

        v_desk = desk.get_world_frame_velocity()
        xyt = np.array(desk.pos).reshape(-1, 1)

        # concatenate xyt with v_desk
        xyt_and_v = np.vstack((xyt, v_desk))
        
        # Did the user click the window close button?
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if USE_JOYSTICK:
            horizontal = joystick.get_axis(0)
            vertical = joystick.get_axis(1)
            yaw = joystick.get_axis(2)
            horizontal *= 35
            vertical *= -35
            yaw *= -35
        else:
            horizontal = 0.7
            vertical = -1.0
            yaw = 0.0
            if USE_FT_FILTER:
                horizontal = tmp_force_filter_arr[0,0]
                vertical = tmp_force_filter_arr[1,0]
                yaw = tmp_force_filter_arr[5,0]
            else:
                horizontal = tmp_force_arr[0,0]
                vertical = tmp_force_arr[1,0]
                yaw = tmp_force_arr[5,0]
        if PRINT_DEBUG:
            print(horizontal, vertical, yaw)

        if abs(horizontal) < 0.01:
            horizontal = 0.0

        if abs(vertical) < 0.01:
            vertical = 0.0
        
        if abs(yaw) < 0.01:
            yaw = 0.0



        F_human = np.array([horizontal, vertical, yaw]).reshape(-1,1)
        # this is the sensed force from the human, not the same as F_human
        # F_human is imagining if all acceration is from human, what would the force be
        # print("sensed", F_human_sensed)
        
        desired_v, attractor, _, _ = pfds.run(F_human, xyt, v_desk)

        if GAME_THEORY_CONTROLLER == 1:
            pass
            # F_assist = np.zeros_like(F_human)
            # human_goal = attractor.reshape(-1)
            # x_negotiate,_,F_assist[0] = self.ctrlx.update_x_negotiate(human_goal[0])
            # y_negotiate,_,F_assist[1] = self.ctrly.update_x_negotiate(human_goal[1])
            # theta_negotiate,_,F_assist[2] = self.ctrltheta.update_x_negotiate(human_goal[2])
            
            # F_assist[0,0] = (x_negotiate[0,0] - self.desk.pos[0]) * 35
            # F_assist[1,0] = (y_negotiate[0,0] - self.desk.pos[1]) * 35
            # F_assist[2,0] = (theta_negotiate[0,0] - self.desk.pos[2]) * 35

        elif GAME_THEORY_CONTROLLER == 0:
            if tank_delta < 0.0:
                F_assist = Kv * (desired_v - v_desk)  # / t_expected #feedback ctrl
            else:
                F_assist = F_human
            #print("F_fdfw = ", F_fdfw.reshape(-1), "F_fdbk = ", F_fdbk.reshape(-1))
            x_negotiate = np.zeros_like(F_human)

        else:
            F_assist = F_human # perfect input from human
            x_negotiate = np.zeros_like(F_human)

            #for testing, get a attractor at 3,3. then use a pid controller to get there
            # attractor_test = np.array([3.0, 3.0, 0.0]).reshape(-1,1)
            # A_test = np.array([[-1.5, 0, 0], [0, -1.0, 0], [0, 0, -0.9]])
            # v1_test = A_test @ (xyt - attractor_test)
            # F_human = Kv * (v1_test - v_desk)
            # F_assist = F_human
        # print("desired_v", desired_v)
        # pfds.run(F_human_sensed)
        desk.move_world_frame(F_assist, F_human)
        goal_and_eig = pred_goal_from_hist(pfds.xyt_history)
        rt_plot.plot_realtime(desk, pfds.pf, F_assist, F_human, attractor, -np.ones((3,3)), x_negotiate, pfds.xyt_history, goal_and_eig, tank_state)

    # Done! Time to quit.
    pygame.quit()