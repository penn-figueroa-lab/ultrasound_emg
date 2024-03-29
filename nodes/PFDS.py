#!/bin/python3
from matplotlib.patches import Arc
import numpy as np
import matplotlib.pyplot as plt
import copy
from pfilter import ParticleFilter, t_noise
from plot_util import generate_surface, plot_sample, SimPlot, RealTimePlot
from functools import partial
# from objects.Desk import Desk2D
from scipy.spatial.transform import Rotation as R
import math
import time
from helper_fun import *
from colorama import Fore
# from game_theory_controller import NegotiationController
import sys, os
import time
#Following imports for CUROBO

# Third Party
import torch

# CuRobo
from curobo.geom.types import WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import RobotConfig
from curobo.util_file import get_robot_configs_path, get_world_configs_path, join_path, load_yaml
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig

torch.backends.cudnn.benchmark = True

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True




class PFDS:
    def __init__(self, **kwargs):
        
        self.dt = kwargs['dt']
        self.n_particles = kwargs['n_particles']
        self.verbose = kwargs['verbose']
        self.desk_object = kwargs['object']

        self.controller_type = 0 # 0: original, 1: negotiation controller, 2: copy force controller

        self.save_in_folder = kwargs['save_in_folder']
        sim_bool = kwargs['sim_bool']
        if self.save_in_folder:
            self.sim_plot = SimPlot(self.dt)



        columns = ["x", "y", "theta", "A00", "A01", "A10", "A11"]

        tensor_args = TensorDeviceType()

        # config_file = load_yaml(join_path(get_robot_configs_path(), "iiwa.yml"))
        # urdf_file = config_file["robot_cfg"]["kinematics"]["urdf_path"]  # Send global path starting with "/"
        # base_link = config_file["robot_cfg"]["kinematics"]["base_link"]
        # ee_link = config_file["robot_cfg"]["kinematics"]["ee_link"]
        robot_cfg = RobotConfig.from_dict(
            load_yaml(join_path(get_robot_configs_path(), "iiwa14_pot.yml"))["robot_cfg"]
        )

        ik_config = IKSolverConfig.load_from_robot_config(
            robot_cfg,
            None,
            rotation_threshold=0.05,
            position_threshold=0.005,
            num_seeds=20,
            self_collision_check=False, # true about 10 hz, false about 20 hz
            self_collision_opt=False,
            tensor_args=tensor_args,
            use_cuda_graph=False,
        )
        self.ik_solver = IKSolver(ik_config)

        self.pf = ParticleFilter(
            prior_fn=partial(prior_uniform_with_A, ik_solver = self.ik_solver, theta_limit=np.pi/2, manipuated_obj = self.desk_object),
            observe_fn=partial(
                traj_to_particle_with_obs, obstacles=np.array([0.6, 0.6])
            ),
            n_particles=self.n_particles,
            dynamics_fn=ForceVelocity,
            # noise_fn=partial(simple_xy_noise_A_nonneg, x_noise=0.01, y_noise=0.01, theta_noise=0.03, x_lim = 1.0, y_lim = 1.0, theta_limit=np.pi/2 ),
            noise_fn=partial(ellipsoid_xy_noise_A_nonneg, x_noise=0.01, y_noise=0.01, theta_noise=0.03, x_lim = 1.0, y_lim = 1.0, theta_limit=np.pi/2),
            weight_fn=partial(multi_part_weighing, manipuated_obj = self.desk_object, k_exp = 0.33, ik_solver = self.ik_solver),
            resample_proportion = 0.0,
            n_eff_threshold=1.0,# not sure if lower this helps, in a lot of online resource, it says lower resampling freuquency helps
            column_names=columns,
        )


        self.xyt_history = None



        self.F_fdfw = np.zeros((3, 1))


    def update_config(self, config, M=np.ones(3)):
        self.controller_type = config['controller_type']
        # self.pf.noise_fn = partial(simple_xy_noise_A_nonneg, x_noise= config['noise_x'], y_noise= config['noise_y'], theta_noise= config['noise_theta'], 
        #                            x_lim = config['x_limit'], y_lim = config['y_limit'], theta_limit= config['theta_limit'])
        self.pf.noise_fn = partial(ellipsoid_xy_noise_A_nonneg, x_noise= config['noise_x'], y_noise= config['noise_y'], theta_noise= config['noise_theta'], 
                                   x_lim = config['x_limit'], y_lim = config['y_limit'], theta_limit= config['theta_limit'])
        self.pf.resample_proportion = config['resample_proportion']
        self.pf.weight_fn = partial(multi_part_weighing, manipuated_obj = self.desk_object,  k_exp = config['k_exp'], L1 = config['L1_gain'], L2 = config['L2_gain'], weight_velocity = config['weight_velocity'], weight_A = config['weight_A'], ik_solver = self.ik_solver)
        self.pf.prior_fn  = partial(prior_uniform_with_A, ik_solver = self.ik_solver, theta_limit=config['theta_limit'], manipuated_obj = self.desk_object)
    
    def run(self, F_human, obj_pos, obj_vel, ellipsoid_matrix):
        return_attractor = np.zeros(3)
        force_sensor_reading = np.array([0, 0])

        xyt = np.array(obj_pos).reshape(-1, 1)

        if len(self.xyt_history) > 3:
            pass #start running things
        else:
            # dont do anything
            v1 = np.zeros((3, 1))
            print("not enough history, returning zero")
            return v1, return_attractor, np.eye(3) * -1, 1.0
        
        # input to particle filter the actual observation
        # y = travel direction(2), travel velocity(1), ft sensor induced velocity(2), yaw(1)
        # xy_diff = self.xyt_history[-1, 0:2] - self.xyt_history[-2, 0:2]
        # # if xy_diff_mag < 1e-3:
        # #     v1 = np.zeros((3, 1))
        # # else:

        # travel_dir = xy_diff / (np.linalg.norm(xy_diff)+0.001)  # 1 x 2
        # travel_vel = np.linalg.norm(xy_diff) / self.dt
        # yaw = 0.5 * obj_pos[-1] + 0.5 * np.arctan2(
        #     travel_dir[1], travel_dir[0]
        # )  # + 0.5*np.cos(5*t) #this is human yaw, may be different from travel direction, add random noise
        # print(travel_dir)
        # concat observation
        y_ob = np.zeros(6)
        y_ob[0:2] = 0#travel_dir.reshape(-1)
        y_ob[2] = 0#travel_vel
        y_ob[3:5] = np.zeros(2)
        y_ob[5] = 0#yaw
        self.pf.update(y_ob, xy_history=self.xyt_history, F = F_human, ellipsoid_matrix)

        # provide assianance
        A_est, attractor = self.pf.aggregate_fun()
        assert (A_est[0,0] < 0)
        assert (A_est[1,1] < 0)
        assert (A_est[2,2] < 0)
        if self.verbose:
            print("attractor = ", attractor.reshape(-1))

        v1 = A_est @ (xyt - attractor)
        return_attractor = attractor.reshape(-1)
        # p_predict = xyt + v1 * self.dt
        # v2 = A_est @ (p_predict - attractor)
        # if self.verbose:
        #     print("attractor dist = ",np.linalg.norm(xyt[0:2] - attractor[0:2]))
        # if np.linalg.norm(xyt[0:2] - attractor[0:2]) < 1.0:
        #     a_predict = np.zeros((3,1))
        #     if self.verbose:
        #         print(Fore.RED + 'Close to attractor, not feedforwarding!')
        # else:
        #     a_predict = (v2 - v1) / self.dt
        # if self.verbose:
        #     print("v1 = ", v1.reshape(-1), "v2 = ", v2.reshape(-1))
        #     print('attractor', attractor)
        # #print("v1 = ", v1.reshape(-1), "v2 = ", v2.reshape(-1))

        # #print("v desk = ", v_desk.reshape(-1))

        # F_fdfw = obj_m * a_predict  # feed forward ctrl
        # # t_expected = 0.5 #expected time to reach desired velocity
        # F_fdbk = Kv * (v1 - obj_vel)  # / t_expected #feedback ctrl
        # #print("F_fdfw = ", F_fdfw.reshape(-1), "F_fdbk = ", F_fdbk.reshape(-1))
        # F_assist = F_fdfw + F_fdbk

        return v1, return_attractor, A_est, np.mean(self.pf.particle_alive_time)

