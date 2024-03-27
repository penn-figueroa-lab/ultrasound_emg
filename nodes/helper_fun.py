import numpy as np
from pfilter import ParticleFilter, t_noise
from curobo.types.math import Pose
import torch
import pytorch3d.transforms as torch3d
from tf.transformations import euler_from_quaternion, quaternion_from_euler, quaternion_matrix
import time
import ros
import rospy

Kv = 30  # linear control gain
Kw = 100  # angular control gain
A_UPPER_BOUND = -0.4
A_LOWER_BOUND = -0.6
A_UPPER_BOUND_ROLL = -0.2
A_LOWER_BOUND_ROLL = -0.4    
A_AVG = 0.5 * (A_UPPER_BOUND + A_LOWER_BOUND)
A_AVG_ROLL = 0.5 * (A_UPPER_BOUND_ROLL + A_LOWER_BOUND_ROLL)

DESIRED_ROLL = np.pi
DESIRED_PITCH = np.pi/2

def ForceVelocity(x, xy_history, F):
    ## go slightly towards xy_history
    # curr_state = xy_history[-1]
    # #### particles go back to current state
    # saturate_velocity = 0.005
    # update_vec_idv = np.zeros_like(x)
    # update_vec_idv[:,0:3] = (- x[:,0:3] + curr_state[0:3] ) * saturate_velocity
    # #saturation
    # update_vec_idv[:,0:3] = np.clip(update_vec_idv[:,0:3], -saturate_velocity, saturate_velocity)
    # update_vec_idv[:,2] = np.clip(update_vec_idv[:,2], -0.002, 0.002)
    update_vec_idv = np.zeros_like(x)
    update_vec_idv[:,[3,7,11]] = -0.1*(x[:,[3,7,11]] - np.mean(x[:,[3,7,11]], axis=0))
    ####NOT USING UPDATE TOWRAD CURRENT STATE####
    
    ##### particles move according to force, this is not a great idea, not very realistic
    # update_vec = np.zeros(12)
    # update_vec[0:3] = F.flatten()
    return x# + update_vec_idv #+ update_vec * 0.005


def prior_bimodal_with_A(n, A_prior):
    # x y
    first_half_n = int(n / 2)
    second_half_n = n - first_half_n
    mu = np.array([1, 4])
    cov = np.array([[0.001, 0], [0, 0.001]])
    particles_xy_1 = np.random.multivariate_normal(mu, cov, first_half_n)
    mu = np.array([4, 1])
    cov = np.array([[0.001, 0], [0, 0.001]])
    particles_xy_2 = np.random.multivariate_normal(mu, cov, second_half_n)
    particles_xy = np.concatenate((particles_xy_1, particles_xy_2), axis=0)

    mu_A = A_prior.flatten()
    cov_A = np.array(
        [[0.01, 0, 0, 0], [0, 0.00, 0, 0], [0, 0, 0.00, 0], [0, 0, 0, 0.01]]
    )  # only diagonal
    particles_A = np.random.multivariate_normal(mu_A, cov_A, n)

    particles = np.concatenate((particles_xy, particles_A), axis=1)

    return particles

def curobo_ik_filter(y_hat, ik_solver, manipuated_obj):
    """
    Return ones and zeros for a given set of particles, set to 1 if the particle is valid and 0 if it is not.
    """
    # return np.ones(states.shape[0], dtype=bool) # test
    #: End-effector position stored as x,y,z in meters [b, 3]. End-effector is defined by
    #: :py:attr:`curobo.cuda_robot_model.cuda_robot_generator.CudaRobotGeneratorConfig.ee_link`.
    # ee_position: torch.Tensor

    #: End-effector orientaiton stored as quaternion qw, qx, qy, qz [b,4]. End-effector is defined
    # by :py:attr:`CudaRobotModelConfig.ee_link`.
    # ee_quaternion: torch.Tensor
    # left_coord = np.zeros((y_hat.shape[0], 2))
    # # right_coord = np.zeros((y_hat.shape[0], 2))
    # for i in range(y_hat.shape[0]):
    #     particle_xyt = y_hat[i, :]
    #     left, _ = manipuated_obj.get_left_right_coordinate_state(particle_xyt)
    #     left_coord[i,:] = left
    #     # right_coord[i,:] = right

    
    states = y_hat # np.concatenate((left_coord, y_hat[:, 2:3]), axis=1)


    st_time = time.time()
    x = states[:, 0]
    y = states[:, 1]
    angles = states[:, 2]
    n = states.shape[0]
    ee_position = np.column_stack((x, y, np.zeros_like(x)))
    # turn ee_position into Tensor
    ee_position = torch.from_numpy(ee_position).float().to("cuda")
    # get euler_angles in n x 3
    euler_angles = np.zeros((n, 3))
    euler_angles[:, 0] = np.ones(n) * DESIRED_PITCH
    euler_angles[:, 1] = -np.arctan2(y, x)
    euler_angles[:, 2] = angles
    ee_quaternion = torch3d.matrix_to_quaternion(torch3d.euler_angles_to_matrix(torch.from_numpy(euler_angles).float().cuda(), "YXZ"))
    # send to ik solver as cuda tensor 
    #!!! Consider using quat directly!!!!
    goal = Pose(ee_position, ee_quaternion)
    
    #time ik solver
    result = ik_solver.solve_batch(goal)
    torch.cuda.synchronize()
    rospy.loginfo_throttle(5, "IK count:" + str(torch.count_nonzero(result.success).item()) + "time: " + str( time.time() - st_time) + ' sec')
    return result.success.cpu().numpy().flatten() # return success as 1, fail as 0


def sample_points_on_circle_band(n, inner_radius, outer_radius, min_angle, max_angle, ik_solver, theta_limit, theta_noise = 0.0, obj = None):
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
    # theta_noise_add = np.random.uniform(-theta_noise, theta_noise, n)
    # theta = angles + theta_noise_add
    # saturate theta to THETA_PRIOR_LIMIT
    theta = np.random.uniform(-theta_limit, theta_limit, n)
    states = np.column_stack((x, y, theta))

    bool_ik = np.ones(n, dtype=bool)
    # bool_ik = curobo_ik_filter(states, ik_solver, manipuated_obj=obj)
    states_filtered = states[bool_ik,:]
    if states_filtered.shape[0] < n:
        num_repetition = int(n/states_filtered.shape[0]) + 1
        rospy.logwarn_throttle(5, "not enough ik solutions, only" + str( states_filtered.shape[0]))
        rospy.logwarn_throttle(5, "repeating" + str(num_repetition) + " times")
        states_filtered = np.tile(states_filtered, (num_repetition, 1))
    
    states_filtered = states_filtered[0:n,:]
    return states_filtered

def prior_uniform_with_A(n, ik_solver, theta_limit, manipuated_obj = None):
    # x y
    inner_radius = 0.5
    outer_radius = 1.3
    min_angle = -80
    max_angle = 80

    particles_xy = sample_points_on_circle_band(n, inner_radius, outer_radius, min_angle, max_angle, ik_solver, theta_limit, theta_noise = 0.0, obj = manipuated_obj)
    noise_eig = 0.05
    mu_A = np.array([A_AVG, 0, 0, 0, A_AVG, 0, 0, 0, A_AVG_ROLL])
    cov_A = np.array(
        [
            [noise_eig, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, noise_eig, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, noise_eig],
        ]
    )
    # cov_A = np.array(
    #     [[0.5, 0, 0, 0], [0, 0.00, 0, 0], [0, 0, 0.00, 0], [0, 0, 0, 0.5]]
    # )  # only diagonal
    particles_A = np.random.multivariate_normal(mu_A, cov_A, n)

    particles = np.concatenate((particles_xy, particles_A), axis=1)
    return particles


def prior_unimodal_with_A(n, A_prior):
    # x y
    mu = np.array([4, 2, 1])
    cov = np.array([[0.001, 0, 0], [0, 0.001, 0], [0, 0, 0.001]])
    particles_xy = np.random.multivariate_normal(mu, cov, n)
    mu_A = A_prior.flatten()  # 3x3 -> 9
    cov_A = np.array(
        [
            [0.5, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0.5, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0.5],
        ]
    )

    particles_A = np.random.multivariate_normal(mu_A, cov_A, n)

    particles = np.concatenate((particles_xy, particles_A), axis=1)
    return particles


def ellipse_r(phi, a, b):
    # a is longer axis, b is shorter axis
    theta = np.arctan(np.tan(phi) * a / b)
    e = np.sqrt(1 - b**2 / a**2)
    r = a * np.sqrt(1 - e**2 * np.sin(theta) ** 2)
    return r


def wrap_to_pos_neg_pi(x):
    return (x + np.pi) % (2 * np.pi) - np.pi

def lsqL2(A, y, lamb=0.0):
    U,S,Vt = np.linalg.svd(A, full_matrices=False)
    return Vt.T@((U.T@y)*(S/(S**2+lamb)))

def A_only_est(xy_history):
    #Sequential Estimate
    data_hist_pos = xy_history[:, 0:3]
    data_hist_vel = xy_history[:, 3:6]
    A_est = np.zeros(3)
    for i in range(3):
        data_left = np.diff(data_hist_pos[:,i])
        data_right = np.diff(data_hist_vel[:,i])
        result = np.linalg.lstsq(data_left.reshape(-1,1), data_right, rcond=None)
        A_est[i] = result[0][0]
    
    return A_est
def pred_goal_from_hist(xy_history):
    data_hist_pos = xy_history[:, 0:3]
    data_hist_vel = xy_history[:, 3:6]
 #run least square to get the best A and goal
    A_data_x = np.hstack( (np.ones((data_hist_pos.shape[0], 1)), (data_hist_vel[:,0:1] ))) # n x 2
    A_data_y = np.hstack( (np.ones((data_hist_pos.shape[0], 1)), (data_hist_vel[:,1:2] ))) # n x 2
    A_data_theta = np.hstack( (np.ones((data_hist_pos.shape[0], 1)), (data_hist_vel[:,2:3] ))) # n x 2
    b_data_x = data_hist_pos[:,0] # n x 1
    b_data_y = data_hist_pos[:,1]
    b_data_theta = data_hist_pos[:,2]
    goal_eig_x = np.linalg.lstsq(A_data_x, b_data_x, rcond=None)
    goal_eig_y = lsqL2(A_data_y, b_data_y)
    goal_eig_theta = np.linalg.lstsq(A_data_theta, b_data_theta, rcond=None)
    goal_pred = [goal_eig_x[0][0], goal_eig_y[0], goal_eig_theta[0][0]]
    eig_pred  = [1.0/goal_eig_x[0][1], 1.0/goal_eig_y[1], 1.0/goal_eig_theta[0][1]]
    A_est = A_only_est(xy_history)
    
    return goal_pred, eig_pred, A_est

def multi_part_weighing(y_hat, y, xy_history, F, manipuated_obj, dt = 0.02, k_exp= 0.33, L1 = 35.0, L2 = 0.0, weight_velocity = 0.0, weight_A = 1.0, ik_solver = None,  ):

    weight_force = 0.0 # this is not used for now

    y = y.reshape((-1))
    d = np.zeros(y_hat.shape[0])

    current_state = xy_history[-1].reshape(-1)[0:3]
    data_hist_pos = xy_history[:, 0:3]
    data_hist_vel = xy_history[:, 3:6]

    v_state = data_hist_vel[-1]

    A_closed_form = A_only_est(xy_history)
    for i in range(A_closed_form.shape[0]):
        if i == 2:
            if A_closed_form[i] > A_UPPER_BOUND_ROLL:
                A_closed_form[i] = A_UPPER_BOUND_ROLL
            elif A_closed_form[i] < A_LOWER_BOUND_ROLL:
                A_closed_form[i] = A_LOWER_BOUND_ROLL
        else:
            if A_closed_form[i] > A_UPPER_BOUND:
                A_closed_form[i] = A_UPPER_BOUND
            elif A_closed_form[i] < A_LOWER_BOUND:
                A_closed_form[i] = A_LOWER_BOUND
    #vectorized operation of the following
    # goal_state_vec = y_hat[:, 3:6]
    # state_expect_vec = goal_state_vec +  1.0/y_hat[:, 6:9] * data_hist_vel
    # s_inv_A_vec = -np.linalg.norm(state_expect_vec[:,0:3] - data_hist_pos[:,0:3], ord=2, axis=1)**2
    # s_A_alone_vec = -np.linalg.norm(y_hat[:, 6:9] - A_closed_form, ord=2, axis=1)**2

    for i in range(y_hat.shape[0]):
        goal_state = y_hat[i, 3:6]
        # print("particle state", particle_state)
        # print("current state", current_state)
        #                 3             3                  8 x 3            
        state_expect = goal_state +  1.0/y_hat[i, 6:9] * data_hist_vel
        # print("state expect", state_expect)
        s_inv_A = -np.linalg.norm(state_expect[:,0:3] - data_hist_pos[:,0:3], ord=2)**2
        s_A_alone = -np.linalg.norm(y_hat[i, 6:9] - A_closed_form, ord=2)**2
        
        # state_expect_closed_form = goal_state +  1.0/A_closed_form * data_hist_vel
        # s_inv_A_closed_form = -np.linalg.norm(state_expect_closed_form[:,0:3] - data_hist_pos[:,0:3], ord=2)
        # L1 = 35
        # L2 = 0

        # L1_yaw = 10
        # L2_yaw = 5
        # print("particle state", particle_state)
        # print("current state", current_state)
        # F_expected = np.clip(L1 * (goal_state - current_state), -L1, L1) - L2 * v_state
        # F_expected[2] = np.clip(L1_yaw*(particle_state[2] - current_state[2]), -L1_yaw, L1_yaw) - L2_yaw * yaw
        # F_expected = L1 * (particle_state - current_state) - L2 * v_state
        # Saturate max distance of the target to 1m away in each direction, and if larger, still expect same max force
        # Reduce expected force if table is already moving

        s_f = 0.0#-np.linalg.norm(F_expected - F.reshape(-1))/70  # [0,1] # hopefully
        # print("F sensor", F)
        # print(s_f)
        # yaw_diff = wrap_to_pos_neg_pi(yaw - ob_yaw)
        # if abs(yaw_diff) < np.pi / 2:
        #     s_yaw = ellipse_r(yaw_diff, 2, 1)
        # else:
        #     s_yaw = 1
        ######################
        ### POST PROCESSING, READ THE FOLLOWING !!!!###
        ######################
        # s_yaw = 0.0 # no sensor for pose yet
        # s_dir = 0.0 # this works
        # s_v = 0.0 # this does not work that well
        # s_f = 0.0 # this works
        # s_inv_A = 0.0 # this works
        # s_yaw = max(ellipse_r( 2, 1), 1)#[0,1]
        # print("Particle Scores:", s_dir, s_v, s_vf, s_yaw)
        d[i] = weight_velocity * s_inv_A + weight_force * s_f + weight_A*s_A_alone
    # print("=====================================")
    
    #particle trimming
    # apply manipuated_obj.get_left_right_coordinate to each particle

    # print(left_coord_rot - y_hat[:, 3:6])
    # print("yhat", y_hat[:, 3:6])
    # print("left", left_coord_rot)

    # particle_bool = np.ones(y_hat.shape[0], dtype=bool)
    particle_bool = curobo_ik_filter(y_hat[:,3:6], ik_solver, manipuated_obj)

    # right_coord_rot = np.concatenate((right_coord, y_hat[:, 2:3]), axis=1)
    # right_coord_rot[:, 0] -= human_x
    # right_coord_rot[:, 1] -= human_y
    # right_coord_rot[:, 2] = right_coord_rot[:, 2] + np.pi - human_yaw
    # particle_bool_right = curobo_ik_filter(right_coord_rot, ik_solver)
    # state_feasible = y_hat[particle_bool == 1, 3:6] 
    # state_feasible_right = y_hat[particle_bool_right == 1, 3:6]
    # rospy.loginfo_throttle(15, f"state feasible: {state_feasible}")    
    d[particle_bool == 0] = -1000.0
    # d[particle_bool_right == 0] = -1000.0

    # Also Trim A that is too large or small
    A_mat_xy = y_hat[:, 6:8]
    A_mat_roll = y_hat[:, 8:9]
    bool_A_out_of_range_xy = np.any((A_mat_xy > A_UPPER_BOUND) | (A_mat_xy < A_LOWER_BOUND), axis=1)
    bool_A_out_of_range_roll = np.any((A_mat_roll > A_UPPER_BOUND_ROLL) | (A_mat_roll < A_LOWER_BOUND_ROLL), axis=1)
    bool_A_out_of_range = bool_A_out_of_range_xy | bool_A_out_of_range_roll
    if np.count_nonzero(bool_A_out_of_range) > 0.3 * d.shape[0]:
        rospy.logwarn_throttle(2, "more than 30% A out of range")
        print(A_mat_xy[0:10,:])
    d[bool_A_out_of_range] = -1000.0
    return np.exp(k_exp * d)


def simple_xy_noise_wo_A(x, xy_history):
    return t_noise(x, sigmas=[0.01, 0.01, 0.0, 0.0, 0.0, 0.0], df=100.0)


def simple_xy_noise_A(x, xy_history, F):
    return t_noise(x, sigmas=[0.03, 0.03, 0.5, 0.0, 0.0, 0.5], df=100.0)


def simple_xy_noise_A_nonneg(x, xy_history, F, x_noise, y_noise, theta_noise, x_lim, y_lim, theta_limit):
    plus_noise = t_noise(
        x, sigmas=[x_noise, y_noise, theta_noise, x_noise/3, 0.0, 0.0, 0.0, y_noise/3, 0.0, 0.0, 0.0, theta_noise/3], df=100.0 #real 
        # x, sigmas=[0.05, 0.05, 0.05, 0.3, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.3], df=100.0 #sim table
    )
    arr_1 = plus_noise[:, 3]
    arr_2 = plus_noise[:, 7]
    arr_3 = plus_noise[:, 11]
    arr_y = plus_noise[:, 1]

    arr_theta = plus_noise[:, 2]
    arr_theta[arr_theta > theta_limit] = theta_limit
    arr_theta[arr_theta < -theta_limit] = -theta_limit
    arr_y[arr_y > y_lim] = y_lim
    plus_noise[:, 2] = arr_theta

    arr_1[arr_1 > A_UPPER_BOUND] = A_UPPER_BOUND
    arr_2[arr_2 > A_UPPER_BOUND] = A_UPPER_BOUND
    arr_3[arr_3 > A_UPPER_BOUND_ROLL] = A_UPPER_BOUND_ROLL
    arr_1[arr_1 < A_LOWER_BOUND] = A_LOWER_BOUND
    arr_2[arr_2 < A_LOWER_BOUND] = A_LOWER_BOUND
    arr_3[arr_3 < A_LOWER_BOUND_ROLL] = A_LOWER_BOUND_ROLL
    plus_noise[:, 3] = arr_1
    plus_noise[:, 7] = arr_2
    plus_noise[:, 11] = arr_3
    plus_noise[:, 1] = arr_y
    return plus_noise

def ellipsoid_xy_noise_A_nonneg(x, xy_history, F, x_noise, y_noise, theta_noise, x_lim, y_lim, theta_limit, ellipsoid_matrix):

    M = ellipsoid_matrix[:3,:3]
    human_ee = ellipsoid_matrix[:3,3]
    dx = x - human_ee[:, np.newaxis]
    # dx_xy = dx[:2,:]
    c_influence = 0.4
    # A = self.mean_state[3:12].reshape(3,3)
    h_fn = np.exp(-c_influence*np.linalg.norm(dx, axis=0))

    M_xyz = (1-h_fn)* np.eye(3) + h_fn* M
    M_xy = np.vstack((np.hstack((M_xyz[:2,:2], 0)), np.array([0.0, 0.0, 1.0])))

    plus_noise = t_noise(x, sigmas=[x_noise, y_noise, theta_noise, \
                         M_xy[0,0]*x_noise/3, M_xy[0,1]*y_noise/3, 0.0, \
                         M_xy[1,0]*x_noise/3, M_xy[1,1]*y_noise/3, 0.0, \
                         0.0, 0.0, theta_noise/3], df=100.0 ) #real 
    
    arr_11 = plus_noise[:, 3]
    arr_12 = plus_noise[:, 4]
    arr_21 = plus_noise[:, 6]
    arr_22 = plus_noise[:, 7]
    arr_3 = plus_noise[:, 11]
    arr_y = plus_noise[:, 1]
    arr_theta = plus_noise[:, 2]

    arr_theta[arr_theta > theta_limit] = theta_limit
    arr_theta[arr_theta < -theta_limit] = -theta_limit
    arr_y[arr_y > y_lim] = y_lim
    
    arr_11[arr_11 > A_UPPER_BOUND] = A_UPPER_BOUND
    arr_12[arr_12 > A_UPPER_BOUND] = A_UPPER_BOUND
    arr_21[arr_21 > A_UPPER_BOUND] = A_UPPER_BOUND
    arr_22[arr_22 > A_UPPER_BOUND] = A_UPPER_BOUND
    arr_11[arr_11 < A_LOWER_BOUND] = A_LOWER_BOUND
    arr_12[arr_12 < A_LOWER_BOUND] = A_LOWER_BOUND
    arr_21[arr_21 < A_LOWER_BOUND] = A_LOWER_BOUND
    arr_22[arr_22 < A_LOWER_BOUND] = A_LOWER_BOUND
    arr_3[arr_3 > A_UPPER_BOUND_ROLL] = A_UPPER_BOUND_ROLL
    arr_3[arr_3 < A_LOWER_BOUND_ROLL] = A_LOWER_BOUND_ROLL

    plus_noise[:, 3] = arr_11
    plus_noise[:, 4] = arr_12 
    plus_noise[:, 6] = arr_21 
    plus_noise[:, 7] = arr_22 
    plus_noise[:, 11] = arr_3
    plus_noise[:, 1] = arr_y
    plus_noise[:, 2] = arr_theta

    return plus_noise

def traj_to_particle_with_obs(particles, xy_history, F, obstacles=None):
    curr_state_full = xy_history[-1].reshape(-1, 1)
    curr_state_pos = curr_state_full[0:3, :]
    
    # There are 4 observations we can make from each particle
    # 1. Direction of VelocityA
    # 2. Desired Velocity
    # 3. just the x y pos state
    # 4. Most comfortable posture
    # shape of observation should be #1(2), #2(1), #3(2), #4(1)[yaw]
    y = np.zeros((particles.shape[0], 12))

    for i in range(particles.shape[0]):
        A_mat = particles[i, 3:].reshape(3, 3)
        v_all = A_mat @ (curr_state_pos - particles[i, 0:3].reshape(-1, 1))
        v = v_all[0:2, :]  # linear
        # w = v_all[2, :]  # angular
        v_mag = np.linalg.norm(v)  # 2
        v_dir = v / v_mag  # 1
        yaw = np.arctan2(v_dir[1], v_dir[0])  # 4
        y[i, 0:2] = v_dir.reshape(-1)
        y[i, 2] = v_mag
        y[i, 3:6] = particles[i, 0:3]
        y[i, 6] = particles[i, 3]
        y[i, 7] = particles[i, 7]
        y[i, 8] = particles[i, 11]
        #THERE ARE ALSO 
        # y[i, 9]  10 , 11!!!
    return y

def human_controller(v_human_des, v_desk):
    F_human = Kv * (v_human_des - v_desk)
    # M_human = Kw * (v_human_des[2] - theta_w_desk[1])
    return F_human  # np.vstack((F_human, M_human))
