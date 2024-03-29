# This python file parses npz files saved by the commented out code below
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from copy import deepcopy
from scipy.spatial.transform import Rotation as R
from scipy.integrate import simps
from tf.transformations import euler_from_quaternion, quaternion_from_euler, quaternion_matrix
import seaborn as sns
path_prefix = "/home/yifei/kuka_ws/data/"
colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
num_planner_start = 2
num_planner_end = 2
num_planner = num_planner_end - num_planner_start + 1
planner_names = ["no controller", "Proposed", "Admittance", "Proposed with\nForce Activation"]
N = 10 
plot_bool = True
Impulse = np.zeros((N, num_planner))
Ang = np.zeros((N, num_planner))
Impulse_robot = np.zeros((N, num_planner))
Ang_robot = np.zeros((N, num_planner))
for trial_num in range(N):
    if plot_bool:
        # plot 3d trajectory
        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('roll')
        ax.set_xlim3d(0.5, 1.3)
        ax.set_ylim3d(-1.2, 1.2)
        ax.set_zlim3d(-1.6, 1.6)
        ax.view_init(elev=45, azim=0)
        fig2, axes = plt.subplots(2, num_planner, figsize=(12,6))
        #put fig 2 location to the right of the screen
        
    for planner_id in range(0, num_planner):
        if plot_bool:
            if num_planner == 1:
                ax2 = axes[0]
                ax22 = axes[1]
                ax22twin = ax22.twinx()
            else:
                ax2 = axes[0, planner_id]
                ax22 = axes[1, planner_id]
                ax22twin = ax22.twinx()
        planner_id_file = planner_id + num_planner_start
        file_name = path_prefix + 'ECI_traj_trial_' + str(trial_num) + '_planner_' + str(planner_id_file) + '.npz'
        print(file_name)
        data = np.load(file_name)
        pose_traj = data['pose_traj']
        twist_traj = data['twist_traj']
        wrench_human_traj = data['wrench_human_traj']
        wrench_robot_traj = data['wrench_robot_traj']
        # wrench_robot_exec_traj = data['wrench_robot_effort_traj']
        time_traj = data['time_traj']
        goal_pose = data['goal_pose']
        # print(wrench_human_traj.shape)

        N = len(pose_traj)
        # print(N)
        # convert quaternion to rotation matrix
        rot_traj = np.zeros((N, 3, 3))
        position = np.zeros((N, 3))
        vel = np.zeros((N, 6))
        wrench_human = np.zeros((N, 6))
        roll_traj = np.zeros(N)
        # print('goal angle =', np.arctan2(goal_pose[1], goal_pose[0]))
        # goal_rot = R.from_quat(goal_pose[3:]).as_matrix()#in 2d this is not valid
        # get roll from 


        # print(pose_traj[0,:])
        for i in range(N):
            rot_traj[i,:,:] = R.from_quat(pose_traj[i,3:]).as_matrix()
            position[i,:] = pose_traj[i,0:3]
            two_rot_stacked = np.vstack((np.hstack((np.eye(3), np.zeros((3,3)))), np.hstack((np.zeros((3,3)), rot_traj[i,:,:].T))))
            (ang1, ang2, roll_traj[i]) = euler_from_quaternion(pose_traj[i,3:], axes='ryxz')
            vel[i,:] = np.matmul(two_rot_stacked, twist_traj[i,:])
            wrench_human[i,:3] = wrench_human_traj[i,:3]
            wrench_human[i,3:] = wrench_human_traj[i,3:]#np.matmul(rot_traj[i,:,:].T, wrench_human_traj[i,3:])
            ######THIS ROTATION MAT IS WRONG, it is the end effector frame, but the data is in sensor frame

        force_human_norm_traj = np.linalg.norm(wrench_human[:,0:3], axis=1)
        torque_human_norm_traj = np.linalg.norm(wrench_human[:,3:6], axis=1)
        force_robot_norm_traj = np.linalg.norm(wrench_robot_traj[:,0:3], axis=1)
        torque_robot_norm_traj = np.linalg.norm(wrench_robot_traj[:,3:6], axis=1)
        # print(force_human_norm_traj.shape)
        # print(force_human_norm_traj[::1000])
        impulse_human = simps(force_human_norm_traj, time_traj)
        angular_impulse_human = simps(torque_human_norm_traj, time_traj)
        impulse_robot = simps(force_robot_norm_traj, time_traj)
        angular_impulse_robot = simps(torque_robot_norm_traj, time_traj)
        Impulse[trial_num, planner_id] = impulse_human
        Ang[trial_num, planner_id] = angular_impulse_human
        Impulse_robot[trial_num, planner_id] = impulse_robot
        Ang_robot[trial_num, planner_id] = angular_impulse_robot
        # print(roll_traj)
        if plot_bool:
            ax.quiver(position[:,0], position[:,1], roll_traj, vel[:,0], vel[:,1], vel[:,5], length=0.05, colors  = colors[planner_id], label=planner_names[planner_id_file])
            ax.scatter(goal_pose[0], goal_pose[1], goal_pose[2], color='black', s=100, marker='*')
            ax2.plot(time_traj, np.linalg.norm(vel[:,0:2], axis=1))
            ax2.plot(time_traj, vel[:,5])
            ax2.set_xlabel('t')
            ax2.legend(['xyz vel norm', 'roll vel'])
            ax2.set_ylim([-0.5, 0.5])
            ax2.set_xlim([0, 20])
            if data['success_bool']:
                ax2.text(0.2, 0.9,  "Success", transform=ax2.transAxes, color='green')
            else:
                ax2.text(0.2, 0.9, "Failed", transform=ax2.transAxes, color='red')
            ax22.plot(time_traj, np.linalg.norm(wrench_human[:,0:3], axis=1), color='red')
            ax22.plot(time_traj, np.linalg.norm(wrench_robot_traj[:,0:3], axis=1), color='red', linestyle='dashed')
            # ax22twin.plot(time_traj, wrench_human[:,3], color='red')
            # ax22twin.plot(time_traj, wrench_human[:,4], color='green')
            ax22twin.plot(time_traj, wrench_human[:,5], color='blue')
            # ax22twin.plot(time_traj, np.linalg.norm(wrench_robot_traj[:,3:6], axis=1), color='blue', linestyle='dashed')
            ax22twin.plot(time_traj, wrench_robot_traj[:,5], color='blue', linestyle='dashed')
            ax22twin.hlines(0, 0, 20, color='black', linestyles='dashed')
            ax22twin.set_ylabel('torque', color='blue')
            ax22twin.set_ylim([-8.0, 8.0])
            ax22.set_ylabel('force', color='red')
            ax22.set_xlabel(planner_names[planner_id_file])
            ax22.set_ylim([0, 35])
            ax22.set_xlim([0, 20])
            ax22.legend(['human force', 'robot force'], loc='upper right')
            ax22twin.legend(['human torque', 'robot torque'], loc='lower right')
            # write text on plot
            ax22.text(0.2, 0.9, 'Avg Force: ' + str(round(impulse_human/time_traj[-1], 2)), transform=ax22.transAxes)
            ax22.text(0.2, 0.8, 'Avg Torque: ' + str(round(angular_impulse_human/time_traj[-1], 2)), transform=ax22.transAxes)
            ax22.text(0.4, 0.9, 'Avg Force Robot: ' + str(round(impulse_robot/time_traj[-1], 2)), transform=ax22.transAxes)
            ax22.text(0.4, 0.8, 'Avg Torque Robot: ' + str(round(angular_impulse_robot/time_traj[-1], 2)), transform=ax22.transAxes)

    if plot_bool:
        ax.legend()
        #wait for user input
        # input("Press Enter to continue...")
        plt.show()
        # plt.close('all')

#use seaborn to plot boxplot of impulse and angular impulse
# aspect ratio 3:2
# put planner names on x axis
fig3 = plt.figure(figsize=(9,6))
ax3 = fig3.add_subplot(1,2,1)
ax3.set_ylabel('Impulse')
ax3.set_title('Impulse')
ax3.boxplot(Impulse, labels=planner_names[num_planner_start:num_planner_end+1])
ax4 = fig3.add_subplot(1,2,2)
ax4.set_ylabel('Angular Impulse')
ax4.set_title('Angular Impulse')
ax4.boxplot(Ang, labels=planner_names[num_planner_start:num_planner_end+1])


# save figure
plt.savefig(path_prefix + 'boxplot.png')
# plt.show()



                        # self.success[i,client_idx] = result.success
                        #     tqdm.write("Solve Status: trial "+ str(seed_val) + " planner: " + str(planner_id) + " status: "+ str(result.success))
                        #     success_bool = np.array(result.success, dtype=bool)
                        #     pose_traj = list_of_pose_to_numpy(result.pose)
                        #     twist_traj = list_of_twist_to_numpy(result.twist)
                        #     wrench_human_traj = list_of_wrench_to_numpy(result.human_effort_world)
                        #     wrench_robot_traj = list_of_wrench_to_numpy(result.robot_effort_world)
                        #     time_traj = np.array(result.time_vec)
                        #     goal_pose = np.array([msg.p_final.position.x, msg.p_final.position.y, msg.p_final.position.z, msg.p_final.orientation.x, msg.p_final.orientation.y, msg.p_final.orientation.z, msg.p_final.orientation.w])
                        #     # write numpy to npy file
                        #     np.savez(path_prefix+'ECI_traj_trial_'+str(i)+'_planner_'+str(planner_id)+'.npz', pose_traj=pose_traj, twist_traj=twist_traj,
                        #               wrench_human_traj=wrench_human_traj, wrench_robot_traj=wrench_robot_traj, time_traj=time_traj, success_bool=success_bool, goal_pose=goal_pose)
