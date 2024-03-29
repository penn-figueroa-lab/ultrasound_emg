import bagpy
from bagpy import bagreader
import pandas as pd
import seaborn as sea
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import glob
import copy
import pickle

# def process_bag(b_list, sample_step, plot=False):

#     traj_list = []
#     for bi in b_list:
#         print(b_list)
        
    #     data = bi.message_by_topic('/franka_state_controller/O_T_EE')
    #     df = pd.read_csv(data)
    #     df = df.to_numpy()#[::sample_step, :]
    #     print(df.shape)

    #     if plot:
    #         fig = plt.figure()
    #         ax = plt.axes(projection='3d')
    #         ax.set_aspect('equal')
    #         ax.plot(df[:,5], df[:,6],df[:,7], color='black', label='demonstrations')
    #         handles, labels = plt.gca().get_legend_handles_labels()
    #         by_label = dict(zip(labels, handles))
    #         #plt.legend(by_label.values(), by_label.keys())
    #         ax.set_xlim(0.0, 1.0)
    #         ax.set_ylim(-0.8, 0.8)
    #         ax.set_zlim(-0.1, 1.0)
    #         plt.show()

        
    #     sec = df[:, 2] + (1e-9)*df[:, 3]
    #     first_stamp = copy.deepcopy(sec[0])
    #     sec = (sec - first_stamp).reshape(-1,1)

    #     traj_set = np.array(np.hstack((df[:, 5:8], sec)), dtype=np.float64)

    #     traj_list.append(df[:, 5:8])
    # return traj_list
    

def read_rosbag(filename):
    b_list = bagreader(filename+".bag", tmp=True)
    print(b_list.topic_table)
    ee_vel_data = b_list.message_by_topic('/iiwa/desired_twist')
    ee_vel_df = pd.read_csv(ee_vel_data)
    print(ee_vel_df.columns.values.tolist())
    ee_vel_df = ee_vel_df.to_numpy()#[::sample_step, :]


    ee_force_data = b_list.message_by_topic('/robotiq_force_torque_wrench_filtered')
    ee_force_df = pd.read_csv(ee_force_data)
    print(ee_force_df.columns.values.tolist())
    ee_force_df = ee_force_df.to_numpy()#[::sample_step, :]

    fig, ax = plt.subplots(3,1)


    ax[0].plot(ee_vel_df[:,0]-1694558461.91, ee_vel_df[:,5], color='black', label='particleDS x')
    ax[1].plot(ee_vel_df[:,0]-1694558461.91, ee_vel_df[:,6], color='black', label='particleDS y')
    ax[2].plot(ee_vel_df[:,0]-1694558461.91, ee_vel_df[:,-1], color='black', label='particleDS yaw')


    ax[0].plot(ee_force_df[:,0]-1694558461.91, ee_force_df[:,7]/30, color='red', label='force measure x')
    ax[1].plot(ee_force_df[:,0]-1694558461.91, ee_force_df[:,5]/30, color='red', label='force measure y')
    ax[2].plot(ee_force_df[:,0]-1694558461.91, ee_force_df[:,-2]/10, color='red', label='force measure yaw')

    ax[0].legend()
    ax[0].set_xlim([0, 23])
    ax[0].set_title('x')
    ax[1].legend()
    ax[1].set_xlim([0, 23])
    ax[1].set_title('y')
    ax[2].legend()
    ax[2].set_xlim([0, 23])
    ax[2].set_title('yaw')
    plt.show()

if __name__ == '__main__':
    read_rosbag('/file/path/here')