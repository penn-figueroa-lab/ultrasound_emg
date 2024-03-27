# this is a class for a desk in 2D environment with side length a and b. with mass m and moment of inertia I
# When providing force on two ends of the desk, the desk will move or rotate
import math
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from scipy.spatial.transform import Rotation as R


def Rot2D(theta):
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])


class Desk2D(object):
    def __init__(self, pos, vel, dt, a=1.0, b=0.15, m=15.0):  # mass in kg length in m
        self.a = a
        self.b = b
        self.m = m
        self.I = m * (a**2 + b**2) / 12
        self.pos = pos # x, y, theta
        self.vel = vel # vx, vy, omega
        self.dt = dt

    # def set_initial(self, x, y, theta, vx, vy, omega):
    #     self.x = x
    #     self.pos[1] = y
    #     self.pos[-1] = theta
    #     self.vel[0] = vx
    #     self.vel[1] = vy
    #     self.vel[-1] = omega
    def get_theta_deg(self):
        return math.degrees(self.pos[-1])
    def get_world_frame_theta_w(self):
        print(self.pos[-1], self.vel[-1])
        return np.array([self.pos[-1], self.vel[-1]])

    def get_world_frame_velocity(self):
        # return the velocity in world frame
        return np.vstack(
            (
                Rot2D(self.pos[-1]) @ (np.array([self.vel[0], self.vel[1]]).reshape(-1, 1)),
                self.vel[-1],
            )
        )

    def move_world_frame(self, FM1, FM2):
        # exit()
        # print('vvv')
        F1 = FM1[0:2, :]
        M1 = FM1[2,0]
        F2 = FM2[0:2, :]
        M2 = FM2[2,0]
        F1_desk = (Rot2D(-self.pos[-1]) @ F1).reshape(-1)
        F2_desk = (Rot2D(-self.pos[-1]) @ F2).reshape(-1)
        self.move(F1_desk, F2_desk, M1 + M2)

    def move(self, F1, F2, M):
        # F1 is the force on the left end, F2 is the force on the right end
        # F1 and F2 are numpy arrays each are 2D vectors with x and y components
        # vx and vy are the velocity of the center of mass in ego frame, to get world coordintate, need rotation

        # print('calling move')
        self.pos[0] = (
            self.pos[0]
            + math.cos(self.pos[-1]) * self.vel[0] * self.dt
            - math.sin(self.pos[-1]) * self.vel[1] * self.dt
        )
        self.pos[1] = (
            self.pos[1]
            + math.sin(self.pos[-1]) * self.vel[0] * self.dt
            + math.cos(self.pos[-1]) * self.vel[1] * self.dt
        )
        self.pos[-1] = self.pos[-1] + self.vel[-1] * self.dt
        self.vel[0] = self.vel[0]*0.92 + (F1[0] + F2[0]) / self.m * self.dt
        self.vel[1] = self.vel[1]*0.92 + (F1[1] + F2[1]) / self.m * self.dt
        self.vel[-1] = (
            self.vel[-1]*0.92
            + (-F1[1] * self.a / 2 + F2[1] * self.a / 2 + M) / self.I * self.dt
        )

        # print (self.pos[0], self.pos[1], self.pos[-1], self.vel[0], self.vel[1], self.vel[-1], F1[0] + F2[0])

    def plot(self, ax):
        # plot the desk in the ax
        rectangle = patches.Rectangle(
            (-self.a / 2 + self.pos[0], -self.b / 2 + self.pos[1]),
            self.a,
            self.b,
            linewidth=1,
            edgecolor="r",
            facecolor="none",
            rotation_point="center",
            angle=math.degrees(self.pos[-1]),
        )
        desk_patch = ax.add_patch(rectangle)
        return desk_patch

    def get_left_right_coordinate(self):
        # return the left and right coordinate of the desk
        left = np.array([-self.a / 2, 0]).reshape(-1, 1)
        right = np.array([self.a / 2, 0]).reshape(-1, 1)
        left = Rot2D(self.pos[-1]) @ left + np.array([self.pos[0], self.pos[1]]).reshape(-1, 1)
        right = Rot2D(self.pos[-1]) @ right + np.array([self.pos[0], self.pos[1]]).reshape(-1, 1)
        return left.reshape(-1), right.reshape(-1)
    def get_left_right_coordinate_state(self, state):
        # given state x, y, theta. Return the left and right coordinate of the desk
        left = np.array([-self.a / 2, 0]).reshape(-1, 1)
        right = np.array([self.a / 2, 0]).reshape(-1, 1)
        left = Rot2D(state[-1]) @ left + np.array([state[0], state[1]]).reshape(-1, 1)
        right = Rot2D(state[-1]) @ right + np.array([state[0], state[1]]).reshape(-1, 1)
        return left.reshape(-1), right.reshape(-1)
        

class Pot3D(object):
    def __init__(self, pos, vel, dt, r=0.1525, m=15.0):  # mass in kg length in m
        self.r = r
        self.m = m
        self.I = (m * r**2) / 2.0
        self.pos = pos # x, y, z, roll, yaw, pitch
        self.vel = vel # vx, vy, vz, omega
        self.dt = dt

    # def set_initial(self, x, y, theta, vx, vy, omega):
    #     self.x = x
    #     self.pos[1] = y
    #     self.pos[-1] = theta
    #     self.vel[0] = vx
    #     self.vel[1] = vy
    #     self.vel[-1] = omega
    def get_theta_deg(self):
        return math.degrees(self.pos[-1])
    def get_world_frame_theta_w(self):
        print(self.pos[-1], self.vel[-1])
        return np.array([self.pos[-1], self.vel[-1]])

    def get_world_frame_velocity(self):
        # return the velocity in world frame
        return np.vstack(
            (
                Rot2D(self.pos[-1]) @ (np.array([self.vel[0], self.vel[1]]).reshape(-1, 1)),
                self.vel[-1],
            )
        )

    def move_world_frame(self, FM1, FM2):
        # exit()
        # print('vvv')
        F1 = FM1[0:2, :]
        M1 = FM1[2,0]
        F2 = FM2[0:2, :]
        M2 = FM2[2,0]
        F1_desk = (Rot2D(-self.pos[-1]) @ F1).reshape(-1)
        F2_desk = (Rot2D(-self.pos[-1]) @ F2).reshape(-1)
        self.move(F1_desk, F2_desk, M1 + M2)

    def move(self, F1, F2, M):
        # F1 is the force on the left end, F2 is the force on the right end
        # F1 and F2 are numpy arrays each are 2D vectors with x and y components
        # vx and vy are the velocity of the center of mass in ego frame, to get world coordintate, need rotation

        # print('calling move')
        self.pos[0] = (
            self.pos[0]
            + math.cos(self.pos[-1]) * self.vel[0] * self.dt
            - math.sin(self.pos[-1]) * self.vel[1] * self.dt
        )
        self.pos[1] = (
            self.pos[1]
            + math.sin(self.pos[-1]) * self.vel[0] * self.dt
            + math.cos(self.pos[-1]) * self.vel[1] * self.dt
        )
        self.pos[-1] = self.pos[-1] + self.vel[-1] * self.dt
        self.vel[0] = self.vel[0]*0.92 + (F1[0] + F2[0]) / self.m * self.dt
        self.vel[1] = self.vel[1]*0.92 + (F1[1] + F2[1]) / self.m * self.dt
        self.vel[-1] = (
            self.vel[-1]*0.92
            + (-F1[1] * self.a / 2 + F2[1] * self.a / 2 + M) / self.I * self.dt
        )

        # print (self.pos[0], self.pos[1], self.pos[-1], self.vel[0], self.vel[1], self.vel[-1], F1[0] + F2[0])


    def draw_frame_axis(self, T, ax, origin=False):
        if ax is None:
            return
        
        x_axis = self.T_multi_vec(T, np.array([0.1,    0,    0]))
        y_axis = self.T_multi_vec(T, np.array([0,    0.1,    0]))
        z_axis = self.T_multi_vec(T, np.array([0,    0,    0.1]))

        center = self.T_multi_vec(T, np.array([0.0, 0.0, 0.0]))
        stack_x = np.vstack((center, x_axis))
        stack_y = np.vstack((center, y_axis))
        stack_z = np.vstack((center, z_axis))

        if origin:
            ax.plot(stack_x[:,0], stack_x[:,1], stack_x[:,2], color='black')
            ax.plot(stack_y[:,0], stack_y[:,1], stack_y[:,2], color='black')
            ax.plot(stack_z[:,0], stack_z[:,1], stack_z[:,2], color='black')
        else:
            ax.plot(stack_x[:,0], stack_x[:,1], stack_x[:,2], color='red')
            ax.plot(stack_y[:,0], stack_y[:,1], stack_y[:,2], color='green')
            ax.plot(stack_z[:,0], stack_z[:,1], stack_z[:,2], color='blue')


    def T_multi_vec(self, T, vec):
        vec = vec.flatten()
        return (T @ np.append(vec, 1.0).reshape(-1,1)).flatten()[:3]
    
    def getT_euler(self, x, y, z, roll, pitch, yaw):
        rot = R.from_euler('xyz', [roll, pitch, yaw]).as_matrix()
        T = np.array([[1, 0, 0, x], [0, 1, 0, y], [0, 0, 1, z], [0,0,0,1]])
        T[:3, :3] = rot

        return T
    
    def plot(self):
        fig, ax = plt.subplots(1,1, subplot_kw=dict(projection='3d'))
        ax.view_init(elev=30, azim=45)
        self.draw_frame_axis(np.eye(4), ax, origin=True)

        self.draw_frame_axis(self.getT_euler(*self.pos), ax)

        human_side_T, robot_side_T = self.get_left_right_coordinate()
        self.draw_frame_axis(human_side_T, ax)
        self.draw_frame_axis(robot_side_T, ax)

        ax.set_xlim([0.0,1.0])
        ax.set_ylim([-1.0,1.0])
        ax.set_zlim([-1.0,1.0])
        ax.set_xlabel('x') 
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_aspect('equal')
        plt.show()
        

    def get_left_right_coordinate(self):
        pot_in_world_T = self.getT_euler(*self.pos)
        human_side_T = pot_in_world_T @ self.getT_euler(self.r, 0.0, 0.0, 0.0, 0.0, 0.0)
        robot_side_T = pot_in_world_T @ self.getT_euler(-self.r, 0.0, 0.0, 0.0, 0.0, np.pi)
        return human_side_T, robot_side_T
    



