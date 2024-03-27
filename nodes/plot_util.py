import numpy as np
from matplotlib.patches import Arc
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import random
import time
random.seed(0)

PLOT_PARTICLES = True
DT = 0.05

plot_sample = 30
def multivariate_normal(x, d, mean, covariance):
    """pdf of the multivariate normal distribution."""
    x_m = x - mean
    return (1. / (np.sqrt((2 * np.pi)**d * np.linalg.det(covariance))) * 
            np.exp(-(np.linalg.solve(covariance, x_m).T.dot(x_m)) / 2))

def generate_surface(mean, covariance, d):
    """Helper function to generate density surface."""
    nb_of_x = plot_sample # grid size
    x1s = np.linspace(-2.5, 2.5, num=nb_of_x)
    x2s = np.linspace(-2.5, 2.5, num=nb_of_x)
    x1, x2 = np.meshgrid(x1s, x2s) # Generate grid
    pdf = np.zeros((nb_of_x, nb_of_x))
    # Fill the cost matrix for each combination of weights
    for i in range(nb_of_x):
        for j in range(nb_of_x):
            pdf[i,j] = multivariate_normal(
                np.matrix([[x1[i,j]], [x2[i,j]]]), 
                d, mean[:2], covariance[:2,:2])
    return x1, x2, pdf

class SimPlot:
    def __init__(self, dt) -> None:
        self.curr_idx = 0
        self.dt = dt
        fig, self.ax = plt.subplots(1, 1)

    def plot_sim(self, pf, desk, F_right, F_left, yaw, xyt, xyt_hist):
        # mixture DS
        self.curr_idx += 1
        t = self.curr_idx * self.dt

        pf.plot_DS(self.ax)
        left_coord, right_coord = desk.get_left_right_coordinate()
        theta_desk = desk.get_theta_deg()
        F_right_plot = F_right.reshape(-1) / 100
        F_left_plot = F_left.reshape(-1) / 100
        human_moment_plot = F_left_plot[2]*50
        assist_moment_plot = F_right_plot[2]*50
        print("human moment plot", human_moment_plot, desk.pos[-1])
        moment_plot_radius = 0.8
        if human_moment_plot > 0:
            self.ax.add_patch(Arc( (left_coord[0],
                left_coord[1]), moment_plot_radius, moment_plot_radius, theta_desk, theta2=human_moment_plot, color='yellow', lw = 4)) # draw arc
        else:
            self.ax.add_patch(Arc( (left_coord[0],
                left_coord[1]), moment_plot_radius, moment_plot_radius, theta_desk + human_moment_plot, theta2=-human_moment_plot, color='yellow',  lw = 4))
        if assist_moment_plot > 0:
            self.ax.add_patch(Arc( (right_coord[0],
                right_coord[1]), moment_plot_radius, moment_plot_radius, theta_desk, theta2=assist_moment_plot, color='white', lw = 4)) # draw arc
        else:
            self.ax.add_patch(Arc( (right_coord[0],
                right_coord[1]), moment_plot_radius, moment_plot_radius, theta_desk + assist_moment_plot, theta2=-assist_moment_plot, color='white',  lw = 4))
        
        self.ax.arrow(
            left_coord[0],
            left_coord[1],
            F_left_plot[0],
            F_left_plot[1],
            width=0.03,
            color="yellow",
            label="human F",
        )
        self.ax.arrow(
            right_coord[0],
            right_coord[1],
            F_right_plot[0],
            F_right_plot[1],
            width=0.03,
            color="white",
            label="robot F",
        )
        desk.plot(self.ax)

        # draw actual observation
        self.ax.arrow(
            left_coord[0],
            left_coord[1],
            0.5 * np.cos(yaw),
            0.5 * np.sin(yaw),
            width=0.06,
            color="red",
        )
        # fit a distribution
        pf_mean = pf.mean_state
        pf_cov = pf.cov_state

        x1, x2, p_sum = generate_surface(pf_mean.reshape(-1, 1), pf_cov, 2)
        self.ax.contourf(x1, x2, p_sum, zorder=0, alpha=1.0)
        # plt.show()
        # break

        # draw individual particles
        if PLOT_PARTICLES:
            for i in range(len(pf.original_particles)):
                xa, ya, yawa= pf.original_particles[i, :3]
                w = pf.weights_vis[i]
                self.ax.scatter(xa, ya, color="red", s=w * 2000)
                self.ax.arrow(xa, ya, 
                            0.2 * np.cos(yawa),
                            0.2 * np.sin(yawa),
                            width=0.1*w,
                            color="orange")
        self.ax.set_title("t = " + str(round(t,2)))

        # draw ds traj
        # for i in range(len(pf.hypotheses)):
        #     A_mat = pf.original_particles[i,2:].reshape(2,2)
        #     traj_xy =
        #     ax.plot(traj_xy[:,0], traj_xy[:,1])

        # draw obstacle
        # obs1 = plt.Circle((obstable_position[0], obstable_position[1]), 0.1, color='r')
        # ax.add_patch(obs1)

        self.ax.scatter(xyt[0], xyt[1], c="yellow")
        self.ax.legend(loc="upper left")
        self.ax.set_xlim([-0.2, 1.3])
        self.ax.set_ylim([-1.1, 1.1])
        self.ax.set_aspect(1)

        plt.savefig("res_img/" + str(self.curr_idx) + ".png")
        plt.cla()
        plt.clf()
        plt.close()

class RealTimePlot:
    def __init__(self, desk, pf):
        self.fig = plt.figure(figsize=(20,20))
        self.ax = self.fig.add_subplot(1, 3, 1)
        #create a second subplot for data history
        self.ax2 = self.fig.add_subplot(1, 3, 2)
        self.ax3 = self.fig.add_subplot(1, 3, 3)
        self.ax.set_aspect('equal')
        self.ax.set_xlim(-0.2, 1.3)
        self.ax.set_ylim(-1.1, 1.1)

        self.i = 0

        self.timer_start = time.time()
        self.time_to_goal = []
        self.effort_temp = np.zeros((3,1))
        self.effort_to_goal = []
        self.goal = self.gen_new_goal()

        plt.show(block=False)
        plt.draw()
        self.background = self.fig.canvas.copy_from_bbox(self.ax.bbox)

        left_coord, right_coord = desk.get_left_right_coordinate()
        self.human_points = self.ax.plot(left_coord[0], left_coord[1], 'o')[0]
        self.robot_points = self.ax.plot(right_coord[0], right_coord[1], 'o')[0]

        self.attractor_text = self.ax.text(0.5, 0.5, "attractor", fontsize=12, color="red")
        self.attractor_lsq_text = self.ax.text(0.5, 0.5, "attractor", fontsize=12, color="black")
        rectangle = patches.Rectangle(
            (-desk.a / 2 + desk.pos[0], -desk.b / 2 + desk.pos[1]),
            desk.a,
            desk.b,
            linewidth=1,
            edgecolor="r",
            facecolor="none",
            rotation_point="center",
            angle=math.degrees(desk.pos[-1]),
            zorder=1
        )
        self.desk_patch = self.ax.add_patch(rectangle,)

        # plot a random pose rectangle for goal pose in yellow
        rectangle_goal = patches.Rectangle(
            (-desk.a / 2 + self.goal[0], -desk.b / 2 + self.goal[1]),
            desk.a,
            desk.b,
            linewidth=1,
            edgecolor="#FFFB42",
            facecolor="#FFFB42",
            rotation_point="center",
            angle=math.degrees(self.goal[2]),
            zorder=0 #for showing table on top of goal
        )
        self.goal_patch = self.ax.add_patch(rectangle_goal)


        self.human_force = self.ax.arrow(
            left_coord[0],
            left_coord[1],
            0.0,
            0.0,
            width=0.03,
            color="blue",
            label="human F",
        )
        self.human_torque = self.ax.arrow(
            left_coord[0],
            left_coord[1],
            0.0,
            0.0,
            width=0.03,
            color="cornflowerblue",
            label="human T",
        )


        self.robot_force = self.ax.arrow(
            right_coord[0],
            right_coord[1],
            0.0,
            0.0,
            width=0.03,
            color="green",
            label="robot F",
        )

        self.robot_torque = self.ax.arrow(
            right_coord[0],
            right_coord[1],
            0.0,
            0.0,
            width=0.03,
            color="limegreen",
            label="robot F",
        )

        self.attractor =  self.ax.arrow(
            right_coord[0],
            right_coord[1],
            0.0,
            0.0,
            width=0.03,
            color="red",
            label="attractor",
        )

        self.attractor_lsq =  self.ax.arrow(
            right_coord[0],
            right_coord[1],
            0.0,
            0.0,
            width=0.03,
            color="black",
            label="attractor",
        )

        if PLOT_PARTICLES:
            self.particle_arrow = []
            for i in range(len(pf.original_particles)):
                self.particle_arrow.append( self.ax.arrow(pf.original_particles[i, 0], pf.original_particles[i, 1],
                                            0.2 * np.cos(pf.original_particles[i, 2]),
                                            0.2 * np.sin(pf.original_particles[i, 2]),
                                            width=0.01,
                                            color="orange"))
                

            self.particle_pt = self.ax.scatter(pf.original_particles[:, 0], pf.original_particles[:, 1], color="red", s=0.0)
        
        #plot xyt history
        self.xyt_hist_plot = self.ax.scatter(0,0, color = "red", s=0.0)
        

        self.data_hist_plot = self.ax2.scatter(0, 0, color="red", s=0.0)
        self.data_hist = np.zeros((0,2))
        self.attractor_hist_plot = self.ax2.scatter(0, 0, color="red", s=0.0)
        self.attractor_hist = np.zeros((0,4))

        self.tank_state_plot = self.ax3.scatter(0, 0, color="red", s=0.0)
        self.tank_state = np.zeros((0,2))


    def wrap_to_pi_over_2(self, angle):
        while angle > np.pi/2:
            angle -= np.pi
        while angle < -np.pi/2:
            angle += np.pi
        return angle
    
    def gen_new_goal(self):
        return (random.uniform(-2.5, 2.5), random.uniform(-2.5, 2.5), random.uniform(-np.pi/2, np.pi/2))

    def plot_realtime(self, desk, pf, F_human, F_assist, attractor, A_est, x_negotiate, xyt_hist, goal_and_eig, tank_state):
        self.i += 1
        self.effort_temp += F_human*DT
        F_assist_plot = F_assist.reshape(-1) / 100
        F_human_plot = F_human.reshape(-1) / 100
        left_coord, right_coord = desk.get_left_right_coordinate()
        self.human_points.set_data(left_coord[0], left_coord[1])
        self.robot_points.set_data(right_coord[0], right_coord[1])
        self.attractor.set_data(x=attractor[0], y=attractor[1], dx=0.2*np.cos(attractor[2]), dy=0.2*np.sin(attractor[2]))
        self.attractor_text.set_position((attractor[0], attractor[1]+0.3))

        # print(goal_and_eig)
        self.attractor_lsq.set_data(x=goal_and_eig[0][0], y=goal_and_eig[0][1], dx=0.2*np.cos(goal_and_eig[0][2]), dy=0.2*np.sin(goal_and_eig[0][2]))
        
        self.attractor_lsq_text.set_position((goal_and_eig[0][0], goal_and_eig[0][1]+0.3))
        self.attractor_lsq_text.set_text("{:.2g}".format(goal_and_eig[1][0]) + ", " + "{:.2g}".format(goal_and_eig[1][1]) + ", " + "{:.2g}".format(goal_and_eig[1][2]))

        formatted_ax = "{:.2g}".format(A_est[0,0])
        formatted_ay = "{:.2g}".format(A_est[1,1])
        formatted_atheta = "{:.2g}".format(A_est[2,2])
        self.attractor_text.set_text( formatted_ax + ", " + formatted_ay + ", " + formatted_atheta)
        self.human_force.set_data(x=left_coord[0], y=left_coord[1], dx=F_human_plot[0], dy=F_human_plot[1])
        self.robot_force.set_data(x=right_coord[0], y=right_coord[1], dx=F_assist_plot[0], dy=F_assist_plot[1])

        self.human_torque.set_data(x=left_coord[0], y=left_coord[1], dx=-F_human_plot[2], dy=0)
        self.robot_torque.set_data(x=right_coord[0], y=right_coord[1], dx=-F_assist_plot[2], dy=0)
        
        if PLOT_PARTICLES:
            for i in range(len(pf.original_particles)):
                self.particle_arrow[i].set_data(x=pf.original_particles[i, 0], y=pf.original_particles[i, 1], dx=0.03 * np.cos(pf.original_particles[i, 2]), dy=0.03* np.sin(pf.original_particles[i, 2]))

        self.desk_patch.set_x(-desk.a / 2 + desk.pos[0])
        self.desk_patch.set_y(-desk.b / 2 + desk.pos[1])
        self.desk_patch.set_angle(math.degrees(desk.pos[-1]))

        #if desk pose is close to goal pose, generate a new random goal pose
        theta_test = self.wrap_to_pi_over_2(desk.pos[-1])
        if (abs(desk.pos[0] - self.goal[0]) < 0.2) and (abs(desk.pos[1] - self.goal[1]) < 0.2) and (abs(theta_test - self.goal[2]) < 0.2): #\
            # and abs(desk.vx) < 0.2 and abs(desk.vy) < 0.2 and abs(desk.omega) < 0.2:
            self.time_to_goal.append(time.time() - self.timer_start)
            self.effort_to_goal.append(self.effort_temp)
            self.effort_temp = np.zeros((3,1))
            self.timer_start = time.time()
            self.goal = self.gen_new_goal()
            print(self.effort_to_goal)
            print(self.time_to_goal)
            self.goal_patch.set_x(-desk.a / 2 + self.goal[0])
            self.goal_patch.set_y(-desk.b / 2 + self.goal[1])
            self.goal_patch.set_angle(math.degrees(self.goal[2]))

        self.desk_patch.set_x(-desk.a / 2 + desk.pos[0])
        self.desk_patch.set_y(-desk.b / 2 + desk.pos[1])
        self.desk_patch.set_angle(math.degrees(desk.pos[-1]))

        #update data history
        self.attractor_hist = np.vstack((self.attractor_hist, np.append(attractor.reshape(-1),self.i)))
        self.attractor_hist_plot.remove()
        self.attractor_hist_plot = self.ax2.scatter(self.attractor_hist[:,-1],self.attractor_hist[:,0], color="black")
        # print(x_negotiate)
        self.data_hist = np.vstack((self.data_hist, np.array([self.i, x_negotiate[0][0]])))
        self.data_hist_plot.remove()
        self.data_hist_plot = self.ax2.scatter(self.data_hist[:,0], self.data_hist[:,1], color="red")
        #if desk pose is close to goal pose, generate a new random goal pose
        theta_test = self.wrap_to_pi_over_2(desk.pos[-1])

        self.xyt_hist_plot.remove()
        self.xyt_hist_plot = self.ax.scatter(xyt_hist[:,0], xyt_hist[:,1], color="red", s=0.5)

        self.tank_state = np.vstack((self.tank_state, np.array([self.i, tank_state])))
        self.tank_state_plot.remove()
        self.tank_state_plot = self.ax3.scatter(self.tank_state[:,0], self.tank_state[:,1], color="red", s=10)

        if (abs(desk.pos[0] - self.goal[0]) < 0.2) and (abs(desk.pos[1] - self.goal[1]) < 0.2) and (abs(theta_test - self.goal[2]) < 0.2): #\
            # and abs(desk.vx) < 0.2 and abs(desk.vy) < 0.2 and abs(desk.omega) < 0.2:
            self.time_to_goal.append(time.time() - self.timer_start)
            self.effort_to_goal.append(self.effort_temp)
            self.effort_temp = np.zeros((3,1))
            self.timer_start = time.time()
            self.goal = self.gen_new_goal()
            print(self.effort_to_goal)
            print(self.time_to_goal)
            self.goal_patch.set_x(-desk.a / 2 + self.goal[0])
            self.goal_patch.set_y(-desk.b / 2 + self.goal[1])
            self.goal_patch.set_angle(math.degrees(self.goal[2]))



        plt.pause(DT)
