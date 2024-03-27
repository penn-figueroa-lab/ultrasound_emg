import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt


def linearize_func(fun, x):
    """
    Linearize a function using the complex step differentiation method to compute the Jacobian.

    Parameters:
    fun (callable): A Python callable that can take a NumPy array as input and returns a NumPy array.
    x (np.array): A 1-D NumPy array at which to linearize the function.

    Returns:
    np.array: The Jacobian matrix of the function evaluated at x.
    """
    # LinearizeFunc = @(fun,x) imag(fun(x(:,ones(1,numel(x)))+(numel(x)*eps)*1i*eye(numel(x))))/(numel(x)*eps);
    n = x.shape[0]
    step_size = 1e-8
    x_matrix = np.tile(x, (1,n))
    complex_step = step_size * 1j * np.eye(n)
    jacobian = np.imag(fun(x_matrix + complex_step)) / step_size
    # Return the imaginary part of the evaluation divided by the step (Jacobian)
    return jacobian
class NegotiationController:
    def __init__(self, dt,  target_human, target_robot):
        self.dt = dt
        self.target_human = target_human
        self.target_robot = target_robot
        self.A = np.array([[1, dt], [0, 1]])
        self.B = np.array([[0], [dt/1.0]])
        self.Q_human = np.diag([100, 0])  # Placeholder for the Q matrix for the human
        self.Q_robot = np.diag([500, 0])  # Placeholder for the Q matrix for the robot
        self.R_human = 0.1
        self.R_robot = 0.1
        self.L_human = self.dlqr(self.A, self.B, self.Q_human, self.R_human)
        print("L_Human" ,self.L_human)
        self.L_robot = self.dlqr(self.A, self.B, self.Q_robot, self.R_robot)
        print("L_Robot" ,self.L_robot)
        self.Q_kf = 10 * dt ** 2
        self.H = np.array([[0, 1]])
        self.R_kf = np.diag([0.001])
        self.x_negotiate = np.array([[0], [0]])
        self.u_robot_negotiate = 0.0
        self.u_human_negotiate = 0.0
        self.x_hat_human = np.array([[0], [0]])
        self.x_hat_robot = np.array([[0], [0]])
        self.P_human = np.eye(2)
        self.P_robot = np.eye(2)
        self.lambda_robot = 0.0



    def dlqr(self, A, B, Q, R):
        # Solve the discrete time lqr controller and return the gain matrix
        X = np.matrix(scipy.linalg.solve_discrete_are(A, B, Q, R))
        K = np.matrix(scipy.linalg.inv(B.T * X * B + R) * (B.T * X * A))
        return np.squeeze(np.asarray(K))


    def update_lambda_robot(self, lambda_robot_update):
        self.lambda_robot = lambda_robot_update

    def update_x_negotiate(self, target_human_update):

        self.target_human = target_human_update

        Lambda_robot = self.lambda_robot
        Lambda_human = 1.0 - Lambda_robot
        
        def internal_model_human(xTarget):
            return np.array([xTarget[0,:], (xTarget[0,:]-self.x_negotiate[0])*self.L_human[0] + (0-self.x_negotiate[1])*self.L_human[1]])

        
        # Linearize internal model for human
        # print(self.x_hat_human, "x_hat_human")
        A_temp = linearize_func(internal_model_human, self.x_hat_human)
        
        # Predict for human
        x_temp = internal_model_human(self.x_hat_human)
        self.P_human = A_temp @ self.P_human @ A_temp.T + self.Q_kf
        K_temp = self.P_human @ self.H.T @ np.linalg.inv(self.H @ self.P_human @ self.H.T + self.R_kf)
        
        # Correct for human
        z = self.u_robot_negotiate + np.sqrt(self.R_kf) * np.random.randn(self.R_kf.shape[0], 1)
        self.x_hat_human = x_temp + K_temp @ (z - self.H @ x_temp)
        self.P_human = (np.eye(A_temp.shape[0]) - K_temp @ self.H) @ self.P_human
        
        ########################################################################
        # TARGET ESTIMATED USING ROBOT CONTROLLER GAIN
        ########################################################################
        # Define internal model for robot
        # def internal_model_robot(x_target):
        #     return np.array([
        #         x_target[0],
        #         (x_target[0] - self.x_negotiate) * self.L_robot[0] + (0 - self.x_negotiate) * self.L_robot[1]
        #     ])
        
        def internal_model_robot(xTarget):
            return np.array([xTarget[0,:], (xTarget[0,:]-self.x_negotiate[0])*self.L_robot[0] + (0-self.x_negotiate[1])*self.L_robot[1]])
        # Linearize internal model for robot
        A_temp = linearize_func(internal_model_robot, self.x_hat_robot)
        
        # Predict for robot
        x_temp = internal_model_robot(self.x_hat_robot)
        self.P_robot = A_temp @ self.P_robot @ A_temp.T + self.Q_kf
        K_temp = self.P_robot @ self.H.T @ np.linalg.inv(self.H @ self.P_robot @ self.H.T + self.R_kf)
        
        # Correct for robot
        z = self.u_human_negotiate + np.sqrt(self.R_kf) * np.random.randn(self.R_kf.shape[0], 1)
        self.x_hat_robot= x_temp + K_temp @ (z - self.H @ x_temp)
        self.P_robot = (np.eye(A_temp.shape[0]) - K_temp @ self.H) @ self.P_robot
        
        ########################################################################
        
        # Target update
        self.target_robot_estimate = Lambda_robot * self.target_robot + (1 - Lambda_robot) * self.x_hat_robot[0]
        self.target_human_estimate = Lambda_human * self.target_human + (1 - Lambda_human) * self.x_hat_human[0]
        
        # Control policy
        self.u_robot_negotiate = -self.L_robot @ (self.x_negotiate - np.array([self.target_robot_estimate, [0]]))
        self.u_human_negotiate = -self.L_human @ (self.x_negotiate - np.array([self.target_human_estimate, [0]]))
        
        # Human and robot system update
        self.x_negotiate = self.A @ self.x_negotiate + self.B @ (self.u_human_negotiate + self.u_robot_negotiate).reshape([1,1])


        return self.x_negotiate, self.u_human_negotiate, self. u_robot_negotiate

# Example of how to instantiate and use the class:
if __name__ == "__main__":
    N_sim = 150
    dt = 0.01
    time_vec = np.arange(0, N_sim*dt, dt)
    sim = NegotiationController(dt=dt,  target_human=0.3, target_robot=-0.3)

    new_target_human = 0.3  # Let's say the human's target has been updated to 0.5
    x_record = np.zeros((N_sim, 2))
    u_human_record = np.zeros((N_sim, 1))
    u_robot_record = np.zeros((N_sim, 1))
    for i in range(N_sim):
        if i < 75:
            new_target_human = 0.3
        else:
            new_target_human = 0.5
        x_negotiate_updated, u_human, u_robot = sim.update_x_negotiate(new_target_human)
        x_record[i] = x_negotiate_updated.flatten()
        u_human_record[i] = u_human
        u_robot_record[i] = u_robot
    #plot
    ax = plt.gca()
    ax.plot(time_vec, x_record[:,0])
    #plot on the right axis
    ax2 = ax.twinx()

    ax2.plot(time_vec, u_human_record)
    ax2.plot(time_vec, u_robot_record)
    plt.show()