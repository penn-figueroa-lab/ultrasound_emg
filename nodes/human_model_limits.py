#!/usr/bin/env python3

# A simple dummy joint state publisher that resets the robot's state to the home state
#
#
#
#

# import roslib
import rospy
import sensor_msgs.msg
# import geometry_msgs.msg
import numpy as np


def get_human_parameters():
    joint_states.name=['jL5S1_rotx', 'jL5S1_roty', 'jL4L3_rotx', 'jL4L3_roty',  'jL1T12_rotx', 'jL1T12_roty',
    'jT9T8_rotx', 'jT9T8_roty', 'jT9T8_rotz', 'jT1C7_rotx', 'jT1C7_roty', 'jT1C7_rotz',
    'jC1Head_rotx', 'jC1Head_roty', 
    'jRightC7Shoulder_rotx', 'jRightShoulder_rotx', 'jRightShoulder_roty', 'jRightShoulder_rotz', 'jRightElbow_roty', 'jRightElbow_rotz', 'jRightWrist_rotx', 'jRightWrist_rotz',
    'jLeftC7Shoulder_rotx', 'jLeftShoulder_rotx', 'jLeftShoulder_roty', 'jLeftShoulder_rotz', 'jLeftElbow_roty', 'jLeftElbow_rotz', 'jLeftWrist_rotx', 'jLeftWrist_rotz',
    'jRightHip_rotx', 'jRightHip_roty', 'jRightHip_rotz', 'jRightKnee_roty', 'jRightKnee_rotz', 'jRightAnkle_rotx', 'jRightAnkle_roty', 'jRightAnkle_rotz', 'jRightBallFoot_roty',
    'jLeftHip_rotx','jLeftHip_roty','jLeftHip_rotz', 'jLeftKnee_roty', 'jLeftKnee_rotz', 'jLeftAnkle_rotx', 'jLeftAnkle_roty', 'jLeftAnkle_rotz', 'jLeftBallFoot_roty']
    RIGHT_ARM = ['jRightC7Shoulder_rotx', 'jRightShoulder_rotx', 'jRightShoulder_roty', 'jRightShoulder_rotz', 
                 'jRightElbow_roty', 'jRightElbow_rotz', 'jRightWrist_rotx', 'jRightWrist_rotz']
    LEFT_ARM = ['jLeftC7Shoulder_rotx', 'jLeftShoulder_rotx', 'jLeftShoulder_roty', 'jLeftShoulder_rotz', 
                 'jLeftElbow_roty', 'jLeftElbow_rotz', 'jLeftWrist_rotx', 'jLeftWrist_rotz']
    LEFT_ARM_LINKS = ['leftShoulderPitchLink', 'leftShoulderRollLink', 'leftShoulderYawLink',
                      'leftElbowPitchLink', 'leftForearmLink', 'leftWristRollLink', 'leftPalm']
    NECK = ['lowerNeckPitch', 'neckYaw', 'upperNeckPitch']
    TORSO = ['torsoYaw', 'torsoPitch', 'torsoRoll']

    RIGHT_ARM_MAX_POSITION_LIMITS = [2.0, 1.51, 2.18, 2.174, 3.14, 0.62, 0.36]
    LEFT_ARM_MAX_POSITION_LIMITS = [2.0, 1.26, 2.17, 0.12, 3.13, 0.625, 0.49]
    TORSO_MAX_POSITION_LIMITS = [1.181, 0.666, 0.255]
    RIGHT_ARM_MIN_POSITION_LIMITS = [-2.85, -1.26, -3.1, -0.12, -2.019, -0.625, -0.49]
    LEFT_ARM_MIN_POSITION_LIMITS = [-2.85, -1.51, -3.09, -2.174, -2.018, -0.62, -0.36]
    TORSO_MIN_POSITION_LIMITS = [-1.329, -0.13, -0.23]

    RIGHT_LEG_MAX_POSITION_LIMITS = [0.41, 0.467, 1.619, 2.057, 0.875, 0.348]
    RIGHT_LEG_MIN_POSITION_LIMITS = [-1.1, -0.5515, -2.42, -0.083, -0.86, -0.349]

    NECK_MAX = [1.162, 1.047, 0.01]
    NECK_MIN = [-0.02, -1.047, -0.872]

    JOINT_NAMES = [LEFT_ARM, RIGHT_ARM, TORSO, NECK]
    JOINT_POSITION_MAX = np.array([LEFT_ARM_MAX_POSITION_LIMITS,
                                   RIGHT_ARM_MAX_POSITION_LIMITS,
                                   TORSO_MAX_POSITION_LIMITS,
                                   NECK_MAX])
    JOINT_POSITION_MIN = np.array([LEFT_ARM_MIN_POSITION_LIMITS,
                                   RIGHT_ARM_MIN_POSITION_LIMITS,
                                   TORSO_MIN_POSITION_LIMITS,
                                   NECK_MIN])

    LEFT_ARM_MAX_VELOCITY_LIMITS = np.array([5.89, 5.89, 11.5, 11.5, 5, 1, 1]) * 0.5
    LEFT_ARM_MIN_VELOCITY_LIMITS = -LEFT_ARM_MAX_VELOCITY_LIMITS

    DANGER_VALUE = 60
    a1 = 100
    b1 = 10

    Tne = np.eye(4)
    Tne[1, 3] = 0.08

    lf_leftSupport = np.array([[0.16, 0.1, 0.],
                               [0.16, -0.1, 0.],
                               [-0.08, -0.1, 0.],
                               [-0.08, 0.1, 0.]])

    RobotChain = {
        'RIGHT_ARM_MAX_POSITION_LIMITS': RIGHT_ARM_MAX_POSITION_LIMITS,
        'LEFT_ARM_MAX_POSITION_LIMITS': LEFT_ARM_MAX_POSITION_LIMITS,
        'TORSO_MAX_POSITION_LIMITS': TORSO_MAX_POSITION_LIMITS,
        'RIGHT_ARM_MIN_POSITION_LIMITS': RIGHT_ARM_MIN_POSITION_LIMITS,
        'LEFT_ARM_MIN_POSITION_LIMITS': LEFT_ARM_MIN_POSITION_LIMITS,
        'TORSO_MIN_POSITION_LIMITS': TORSO_MIN_POSITION_LIMITS,
        'RIGHT_LEG_MAX_POSITION_LIMITS': RIGHT_LEG_MAX_POSITION_LIMITS,
        'RIGHT_LEG_MIN_POSITION_LIMITS': RIGHT_LEG_MIN_POSITION_LIMITS,
        'TORSO': TORSO,
        'LEFT_ARM': LEFT_ARM,
        'LEFT_ARM_LINKS': LEFT_ARM_LINKS,
        'RIGHT_ARM': RIGHT_ARM,
        'JOINT_NAMES': JOINT_NAMES,
        'JOINT_POSITION_MAX': JOINT_POSITION_MAX,
        'JOINT_POSITION_MIN': JOINT_POSITION_MIN,
        'LEFT_ARM_MAX_VELOCITY_LIMITS': LEFT_ARM_MAX_VELOCITY_LIMITS,
        'LEFT_ARM_MIN_VELOCITY_LIMITS': LEFT_ARM_MIN_VELOCITY_LIMITS,
        'Tne': Tne,
        'lf_leftSupport': lf_leftSupport,
    }

    gains = {
        'a1': a1,
        'b1': b1,
        'DANGER_VALUE': DANGER_VALUE,
    }

    return RobotChain, gains

if __name__ == '__main__':
    rospy.init_node('human_chain_publisher')
    # joint_timeout=rospy.get_param('~reset_time',5.0)

    pub = rospy.Publisher('human_chains', sensor_msgs.msg.JointState, queue_size=10)
    joint_states=sensor_msgs.msg.JointState()
    joint_states.header.frame_id='Pelvis'
    joint_states.name=['jL5S1_rotx', 'jL5S1_roty', 'jL4L3_rotx', 'jL4L3_roty',  'jL1T12_rotx', 'jL1T12_roty',
    'jT9T8_rotx', 'jT9T8_roty', 'jT9T8_rotz', 'jT1C7_rotx', 'jT1C7_roty', 'jT1C7_rotz',
    'jC1Head_rotx', 'jC1Head_roty', 
    'jRightC7Shoulder_rotx', 'jRightShoulder_rotx', 'jRightShoulder_roty', 'jRightShoulder_rotz', 'jRightElbow_roty', 'jRightElbow_rotz', 'jRightWrist_rotx', 'jRightWrist_rotz',
    'jLeftC7Shoulder_rotx', 'jLeftShoulder_rotx', 'jLeftShoulder_roty', 'jLeftShoulder_rotz', 'jLeftElbow_roty', 'jLeftElbow_rotz', 'jLeftWrist_rotx', 'jLeftWrist_rotz',
    'jRightHip_rotx', 'jRightHip_roty', 'jRightHip_rotz', 'jRightKnee_roty', 'jRightKnee_rotz', 'jRightAnkle_rotx', 'jRightAnkle_roty', 'jRightAnkle_rotz', 'jRightBallFoot_roty',
    'jLeftHip_rotx','jLeftHip_roty','jLeftHip_rotz', 'jLeftKnee_roty', 'jLeftKnee_rotz', 'jLeftAnkle_rotx', 'jLeftAnkle_roty', 'jLeftAnkle_rotz', 'jLeftBallFoot_roty']

    joint_states.position=[0, 0, 0, 0, 0, 0,    
                           0, 0, 0, 0, 0, 0,    
                           0, 0,                        #Head
                           0, 0, 0, 0, 0, 0, 0, 0,      #Right Arm
                           0, 0, 0, 0, 0, 1, 0, 0,      #Left Arm
                           0, 0, 0, 0, 0, 0, 0, 0, 0,   #Right Leg
                           0, 0, 0, 0, 0, 0, 0, 0, 0,]  #Left Leg

    joint_states.velocity=[]

    joint_states.effort=[]

    while not rospy.is_shutdown():
        joint_states.header.seq=joint_states.header.seq+1
        joint_states.header.stamp=rospy.Time.now()
        pub.publish(joint_states)
        # rospy.sleep(joint_timeout)
    rospy.spin()

# Example usage:
RobotChain, gains = get_human_parameters()
print(RobotChain)
print(gains)
