#!/usr/bin/env python3

# A simple human joint state publisher that resets the human's state to the home config
#
#
#
#

# import roslib
import rospy
import sensor_msgs.msg
# import geometry_msgs.msg
from math import pi

if __name__ == '__main__':
    rospy.init_node('human_joint_publisher')
    # joint_timeout=rospy.get_param('~reset_time',5.0)

    pub = rospy.Publisher('/human/joint_states', sensor_msgs.msg.JointState, queue_size=10)
    joint_states=sensor_msgs.msg.JointState()
    joint_states.header.frame_id='/world'
    joint_states.name=[ #'static_offset_mocap',#'Hip_x', 'Hip_y', 'Hip_z', 'Hip_rotz', 'Hip_rotx', 'Hip_roty', 
    'Ab_rotx', 'Ab_roty', 'Ab_rotz', 'Chest_rotx', 'Chest_roty','Chest_rotz', 
    'Neck_rotx', 'Neck_roty', 'Neck_rotz',  'Head_rotx', 'Head_roty', 'Head_rotz',
    'jRightC7Shoulder_rotx', 'jRightC7Shoulder_roty', 'jRightC7Shoulder_rotz', 'jRightShoulder_rotx', 'jRightShoulder_roty', 'jRightShoulder_rotz', 
    'jRightElbow_rotx', 'jRightElbow_roty', 'jRightElbow_rotz', 'jRightWrist_rotx', 'jRightWrist_roty', 'jRightWrist_rotz',
    'jLeftC7Shoulder_rotx', 'jLeftC7Shoulder_roty', 'jLeftC7Shoulder_rotz', 'jLeftShoulder_rotx', 'jLeftShoulder_roty', 'jLeftShoulder_rotz', 
    'jLeftElbow_rotx', 'jLeftElbow_roty', 'jLeftElbow_rotz', 'jLeftWrist_rotx', 'jLeftWrist_roty', 'jLeftWrist_rotz']

    # joint_states.position=[1.5,0.5, 0.5, 0, 0, pi,            #Hip
    #                        0, 0, 0, 0, 0, 0,            #Ab, Chest
    #                        0, 0, 0, 0, 0, 0,            #Neck, Head
    #                        0, 0, 0, 0, 0, 0, 0, 0,      #Right Arm
    #                        0, 0, pi/3, -pi/2, -pi/2, -pi/4, 0, 0]      #Left Arm

    joint_states.position=[ 0, 0, 0, 0, 0, 0,            #Ab, Chest
                            0, 0, 0, 0, 0, 0,            #Neck, Head
                            0, 0, 0, 0, 0, 0,            #Right shoulder/upperarm
                            0, 0, 0, 0, 0, 0,            #Right forearm/hand
                            0, 0, 0, 0, pi/3, -pi/2,            # Left shoulder/upperarm
                            pi/2, -pi/4, 0, 0, 0, 0]            # Left forearm/hand

    joint_states.velocity=[]

    joint_states.effort=[]

    r = rospy.Rate(10) 
    while not rospy.is_shutdown():
        joint_states.header.seq=joint_states.header.seq+1
        joint_states.header.stamp=rospy.Time.now()
        pub.publish(joint_states)
        r.sleep()
    rospy.spin()
