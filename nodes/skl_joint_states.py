#!/usr/bin/env python3

# Human joint state publisher from mocap data
import numpy as np
# import roslib
import rospy
import sensor_msgs.msg
from geometry_msgs.msg import Pose, PoseArray, PoseStamped
from math import pi
from tf.transformations import euler_from_quaternion, quaternion_from_euler, quaternion_matrix
from visualization_msgs.msg import Marker
#MARKER ARRAY
from visualization_msgs.msg import MarkerArray
fixed_frame_name = "/world"
MOCAP = True

def make_text(text, attractor, id):
    marker = Marker()
    marker.header.frame_id = "world"
    marker.header.stamp = rospy.Time.now()
    marker.type = marker.TEXT_VIEW_FACING
    marker.action = marker.ADD
    marker.pose.position.x = attractor[0]
    marker.pose.position.y = attractor[1]
    marker.pose.position.z = attractor[2] # yaw
    marker.scale.z = 0.05
    marker.text = text
    marker.color.a = 1.0  # Alpha
    # white text  
    marker.color.r = 1.0  # Red
    marker.color.g = 1.0  # Green
    marker.color.b = 1.0  # Blue
    marker.id = id
    return marker


class SKL:

    def __init__(self, dt) -> None:
        self.dt = dt
        self.time_start = rospy.Time.now()
        self.time_last_update = rospy.Time.now()
        self.counter = 10
        self.counter_save = 10
        # self.last_iteration_time == None

        skl_bones_num = 51
        self.kuka_pos = np.zeros((1,3))
        self.bone_pose = np.zeros((skl_bones_num,7))
        self.bone_pos = np.zeros((skl_bones_num,3))
        self.bone_rot = np.zeros((skl_bones_num,3))
        self.joints = [] #np.zeros((1,))
        self.joints_list = []

        self.natnet_skl_sub = rospy.Subscriber('/natnet_ros/fullbody/pose', PoseArray, self.skl_callback, tcp_nodelay=True)
        self.skl_joint_state_pub = rospy.Publisher('/human/joint_states', sensor_msgs.msg.JointState, queue_size=10)
        self.text_pub = rospy.Publisher('/human/vis/text', MarkerArray, queue_size=10)

        # self.start = False

    def skl_callback(self,msg):
        '''
        0 hip
        1 abdomen
        2 chest
        22 neck
        23 head
        3 r_shoulder
        4 r_u_arm
        5 r_f_arm
        6 r_hand
        24 l_shoulder
        25 l_u_arm
        26 l_f_arm
        27 l_hand
        
        '''
        MarkerArr = MarkerArray()

        for idx, bone in enumerate(msg.poses):               
            # print(idx, bone)
            self.bone_pose[idx,0] = bone.position.x 
            self.bone_pose[idx,1] = bone.position.y
            self.bone_pose[idx,2] = bone.position.z
            self.bone_pose[idx,3] = bone.orientation.x
            self.bone_pose[idx,4] = bone.orientation.y
            self.bone_pose[idx,5] = bone.orientation.z
            self.bone_pose[idx,6] = bone.orientation.w
            marker_idx = make_text (str(idx), [bone.position.x, bone.position.y, bone.position.z], idx)
            MarkerArr.markers.append(marker_idx)

            (ang1, ang2, ang3) = euler_from_quaternion(self.bone_pose[idx,3:], axes='rxyz') # ???
            # rospy.loginfo_throttle(5, str(self.bone_pose[idx,0]) + " " + str(self.bone_pose[idx,1]) + " " + str(self.bone_pose[idx,2]) + " " + str(self.bone_pose[idx,3]) + " " + str(self.bone_pose[idx,4]) + " " + str(self.bone_pose[idx,5]) + " " + str(self.bone_pose[idx,6]) )

            self.bone_pos[idx,0] = bone.position.x
            self.bone_pos[idx,1] = bone.position.y
            self.bone_pos[idx,2] = bone.position.z
            self.bone_rot[idx,0] = -ang2 # ang3 # ###????
            self.bone_rot[idx,1] = ang1
            self.bone_rot[idx,2] = ang3
            # rospy.loginfo ("idx: %s      bone_rot:  %s", idx, [round(180/pi*element, 2) for element in self.bone_rot[idx,:]] )
        self.text_pub.publish(MarkerArr)
        
        # self.bone_pos[0,:] -= self.kuka_pos[0,:]
        # self.joints.append (self.bone_pos[0,:])     # hip_position
        # self.joints.append (self.bone_rot[0,:])     # hip_orientation
        # body_part_string = ['ab', 'chest', 'neck', 'head', 'r_shoulder', 'r_u_arm', 'r_f_arm', 'r_hand', 'l_shoulder', 'l_u_arm', 'l_f_arm', 'l_hand']
        # body_part_number = [1, 2, 3, 4, 24, 25, 26, 27, 5, 6, 7, 8]
        # make this print every 5 sec
        # time_now = rospy.Time.now()
        # if (time_now - self.time_last_update).to_sec() > 3:
        #     self.time_last_update = time_now
        #     for idx, idx_number in enumerate(body_part_number):
        #         pass
        #         rospy.loginfo_throttle(5, str(body_part_string[idx]) + " " + str(self.bone_rot[idx_number,:]) )
        #         # self.joints.append (self.bone_rot[idx_number,:])     # hip_orientation
        
        self.joints = []
        self.joints.append (self.bone_rot[1,:])     # Ab  
        self.joints.append (self.bone_rot[2,:])     # Chest 
        self.joints.append (self.bone_rot[3,:])     # Neck 
        self.joints.append (self.bone_rot[4,:])     # Head 
        self.joints.append (self.bone_rot[24,:])    # Right Arm 
        self.joints.append (self.bone_rot[25,:])         # upper_arm
        # print("\n \n Right sh: ", [round(element, 2) for element in self.bone_rot[25,:]] )
        self.joints.append (self.bone_rot[26,:])     # forearm
        # print("\n \n Right el: ", [round(element, 2) for element in self.bone_rot[26,:]] )
        self.joints.append (self.bone_rot[27,:])     # hand
        # print("\n \n Right wr: ", [round(element, 2) for element in self.bone_rot[26,:]] )
        self.joints.append (self.bone_rot[5,:])     # Left Arm
        self.joints.append (self.bone_rot[6,:])          # upper_arm
        # print("\n \n Left sh: ", [round(element, 2) for element in self.bone_rot[6,:]] )
        self.joints.append (self.bone_rot[7,:])         # forearm
        # print("\n \n Left el: ", [round(element, 2) for element in self.bone_rot[7,:]] )
        self.joints.append (self.bone_rot[8,:])         # hand

        self.joints_list=[]
        for element in self.joints:
            if isinstance(element, np.ndarray):
                self.joints_list.extend(element.tolist())
            else:
                self.joints_list.append(element)
        # print("\n \n joints: ", [round(element, 2) for element in self.joints_list] )
        # print(len(self.joints_list))


    def skl_joint_states(self):

        joint_states=sensor_msgs.msg.JointState()
        joint_states.header.stamp = rospy.Time.now()
        joint_states.header.seq=joint_states.header.seq+1
        joint_states.header.frame_id=fixed_frame_name
        joint_states.name=[#'static_offset_mocap',
        'Ab_rotx', 'Ab_roty', 'Ab_rotz', 'Chest_rotx', 'Chest_roty','Chest_rotz', 
        'Neck_rotx', 'Neck_roty', 'Neck_rotz',  'Head_rotx', 'Head_roty', 'Head_rotz',
        'jRightC7Shoulder_rotx', 'jRightC7Shoulder_roty', 'jRightC7Shoulder_rotz', 'jRightShoulder_rotx', 'jRightShoulder_roty', 'jRightShoulder_rotz', 
        'jRightElbow_rotx', 'jRightElbow_roty', 'jRightElbow_rotz', 'jRightWrist_rotx', 'jRightWrist_roty', 'jRightWrist_rotz',
        'jLeftC7Shoulder_rotx', 'jLeftC7Shoulder_roty', 'jLeftC7Shoulder_rotz', 'jLeftShoulder_rotx', 'jLeftShoulder_roty', 'jLeftShoulder_rotz', 
        'jLeftElbow_rotx', 'jLeftElbow_roty', 'jLeftElbow_rotz', 'jLeftWrist_rotx', 'jLeftWrist_roty', 'jLeftWrist_rotz']

        if MOCAP:
            joint_states.position=self.joints_list

        else:
            # joint_states.position=[ 0, 0, 0, 0, 0, 0, 0,            #Ab, Chest
            #                         0, 0, 0, 0, 0, 0,            #Neck, Head
            #                         0, 0, 0, 0, 0, 0,            #Right shoulder/upperarm
            #                         0, 0, 0, 0, 0, 0,            #Right forearm/hand
            #                         0, 0, pi/3, -pi/2, -pi/2, -pi/4, 0, 0]      #Left Arm
            
            joint_states.position=[ 0, 0, 0, 0, 0, 0,            #Ab, Chest
                                    0, 0, 0, 0, 0, 0,            #Neck, Head
                                    0, 0, 0, 0, 0, 0,            #Right shoulder/upperarm
                                    0, 0, 0, 0, 0, 0,            #Right forearm/hand
                                    0, 0, 0, 0, pi/3, -pi/2,            # Left shoulder/upperarm
                                    pi/2, -pi/4, 0, 0, 0, 0]            # Left forearm/hand

        # print(len(position))
        # print(joint_states.position)
        joint_states.velocity=[]

        joint_states.effort=[]

        # self.skl_joint_state_pub.publish(joint_states)
        self.skl_joint_state_pub.publish(joint_states)


def main():
    rospy.init_node('skl_joint_publisher')
    freq=10
    r = rospy.Rate(freq)
    skl = SKL(1/freq)

    while not rospy.is_shutdown():
        skl.skl_joint_states()
        r.sleep()
    rospy.spin()
         

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass

