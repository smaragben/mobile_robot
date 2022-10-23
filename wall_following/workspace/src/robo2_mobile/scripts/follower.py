#!/usr/bin/env python3

# Robotics II 2021-2022
# ECE - NTUA
# Benetou Smaragda el18048
# Vatistas Andreas el18020

"""
Start ROS node to publish linear and angular velocities to mymobibot in order to perform wall following.
"""

# Ros handlers services and messages
import rospy, roslib
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState
from sensor_msgs.msg import Range
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Twist
#Math imports
from math import sin, cos, atan2, pi, sqrt
from numpy.linalg import inv, det, norm, pinv
import numpy as np
import time as t

# from tf.transformations import euler_from_quaternion
# from tf.transformations import quaternion_matrix
# matrix = quaternion_matrix([1, 0, 0, 0])

def quaternion_to_euler(w, x, y, z):
    """Converts quaternions with components w, x, y, z into a tuple (roll, pitch, yaw)"""
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x**2 + y**2)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (w * y - z * x)
    pitch = np.where(np.abs(sinp) >= 1, np.sign(sinp) * np.pi / 2, np.arcsin(sinp))

    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y**2 + z**2)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw

class mymobibot_follower():
    """Class to compute and publish joints positions"""
    def __init__(self,rate):

        self.desired_distance = 0.3
        self.front_desired_distance = 0.6
        self.procedure_step = 'init'

        # linear and angular velocity
        self.velocity = Twist()
        # joints' states
        self.joint_states = JointState()
        # Sensors
        self.imu = Imu()
        self.imu_yaw = 0.0 # (-pi, pi]
        self.sonar_F = Range()
        self.sonar_FL = Range()
        self.sonar_FR = Range()
        self.sonar_L = Range()
        self.sonar_R = Range()

        # ROS SETUP
        # initialize subscribers for reading encoders and publishers for performing position control in the joint-space
        # Robot
        self.velocity_pub = rospy.Publisher('/mymobibot/cmd_vel', Twist, queue_size=1)
        self.joint_states_sub = rospy.Subscriber('/mymobibot/joint_states', JointState, self.joint_states_callback, queue_size=1)
        # Sensors
        self.imu_sub = rospy.Subscriber('/imu', Imu, self.imu_callback, queue_size=1)
        self.sonar_front_sub = rospy.Subscriber('/sensor/sonar_F', Range, self.sonar_front_callback, queue_size=1)
        self.sonar_frontleft_sub = rospy.Subscriber('/sensor/sonar_FL', Range, self.sonar_frontleft_callback, queue_size=1)
        self.sonar_frontright_sub = rospy.Subscriber('/sensor/sonar_FR', Range, self.sonar_frontright_callback, queue_size=1)
        self.sonar_left_sub = rospy.Subscriber('/sensor/sonar_L', Range, self.sonar_left_callback, queue_size=1)
        self.sonar_right_sub = rospy.Subscriber('/sensor/sonar_R', Range, self.sonar_right_callback, queue_size=1)
        self.angle_vel = rospy.Publisher('/mymobibot/angle_vel', Float64, queue_size=1)
        self.angle_err = rospy.Publisher('/mymobibot/angle_err', Float64, queue_size=1)
        self.left_err = rospy.Publisher('/mymobibot/left_err', Float64, queue_size=1)
        self.linear_vel = rospy.Publisher('/mymobibot/linear_vel', Float64, queue_size=1)
        self.sonar_left = rospy.Publisher('/mymobibot/sonar_left', Float64, queue_size=1)
        self.sonar_front = rospy.Publisher('/mymobibot/sonar_front', Float64, queue_size=1)

        # Publishers
        self.angle_vel = rospy.Publisher('/mymobibot/angle_vel', Float64, queue_size=1)
        self.angle_err = rospy.Publisher('/mymobibot/angle_err', Float64, queue_size=1)
        self.left_err = rospy.Publisher('/mymobibot/left_err', Float64, queue_size=1)
        self.linear_vel = rospy.Publisher('/mymobibot/linear_vel', Float64, queue_size=1)
        self.sonar_left = rospy.Publisher('/mymobibot/sonar_left', Float64, queue_size=1)
        self.sonar_front = rospy.Publisher('/mymobibot/sonar_front', Float64, queue_size=1)

        #Publishing rate
        self.period = 1.0/rate
        self.pub_rate = rospy.Rate(rate)

        self.publish()

    #SENSING CALLBACKS
    def joint_states_callback(self, msg):
        # ROS callback to get the joint_states

        self.joint_states = msg
        # (e.g. the angular position of the left wheel is stored in :: self.joint_states.position[0])
        # (e.g. the angular velocity of the right wheel is stored in :: self.joint_states.velocity[1])

    def imu_callback(self, msg):
        # ROS callback to get the /imu

        self.imu = msg
        # (e.g. the orientation of the robot wrt the global frome is stored in :: self.imu.orientation)
        # (e.g. the angular velocity of the robot wrt its frome is stored in :: self.imu.angular_velocity)
        # (e.g. the linear acceleration of the robot wrt its frome is stored in :: self.imu.linear_acceleration)

        #quaternion = (msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w)
        #(roll, pitch, self.imu_yaw) = euler_from_quaternion(quaternion)
        (roll, pitch, self.imu_yaw) = quaternion_to_euler(msg.orientation.w, msg.orientation.x, msg.orientation.y, msg.orientation.z)

    def sonar_front_callback(self, msg):
        # ROS callback to get the /sensor/sonar_F

        self.sonar_F = msg
        # (e.g. the distance from sonar_front to an obstacle is stored in :: self.sonar_F.range)

    def sonar_frontleft_callback(self, msg):
        # ROS callback to get the /sensor/sonar_FL

        self.sonar_FL = msg
        # (e.g. the distance from sonar_frontleft to an obstacle is stored in :: self.sonar_FL.range)

    def sonar_frontright_callback(self, msg):
        # ROS callback to get the /sensor/sonar_FR

        self.sonar_FR = msg
        # (e.g. the distance from sonar_frontright to an obstacle is stored in :: self.sonar_FR.range)

    def sonar_left_callback(self, msg):
        # ROS callback to get the /sensor/sonar_L

        self.sonar_L = msg
        # (e.g. the distance from sonar_left to an obstacle is stored in :: self.sonar_L.range)

    def sonar_right_callback(self, msg):
        # ROS callback to get the /sensor/sonar_R

        self.sonar_R = msg
        # (e.g. the distance from sonar_right to an obstacle is stored in :: self.sonar_R.range)

    def publish(self):

        # set configuration
        self.velocity.linear.x = 0.0
        self.velocity.angular.z = 0.0
        tmp_rate = rospy.Rate(1)
        tmp_rate.sleep()
        print("The system is ready to execute your algorithm...")

        rostime_now = rospy.get_rostime()
        time_now = rostime_now.to_nsec()


        # avoiding null values before assignments
        dt = 0.001 
        err, angle_error = 0, 0

        self.velocity.angular.z = 0


        while not rospy.is_shutdown():

            sonar_front = self.sonar_F.range
            sonar_left = self.sonar_L.range
            sonar_front_left = self.sonar_FL.range
            # sonar_front_right = self.sonar_FR.range
            # sonar_right = self.sonar_R.range

            
            if self.procedure_step == 'init':
                """
                Move close to a wall with a relative fast speed
                Performing a small clockwise rotation
                """
                self.velocity.linear.x = 0.3

                if sonar_front < 2.0:
                    self.velocity.angular.z = 0.4

                if sonar_front < 0.1+self.desired_distance:
                    self.procedure_step = 'follow_wall'
                    prev_err_L = 0
                    prev_err_F = 0

            
            if self.procedure_step == 'follow_wall':
                """
                > X = X1 + X2 = 0+8 = 8
                > X mod 2 = 0
                > We perform a clockwise rotation
                """
                self.velocity.linear.x = 0.1
                
                kp, kd = 10, 10
                kp_angle = 2
                kp_F, kd_F = 15, 15
                
                left_err = sonar_left - self.desired_distance
                left_derivative_err = (left_err - prev_err_L) / dt
                prev_err_L = left_err
                
                angle_err = (sonar_left+0.1)/(sonar_front_left+0.2/sqrt(2)) - sqrt(2)/2

                
                if sonar_front < self.front_desired_distance: # steering
                    front_err = sonar_front - self.front_desired_distance
                    front_derivative_error = (front_err - prev_err_F) / dt
                    prev_err_F = front_err
                    self.velocity.angular.z = -(kp*left_err + kd*left_derivative_err) \
                                                 + kp_angle*angle_err \
                                                    - (kp_F*front_err + kd_F*front_derivative_error)
                
                else: # parallel movement
                    self.velocity.angular.z = -(kp*left_err + kd*left_derivative_err) \
                                                + kp_angle*angle_err 
            

            # Calculate time interval (in case is needed)
            time_prev = time_now
            rostime_now = rospy.get_rostime()
            time_now = rostime_now.to_nsec()
            dt = (time_now - time_prev)/1e9
            
            if dt == 0: dt = 0.001 # to avoid division by zero 

            # Publish the new joint's angular positions
            self.velocity_pub.publish(self.velocity)
            self.angle_vel.publish(self.velocity.angular.z)
            self.angle_err.publish(angle_error)
            self.left_err.publish(err)
            self.linear_vel.publish(self.velocity.angular.z)
            self.sonar_left.publish(sonar_left)
            self.sonar_front.publish(sonar_front)

            self.angle_vel.publish(self.velocity.angular.z)
            self.angle_vel.publish(angle_error)
            self.left_err.publish(err)
            self.linear_vel.publish(self.velocity.linear.x)
            self.sonar_left.publish(sonar_left)
            self.sonar_front.publish(sonar_front)


            self.pub_rate.sleep()

    def turn_off(self):
        pass

def follower_py():
    # Starts a new node
    rospy.init_node('follower_node', anonymous=True)
    # Reading parameters set in launch file
    rate = rospy.get_param("/rate")

    follower = mymobibot_follower(rate)
    rospy.on_shutdown(follower.turn_off)
    rospy.spin()

if __name__ == '__main__':
    try:
        follower_py()
    except rospy.ROSInterruptException:
        pass
