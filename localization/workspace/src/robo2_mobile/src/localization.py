#!/usr/bin/env python3

# Robotics II 2021-2022
# ECE - NTUA
# Benetou Smaragda el18048
# Vatistas Andreas el18020

"""
Start ROS node to publish (x,y,theta)
perfoming Kalman Filtering using IMU and sonar measurements
"""

# Ros handlers services and messages
from re import S
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

class mymobibot_localization():
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
        # initialize subscribers for reading encoders and publishers for performing position predicts and updates

        # Sensors
        self.imu_sub = rospy.Subscriber('/imu', Imu, self.imu_callback, queue_size=1)
        self.sonar_front_sub = rospy.Subscriber('/sensor/sonar_F', Range, self.sonar_front_callback, queue_size=1)
        self.sonar_frontleft_sub = rospy.Subscriber('/sensor/sonar_FL', Range, self.sonar_frontleft_callback, queue_size=1)
        self.sonar_frontright_sub = rospy.Subscriber('/sensor/sonar_FR', Range, self.sonar_frontright_callback, queue_size=1)
        self.sonar_left_sub = rospy.Subscriber('/sensor/sonar_L', Range, self.sonar_left_callback, queue_size=1)
        self.sonar_right_sub = rospy.Subscriber('/sensor/sonar_R', Range, self.sonar_right_callback, queue_size=1)

        # Publishers
        self.x = rospy.Publisher('/x_position', Float64, queue_size=1)
        self.y = rospy.Publisher('/y_position', Float64, queue_size=1)
        self.theta = rospy.Publisher('/theta_angle', Float64, queue_size=1)


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

        tmp_rate = rospy.Rate(1)
        tmp_rate.sleep()
        print("********The system is ready to execute Kalman Filtering...*******")


        dt = 0.01 # equal to the rate provided by ros - could be solved with rostimenow and time-diffs

        sigma_ax = 0.002
        sigma_w  = 0.002

        init_angle = 0#1.717

        # initialize state
        xhat = 0
        yhat = 0
        thetahat = init_angle
        vel_x = 0

        P = np.zeros((3,3))

        while not rospy.is_shutdown():

            # PREDICT

            ax = self.imu.linear_acceleration.x
            angular_vel = self.imu.angular_velocity.z
            theta = thetahat

            sigma_x = 1/2*sigma_ax*np.cos(theta)*(dt**2)
            sigma_y = 1/2*sigma_ax*np.sin(theta)*(dt**2)
            sigma_theta = sigma_w * dt

            Cw = np.array([[sigma_x**2,0,0],\
                            [0,sigma_y**2,0],\
                                [0,0,sigma_theta**2]])

            xhat = xhat + (vel_x*dt+1/2*ax*dt**2)*np.cos(theta)
            yhat = yhat + (vel_x*dt+1/2*ax*dt**2)*np.sin(theta)
            thetahat = thetahat + angular_vel*dt
            vel_x += ax*dt

            #linearization_of_state_space
            A = np.array([ [1, 0, -(vel_x*dt+1/2*ax*dt**2)*np.sin(theta)],\
                            [0, 1, (vel_x*dt+1/2*ax*dt**2)*np.cos(theta)],\
                                [0, 0, 1]])

            #prediction probability
            P = np.dot(np.dot(A,P), np.transpose(A)) + Cw



            # UPDATE

            sonar_front = self.sonar_F.range
            sonar_left = self.sonar_L.range
            sonar_front_left = self.sonar_FL.range
            sonar_front_right = self.sonar_FR.range
            sonar_right = self.sonar_R.range
            theta = self.imu
            angle = theta.orientation.z
            wallangle = angle
            dist = 0
            length = 0.4
            predictxabs = 0
            predictyabs = 0
            predictthetaabs = 0
            x,y, angle = xhat, yhat, thetahat
            if sonar_front < 2 and sonar_front_left < 2 :
                if angle > 0 and angle < np.pi/2 :
                    wallangle = angle
                    wall = 1
                elif angle > np.pi/2 :
                    wallangle =  angle - np.pi/2
                    wall = 2
                elif angle < 0 and angle > -np.pi/2 :
                    wallangle = np.pi/2 + angle
                    wall = 4
                else :
                    wallangle = np.pi+angle
                    wall = 3

                l1 = sonar_front_left + 0.106
                l2 = sonar_front + 0.1414
                df = np.pi/4
                l3 = sqrt(l1**2+l2**2 - 2*l1*l2*cos(df))
                a = np.arcsin(sin(df)*l1/l3)
                if  wallangle-0.01<a and a< wallangle+0.01:
                    if predictthetaabs == 0:
                        predictthetaabs = wallangle - a
                    #same wall
                    dist = np.sin(a)*(sonar_front + length/2)
                    if (wall == 1 or wall == 3) and predictyabs == 0:
                        predictyabs = 2 - dist
                    if (wall == 2 or wall == 4) and predictxabs == 0:
                        predictxabs = 2 - dist
                else :
                    #different wall

                    if (x>0 and y > 0 and angle > 0) or (x>0 and y<0 and angle > -np.pi/2) or (x<0 and y<0 and angle <0) or (x<0 and y>0 and angle>0):
                        dist = np.sin(np.pi  - a)*(sonar_front+length/2)
                        wall = wall - 1
                        if wall == 0 :
                            wall = 4
                        if (wall == 1 or wall == 3) and predictyabs == 0 :
                            predictyabs = 2 - dist
                        if (wall == 2 or wall == 4) and predictxabs == 0:
                            predictxabs = 2 - dist
                    else:
                        dist = np.sin(a)*(sonar_front + length/2)
                        if wall == 1 or wall == 3 :
                            predictyabs = 2 - dist
                        if wall == 2 or wall == 4 :
                            predictxabs = 2 - dist

            if sonar_front < 2 and sonar_front_right < 2:
                if angle > 0 and angle < np.pi/2 :
                    wallangle = np.pi - angle
                    wall = 4
                elif angle > np.pi/2 :
                    wall = 1
                    wallangle = np.pi/2 - angle
                elif angle < 0 and angle > -np.pi/2 :
                    wall = 3
                    wallangle = -angle
                else :
                    wall = 2
                    wallangle = -np.pi/2+angle

                l1 = sonar_front_right + 0.106
                l2 = sonar_front = 0.1414
                df = np.pi/4
                l3 = sqrt(l1**2+l2**2 - 2*l1*l2*cos(df))
                a = np.arcsin(sin(df)*l1/l3)
                if  wallangle-0.01<a and a< wallangle+0.01:
                    #same wall
                    if predictthetaabs == 0:
                        predictthetaabs = wallangle - a
                    dist = np.sin(a)*(sonar_front + length/2)
                    if (wall == 1 or wall == 3) and predictyabs ==0 :
                        predictyabs = 2 - dist
                    if (wall == 2 or wall == 4) and  predictxabs ==0 :
                        predictxabs = 2 - dist
                else :
                    #different wall
                    if (x<0 and y > 0 and angle > 0) or (x>0 and y<0 and angle < 0) or (x<0 and y<0 and angle > -np.pi/2) or (x>0 and y>0 and angle<np.pi/2):
                        dist = np.sin(np.pi  - a)*(sonar_front+length/2)
                        wall = wall+1
                        if wall == 5 :
                            wall = 1
                        if (wall == 1 or wall == 3) and predictyabs ==0 :
                            predictyabs = 2 - dist
                        if (wall == 2 or wall == 4) and predictxabs == 0:
                            predictxabs = 2 - dist

                    else:
                        dist = np.sin(wallangle)*(sonar_front + length/2)
                        if (wall == 1 or wall == 3) and predictyabs ==0 :
                            predictyabs = 2 - dist
                        if (wall == 2 or wall == 4) and predictxabs == 0  :
                            predictxabs = 2 - dist

            if sonar_left < 2 and sonar_front_left < 2 :
                if angle > 0 and angle < np.pi/2 :
                    wall = 4
                    wallangle = angle
                elif angle > np.pi/2 :
                    wall = 1
                    wallangle = angle-np.pi/2
                elif angle < 0 and angle > -np.pi/2 :
                    wallangle = np.pi/2 + angle
                    wall = 3
                else :
                    wall = 4
                    wallangle = np.pi+angle

                l1 = sonar_front_left + 0.106
                l2 = sonar_left + 0.3
                df = np.pi/4
                l3 = sqrt(l1**2+l2**2 - 2*l1*l2*cos(df))
                a = np.arcsin(sin(df)*l1/l3)
                if  wallangle-0.01<a and a< wallangle+0.01:
                    if predictthetaabs == 0:
                        predictthetaabs = wallangle - a
                    #same wall
                    dist = np.sin(a)*(sonar_left + length/2)
                    if (wall == 1 or wall == 3) and predictyabs ==0 :
                        predictyabs = 2 - dist
                    if (wall == 2 or wall == 4) and predictxabs==0:
                        predictxabs = 2 - dist

                else :

                    #different wall
                    if (x>0 and y > 0 and angle > np.pi/2) or (x>0 and y<0 and angle > 0) or (x<0 and y<0 and angle <0 and angle > -np.pi/2) or (x<0 and y>0 and angle<0):
                        dist = np.sin(np.pi  - a)*(sonar_left+length/2)
                        wall = wall+1
                        if wall == 5 :
                            wall = 1
                        if (wall == 1 or wall == 3) and predictyabs==0 :
                            predictyabs = 2 - dist
                        if (wall == 2 or wall == 4) and predictxabs==0 :
                            predictxabs = 2 - dist

                    else:
                        dist = np.sin(a)*(sonar_left + length/2)
                        if (wall == 1 or wall == 3) and predictyabs == 0 :
                            predictyabs = 2 - dist
                        if (wall == 2 or wall == 4) and predictxabs == 0:
                            predictxabs = 2 - dist

            if sonar_right < 2 and sonar_front_right < 2 :
                if angle > 0 and angle < np.pi/2 :
                    wall = 4
                    wallangle = angle
                elif angle > np.pi/2 :
                    wall = 1
                    wallangle = angle-np.pi/2
                elif angle < 0 and angle > -np.pi/2 :
                    wallangle = np.pi/2 + angle
                    wall = 3
                else :
                    wall = 4
                    wallangle = np.pi+angle

                l1 = sonar_front_right + 0.106
                l2 = sonar_right + 0.3
                df = np.pi/4
                l3 = sqrt(l1**2+l2**2 - 2*l1*l2*cos(df))
                a = np.arcsin(sin(df)*l1/l3)
                if  wallangle-0.01<a and a< wallangle+0.01:
                    if predictthetaabs == 0:
                        predictthetaabs = wallangle - a
                    #same wall
                    dist = np.sin(a)*(sonar_right + length/2)
                    if (wall == 1 or wall == 3) and predictyabs ==0 :
                        predictyabs = 2 - dist
                    if (wall == 2 or wall == 4) and predictxabs==0:
                        predictxabs = 2 - dist

                else :

                    #different wall
                    if (x>0 and y > 0 and angle > np.pi/2) or (x>0 and y<0 and angle > 0) or (x<0 and y<0 and angle > -np.pi/2) or (x<0 and y>0 and angle<0):
                        dist = np.sin(np.pi  - a)*(sonar_right+length/2)
                        wall = wall+1
                        if wall == 5 :
                            wall = 1
                        if (wall == 1 or wall == 3) and predictyabs==0 :
                            predictyabs = 2 - dist
                        if (wall == 2 or wall == 4) and predictxabs==0 :
                            predictxabs = 2 - dist

                    else:
                        dist = np.sin(a)*(sonar_right + length/2)
                        if (wall == 1 or wall == 3) and predictyabs == 0 :
                            predictyabs = 2 - dist
                        if (wall == 2 or wall == 4) and predictxabs == 0:
                            predictxabs = 2 - dist

            if sonar_front <= 2 :
                #since sonar_front_right and sonar_front_left are >= 2 we know which wall is hitting sonar_front
                if angle > -np.pi/4 and angle < np.pi/4 :
                    wall = 4
                    wallangle = np.pi/2 - np.abs(angle)
                elif angle > np.pi/4 and angle < np.pi*3/4 :
                    if angle > np.pi/2 :
                        wallangle = np.pi - angle
                    else :
                        wallangle = angle
                    wall = 1
                elif angle < -np.pi/4 and angle > -3*np.pi/4 :
                    if angle < -np.pi/2 :
                        wallangle = np.pi+ angle
                    else:
                        wallangle = -angle
                    wall = 3
                else :
                    if angle < 0 :
                        wallangle = np.pi/2 - (np.pi - np.abs(angle))
                    wall = 2
                dist = np.sin(wallangle)*(sonar_front + length/2)
                if (wall == 1 or wall == 3) and predictyabs == 0:
                    predictyabs = 2 - dist
                if (wall == 2 or wall == 4) and predictxabs == 0:
                    predictxabs = 2 - dist






            #linearization_of_measurement_model
            H = np.transpose(np.array([1,1,1]))

            Cv = np.array([[0.01,0,0],\
                            [0,0.01,0],\
                                [0,0,0.01]])

            #Kalman optimal gain
            K = P*np.transpose(H)*np.linalg.inv((H*P*np.transpose(H)+Cv))

            #update
            z = np.array([predictxabs - np.abs(xhat), predictyabs - np.abs(yhat), predictthetaabs]) #predictthetaabs exei ginei hdh afairesh
            xbar = np.array([xhat, yhat, thetahat])

            x = xbar + K @ (z - H @ xbar)


            P = (np.eye(3) - K*H)*P


            print(x[0],", ",x[1],", ",x[2])

            self.x.publish(x[0])
            self.y.publish(x[1])
            self.theta.publish(x[2])


            xhat, yhat, thetahat = x


            self.pub_rate.sleep()

    def turn_off(self):
        pass

def follower_py():
    # Starts a new node
    rospy.init_node('self_localization', anonymous=True)
    # Reading parameters set in launch file
    rate = rospy.get_param("/rate")

    follower = mymobibot_localization(rate)
    rospy.on_shutdown(follower.turn_off)
    rospy.spin()

if __name__ == '__main__':
    try:
        follower_py()
    except rospy.ROSInterruptException:
        pass