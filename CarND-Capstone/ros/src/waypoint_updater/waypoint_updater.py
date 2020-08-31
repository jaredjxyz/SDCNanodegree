#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint, TrafficLightArray, Intersection
from geometry_msgs.msg import TwistStamped

import math
import tf
import numpy as np

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 200 # Number of waypoints we will publish. You can change this number

TARGET_SPEED = 5.0
SAMPLE_RATE = 100
ENABLE_TL = 1
MAX_DECEL = 1
STOPPING_DISTANCE = 5
class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')
        #rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb, queue_size=1)
        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb, queue_size=1)
        self.base_wp_sub = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        rospy.Subscriber('/current_velocity', TwistStamped, self.current_velocity_cb)
        rospy.Subscriber('/traffic_waypoint', Intersection, self.traffic_waypoint_cb)

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below


        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)
        # Subscribe to ground truth traffic light array for development

        # TODO: Add other member variables you need below
        self.car_x = None
        self.car_y = None
        self.car_yaw = None
        self.car_pose = None
        self.car_velo = 0.0
        self.car_closest_wp = None
        self.first_waypoint = 0
        self.base_waypoints = None


        self.tl_list = None # list of all traffic lights
        self.tl_X = None # closest traffic light X
        self.tl_Y = None # closest traffic light Y
        self.tl_S = None # closest traffic light state
        self.tl_wp = None # nearest Waypoint to the next light
        self.tl_stop_wp = None
        self.tl_list = []
        self.braking = 0
        self.stop_at_wp = None

        self.target_velo = TARGET_SPEED

        rospy.spin()


    def pose_cb(self, msg):
        self.car_x = msg.pose.position.x
        self.car_y = msg.pose.position.y
        self.car_pose = msg.pose
        if self.base_waypoints is None:
            return
        #need to know euler yaw angle for car orientation relative to waypoints
        #for quaternion transformation using https://answers.ros.org/question/69754/quaternion-transformations-in-python/
        quaternion = [msg.pose.orientation.x,
                        msg.pose.orientation.y,
                        msg.pose.orientation.z,
                        msg.pose.orientation.w]
        euler = tf.transformations.euler_from_quaternion(quaternion)
        self.car_yaw = euler[2]
        self.car_closest_wp = self.get_closest_waypoint(self.car_x, self.car_y)
        self.check_tl()
        self.publishFinalWaypoints()

    def publishFinalWaypoints(self):
        #if not received base_waypoints message, not able to update final_waypoints
        if self.base_waypoints is None:
            return
        closestWaypoint = self.car_closest_wp #self.get_closest_waypoint(self.car_x, self.car_y)
        car_velocity = self.car_velo
        target_velocity = self.target_velo
        rospy.loginfo("Current car velo = %s", car_velocity)
        #updating final_waypoints
        rospy.loginfo("updating waypoints")
        rospy.loginfo("publishing velocity %s", target_velocity)
        self.first_waypoint = closestWaypoint
        lenWaypoints = len(self.base_waypoints)
        final_waypoints_msg = Lane()
        for i in range(closestWaypoint, min(closestWaypoint + LOOKAHEAD_WPS, lenWaypoints)):
            wp = self.base_waypoints[i]
            new_final_wp = Waypoint()
            new_final_wp.pose = wp.pose
            new_final_wp.twist.twist.linear.x =  wp.twist.twist.linear.x 
            if(self.braking == 1):
                stop_line_wp = self.base_waypoints[self.stop_at_wp-2]
                if (i >= self.stop_at_wp): # For waypoints ahead of intended stop point, set velocities 0
                    vel = 0.
                else:
                    dist = self.dist(wp.pose.pose.position, stop_line_wp.pose.pose.position)
                    vel = math.sqrt(2 * MAX_DECEL * dist)
                    if (vel < 1.):
                        vel = 0
                    
                    #rospy.loginfo("dist = %s,cal_vel = %s, vel = %s", dist, vel, wp.twist.twist.linear.x)
                 # Override velocity  
                new_final_wp.twist.twist.linear.x =  min(vel,wp.twist.twist.linear.x) 
                #rospy.loginfo("velo %s = %s, car velo = %s",i,new_final_wp.twist.twist.linear.x, car_velocity)
            final_waypoints_msg.waypoints.append(new_final_wp)

        self.final_waypoints_pub.publish(final_waypoints_msg)

    def waypoints_cb(self, msg):
        #updating base_waypoints
        if self.base_waypoints is None:
            rospy.loginfo("rcvd base waypoints")
            self.base_waypoints = msg.waypoints

    def traffic_waypoint_cb(self,msg):
         self.tl_X = msg.next_light.pose.pose.position.x
         self.tl_Y = msg.next_light.pose.pose.position.y
         self.tl_S = msg.next_light_detection
         self.tl_wp = msg.next_light_waypoint
         self.tl_stop_wp = msg.stop_line_waypoint
         #self.check_tl()


    def traffic_cb(self, msg):
         #TODO: Callback for /traffic_waypoint message. Implement
        #updating traffic light waypoints
        self.tl_list = msg.lights
        #rospy.loginfo("updated state = %s of TL 0 @ x = %s, y = %s", self.tl_list[0].state, self.tl_list[0].pose.pose.position.x, self.tl_list[0].pose.pose.position.y)
        #rospy.loginfo("updated state = %s of TL 1 @ x = %s, y = %s", self.tl_list[1].state, self.tl_list[1].pose.pose.position.x, self.tl_list[1].pose.pose.position.y)
        #rospy.loginfo("updated state = %s of TL 2 @ x = %s, y = %s", self.tl_list[2].state, self.tl_list[2].pose.pose.position.x, self.tl_list[2].pose.pose.position.y)
        #rospy.loginfo("updated state = %s of TL 3 @ x = %s, y = %s", self.tl_list[3].state, self.tl_list[3].pose.pose.position.x, self.tl_list[3].pose.pose.position.y)

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist

    def get_closest_waypoint(self, X, Y):
        closestLen = 100000
        closestWaypoint = 0
        for i in range(self.first_waypoint, len(self.base_waypoints)):
            wp = self.base_waypoints[i]
            dist = math.sqrt((X - wp.pose.pose.position.x)**2
                                + (Y - wp.pose.pose.position.y)**2)
            if dist < closestLen:
                closestLen = dist
                closestWaypoint = i
            else:
                break

        closest_wp = self.base_waypoints[closestWaypoint]
        heading = math.atan2(closest_wp.pose.pose.position.y - Y,
                               closest_wp.pose.pose.position.x - X)

        if closestWaypoint < len(self.base_waypoints) - 1:
            angle = abs(self.car_yaw - heading)
            if (angle > math.pi/4):
                closestWaypoint += 1
                closestWaypoint %= len(self.base_waypoints)

        return closestWaypoint

    def get_closest_tl(self):
        closestLen = 100000
        closestTL = -1
        for i in range(len(self.tl_list)):
            tl = self.tl_list[i]
            dist = math.sqrt((self.car_x - tl.pose.pose.position.x)**2
                                + (self.car_y - tl.pose.pose.position.y)**2)
            if dist < closestLen:
                closestLen = dist
                closestTL = i

        closest_tl = self.tl_list[closestTL]
        heading = math.atan2(closest_tl.pose.pose.position.y - self.car_y,
                                closest_tl.pose.pose.position.x - self.car_x)
        angle = abs(self.car_yaw - heading)
        if (angle > math.pi/4):
            closestTL += 1
            closestTL %= len(self.tl_list)
            closest_tl = self.tl_list[closestTL]
            closestLen = math.sqrt((self.car_x - closest_tl.pose.pose.position.x)**2
                                + (self.car_y - closest_tl.pose.pose.position.y)**2)
            rospy.loginfo("changing TL to %s", closestTL)
        self.tl_Y = closest_tl.pose.pose.position.y
        self.tl_X = closest_tl.pose.pose.position.x
        self.tl_S = closest_tl.state
        return closestTL

    def check_tl(self):
           if self.car_x is not None and self.tl_wp is not None:
                closestWaypoint = self.tl_wp
                rospy.loginfo("tl_x = %s, tl_y = %s, state = %s, WP = %s, stop_wp = %s", self.tl_X, self.tl_Y, self.tl_S, self.tl_wp, self.tl_stop_wp)
                dist = self.distance(self.base_waypoints, self.car_closest_wp, closestWaypoint)
                rospy.loginfo("closest visible tl at %s distance", dist)
                # Our traffic_waypoint publishes only when the next light is red/orange or unknown.
                #self.update_tl = True
                if dist < 35 and dist > 18 and self.tl_S == 0: ### STOP!!!
                    self.target_velo = 0.0
                    self.braking = 1
                    self.stop_at_wp = self.tl_stop_wp
                elif dist < 100 and dist > 34:
                    self.target_velo = 5
                    self.braking = 1
                    self.stop_at_wp = self.tl_stop_wp
                else: ## FULL THROTTLE!!
                    self.target_velo = TARGET_SPEED
                    self.braking = 0
                    #rospy.loginfo("Setting velo to %s",self.target_velo)

    def dist(self, p1, p2):
        return math.sqrt(pow(p1.x-p2.x,2) + pow(p1.y-p2.y,2))

    def current_velocity_cb(self, msg):
        self.car_velo = msg.twist.linear.x

if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
