#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose, Point
from styx_msgs.msg import TrafficLightArray, TrafficLight, Intersection
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import cv2
import yaml
import math

STATE_COUNT_THRESHOLD = 3

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.waypoints = None
        self.camera_image = None
        self.lights = []
        self.intersection_info = None


        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Intersection, queue_size=1)

        self.bridge = CvBridge()
        self.light_classifier = TLClassifier()
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.intersection = None
        self.last_intersection = None
        self.state_count = 0

        self.previous_light_wp = -1
        self.previous_light_detection = -1

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)

        rospy.spin()

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints

    def traffic_cb(self, msg):
        self.lights = msg.lights

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        self.has_image = True
        self.camera_image = msg
        light_wp, intersection = self.process_traffic_lights()
        if intersection is None:
            return

        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        if light_wp != self.previous_light_wp or intersection.next_light_detection != self.previous_light_detection:
            self.upcoming_red_light_pub.publish(intersection)
            self.previous_light_wp = light_wp
            self.previous_light_detection = intersection.next_light_detection

    def get_closest_waypoint(self, point):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
  
        car_x = point.x
        car_y = point.y

        closest_distance = float('inf')
        closest_waypoint_index = None

        for i, waypoint in enumerate(self.waypoints.waypoints):
            waypoint_x = waypoint.pose.pose.position.x
            waypoint_y = waypoint.pose.pose.position.y
            distance = ((waypoint_x - car_x)**2 + (waypoint_y - car_y)**2)
            if distance < closest_distance:
                closest_distance = distance
                closest_waypoint_index = i
        
        return closest_waypoint_index


    def project_to_image_plane(self, point_in_world):
        """Project point from 3D world coordinates to 2D camera image location

        Args:
            point_in_world (Point): 3D location of a point in the world

        Returns:
            x (int): x coordinate of target point in image
            y (int): y coordinate of target point in image

        """

        fx = self.config['camera_info']['focal_length_x']
        fy = self.config['camera_info']['focal_length_y']
        image_width = self.config['camera_info']['image_width']
        image_height = self.config['camera_info']['image_height']

        # get transform between pose of camera and world frame
        trans = None
        try:
            now = rospy.Time.now()
            self.listener.waitForTransform("/base_link",
                  "/world", now, rospy.Duration(1.0))
            (trans, rot) = self.listener.lookupTransform("/base_link",
                  "/world", now)

        except (tf.Exception, tf.LookupException, tf.ConnectivityException):
            rospy.logerr("Failed to find camera to map transform")

        #TODO Use tranform and rotation to calculate 2D position of light in image

        x = 0
        y = 0

        return (x, y)

    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        if not self.has_image:
            self.prev_light_loc = None
            return False

        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")

        # x, y = self.project_to_image_plane(light.pose.pose.position)

        #TODO use light location to zoom in on traffic light in image

        #Get classification
        classification = self.light_classifier.get_classification(cv_image)
        return classification

    def process_intersections(self):
        """
        Sets self.intersection_info to a dictionary where the key
        is the index of the waypoint closest to a light, and the dictionary
        contains {'light': light,
                  'line': line,
                  'light_wp': light_wp,
                  'line_wp': line_wp}
        """

        # Get the closest waypoint to each light and stop line
        closest_light_wps = {light: (None, float('inf')) for light in self.lights}
        closest_line_wps = {tuple(line): (None, float('inf')) for line in self.config['stop_line_positions']}
        for wp_i, waypoint in enumerate(self.waypoints.waypoints):
            wp_x = waypoint.pose.pose.position.x
            wp_y = waypoint.pose.pose.position.y
            for light in closest_light_wps:
                light_x = light.pose.pose.position.x
                light_y = light.pose.pose.position.y
                wp_distance = (light_x - wp_x)**2 + (light_y - wp_y)**2
               
                closest_wp, closest_distance = closest_light_wps[light]
                if wp_distance < closest_distance:
                    closest_light_wps[light] = (wp_i, wp_distance)
            
            for line in closest_line_wps:
                line_x, line_y = line
                wp_distance = (line_x - wp_x)**2 + (line_y - wp_y)**2
                
                closest_wp, closest_distance = closest_line_wps[line]
                if wp_distance < closest_distance:
                    closest_line_wps[line] = (wp_i, wp_distance)

        # Now that we have the closest waypoint to each light and line,
        # pair lines with lights

        intersection_info = {}
        for light in closest_light_wps:
            light_x = light.pose.pose.position.x
            light_y = light.pose.pose.position.y
            
            closest_line = None
            closest_line_distance = float('inf')
            for line in closest_line_wps:
                line_x, line_y = line
                line_distance = (light_x - line_x)**2 + (light_y - line_y)**2
                
                if line_distance < closest_line_distance:
                    closest_line = line
                    closest_line_distance = line_distance
            
            light_wp = closest_light_wps[light][0]
            line_wp = closest_line_wps[closest_line][0]
            intersection_info[light_wp] = {'light': light, 'line': closest_line, 'light_wp': light_wp, 'line_wp': line_wp}

        self.intersection_info = intersection_info
            
                
    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        light = None

        if not (self.pose and self.lights and self.waypoints):
           return None, None

        if self.intersection_info is None:
            self.process_intersections()

        # Find closest waypoint to the above light
        closest_wp = self.get_closest_waypoint(self.pose.pose.position)            

        next_light_wp = closest_wp
        while next_light_wp not in self.intersection_info:
            next_light_wp += 1
            next_light_wp = next_light_wp % len(self.waypoints.waypoints)

        next_intersection = self.intersection_info[next_light_wp]

        light_state = self.get_light_state(next_intersection['light'])

        # If the light is not red
        if light_state == TrafficLight.GREEN:
            light_state = TrafficLight.UNKNOWN
            next_light_wp += 1
            while next_light_wp not in self.intersection_info:
                next_light_wp += 1
                next_light_wp = next_light_wp % len(self.waypoints.waypoints)
            next_intersection = self.intersection_info[next_light_wp]

        # Build the Intersection for return
        intersection = Intersection()
        intersection.next_light = next_intersection['light']
        intersection.stop_line_waypoint = next_intersection['line_wp']
        intersection.next_light_waypoint = next_intersection['light_wp']
        intersection.next_light_detection = light_state

        return next_intersection['light_wp'], intersection

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
