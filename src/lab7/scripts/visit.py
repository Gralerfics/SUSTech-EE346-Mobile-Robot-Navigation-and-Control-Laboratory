#!/usr/bin/env python

import time
import yaml
import numpy as np
from scipy.ndimage import distance_transform_cdt

import rospy
import actionlib

from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from geometry_msgs.msg import Pose, PoseWithCovariance, PoseWithCovarianceStamped
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry


class MyMovebaseClient(actionlib.SimpleActionClient):
    def __init__(self, initial = (0, 0, 0)):
        super().__init__('move_base', MoveBaseAction)
        
        # wait for action server
        self.wait_for_server()
        
        # initial pose publisher
        self.initialpose_pub = rospy.Publisher('/initialpose', PoseWithCovarianceStamped, queue_size = 10)
        initial_pose = PoseWithCovarianceStamped()
        initial_pose.header.frame_id = 'map'
        initial_pose.header.stamp = rospy.Time.now()
        if len(initial) == 3:
            initial_pose.pose.pose = self.xytheta_to_pose(*initial)
        elif len(initial) == 4:
            initial_pose.pose.pose = self.xytheta_to_pose(initial[0], initial[1], 0)
            initial_pose.pose.pose.orientation.z = initial[2]
            initial_pose.pose.pose.orientation.w = initial[3]
        
        # amcl_pose subscriber
        rospy.Subscriber('/amcl_pose', PoseWithCovarianceStamped, self.amcl_pose_callback)
        self.amcl_pose = None
        
        # publish initial pose and wait for amcl_pose
        while not rospy.is_shutdown():
            rospy.loginfo('Publishing /initialpose, waiting for /amcl_pose ...')
            self.initialpose_pub.publish(initial_pose)
            # self.amcl_pose = rospy.wait_for_message('/amcl_pose', PoseWithCovarianceStamped, timeout = 1.0)
            if self.amcl_pose is not None:
                dx = self.amcl_pose.pose.pose.position.x - initial_pose.pose.pose.position.x
                dy = self.amcl_pose.pose.pose.position.y - initial_pose.pose.pose.position.y
                dtheta = np.arctan2(self.amcl_pose.pose.pose.orientation.z, self.amcl_pose.pose.pose.orientation.w) * 2 - np.arctan2(initial_pose.pose.pose.orientation.z, initial_pose.pose.pose.orientation.w) * 2
                if np.sqrt(dx ** 2 + dy ** 2) < 0.1 and np.mod(abs(dtheta), 2 * np.pi) < np.deg2rad(5):
                    break
    
    def amcl_pose_callback(self, msg):
        self.amcl_pose = msg
    
    def get_robot_xytheta(self): # -> (x, y, theta_rad)
        return (self.amcl_pose.pose.pose.position.x, self.amcl_pose.pose.pose.position.y, np.arctan2(self.amcl_pose.pose.pose.orientation.z, self.amcl_pose.pose.pose.orientation.w) * 2)
    
    def xytheta_to_pose(self, x, y, theta_rad) -> Pose:
        res = Pose()
        res.position.x = x
        res.position.y = y
        res.orientation.z = np.sin(theta_rad / 2)
        res.orientation.w = np.cos(theta_rad / 2)
        return res
    
    def navigate_by_xytheta(self, x, y, theta_rad, blocking = True):
        self.wait_for_server()
        
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = 'map'
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose = self.xytheta_to_pose(x, y, theta_rad)
        self.send_goal(goal)
        
        # rospy.loginfo(f"Navigating to ({x}, {y}, {theta_rad}) ...")
        if blocking:
            self.wait_for_result()
            return self.get_result()
        else:
            return None


class PolesManager:
    def __init__(self):
        self.visited = []
        self.recognize_tolerance = 0.5
        
        self.pole_array_pub = rospy.Publisher('/poles', Odometry, queue_size = 10)
        
        self.map_file_manager = MapFileManager('./src/lab7/data/map.yaml', './src/lab7/data/map.pgm')
    
    def publish_pole(self, pole): # tuple(float, float)
        pole_odom = Odometry()
        pole_odom.header.frame_id = 'map'
        pole_odom.header.stamp = rospy.Time.now()
        pole_odom.pose.pose.position.x = pole[0]
        pole_odom.pose.pose.position.y = pole[1]
        self.pole_array_pub.publish(pole_odom)
    
    def is_same(self, pole1, pole2): # tuple(float, float), tuple(float, float) -> bool
        return np.linalg.norm(np.array(pole1) - np.array(pole2)) < self.recognize_tolerance
    
    def is_visited(self, pole): # tuple(float, float) -> bool
        for visited_pole in self.visited:
            if self.is_same(pole, visited_pole):
                return True
        return False
    
    def wait_for_lidar_ranges(self):
        def circular_median_filter(values, window_radius):
            return np.median([np.roll(values, x) for x in range(1 - window_radius, window_radius)], axis = 0).tolist()
        
        ranges = np.array(rospy.wait_for_message('/scan', LaserScan).ranges)
        ranges_filtered = np.array(circular_median_filter(ranges, 2)) # ranges
        ranges_filtered[ranges_filtered < 0.1] = 1000.0
        return ranges_filtered.tolist()
    
    def detect_rtheta(self): # -> (r, theta_rad)
        ranges = self.wait_for_lidar_ranges()
        
        # TODO
        nearest_idx = np.argmin(ranges)
        theta_rad = (nearest_idx - 360 if nearest_idx >= 180 else nearest_idx) / 180.0 * np.pi
        return ranges[nearest_idx], theta_rad
    
    def rtheta_to_xytheta(self, x0, y0, theta0, r, theta_rad): # -> (x, y, theta_rad)
        return (x0 + r * np.cos(theta0 + theta_rad), y0 + r * np.sin(theta0 + theta_rad), theta0 + theta_rad)


class MapFileManager:
    def __init__(self, yaml_path, pgm_path = None):
        with open(yaml_path, 'r') as file:
            # load .yaml file
            data = yaml.safe_load(file)
            self.pgm_path_from_yaml = data['image']
            if pgm_path is None:
                pgm_path = self.pgm_path_from_yaml
            self.resolution = data['resolution']
            self.origin = data['origin']
            
            # load .pgm file (with comment line in line 2)
            with open(pgm_path, 'rb') as pgm_file:
                pgm_header = pgm_file.readline().decode('utf-8').strip() # header
                if pgm_header != 'P5':
                    raise ValueError('Only support P5 PGM file format')
                pgm_file.readline() # comment line
                self.width, self.height = map(int, pgm_file.readline().decode('utf-8').strip().split()) # dimensions
                self.max_value = int(pgm_file.readline().decode('utf-8').strip()) # max value
                self.map = np.fromfile(pgm_file, dtype = np.uint8).reshape((self.height, self.width)) # data
            
            # calculate distance field
            self.distances = distance_transform_cdt(self.map != 0) * self.resolution
    
    def get_min_distance_from(self, x, y):
        x_idx = int((x - self.origin[0]) / self.resolution)
        y_idx = self.height - int((y - self.origin[1]) / self.resolution)
        return self.distances[y_idx, x_idx]


if __name__ == '__main__':
    try:
        rospy.init_node('visit')
        
        client = MyMovebaseClient((3.114527131479146, 1.9957134028056018, np.deg2rad(176))) # P1
        # P3: (-1.207153081893921, -1.7155438661575317, -0.035394341186713855, 0.9993734240072419)
        
        pole_manager = PolesManager()
        
        goals = [
            (1.892370461567939, 1.2399054878040552, np.deg2rad(180)), # P1-2, in
            (-0.9659202038331747, 2.164093551514219, np.deg2rad(180)), # P2, x-0.1, y-0.1
            (-0.9659202038331747, 2.164093551514219, np.deg2rad(-90)), # P2, x-0.1, y-0.1
            (0.8941441783576553, -0.9154355192352538, np.deg2rad(-90)), # P12-34
            (-1.1849005393302707, -1.7624668049043297, np.deg2rad(180)), # P3, x-0.08, y-0.05
            (-1.1849005393302707, -1.7624668049043297, np.deg2rad(0)), # P3, x-0.08, y-0.05
            (2.899912171347615, -2.0176318836143958, np.deg2rad(0)), # P4, y-0.05
            (2.899912171347615, -2.0176318836143958, np.deg2rad(180)), # P4, y-0.05
            (1.3058982355526936, 1.3136307594119567, 0), # P1-2, out
            (1.892370461567939, 1.2399054878040552, np.deg2rad(0)), # P1-2, in
            (3.114527131479146, 2.0557134028056018, np.deg2rad(176)), # P1, y+0.06
        ]
        
        for goal in goals:
            client.navigate_by_xytheta(*goal, blocking = False)
            
            pole_detected_acc = 0
            pole_detected = None
            while not rospy.is_shutdown() and not client.wait_for_result(rospy.Duration(0.03)): # not rospy.is_shutdown() for SIGINT (?)
                pole_peek_rtheta = pole_manager.detect_rtheta()
                pole_peek_xytheta = pole_manager.rtheta_to_xytheta(*client.get_robot_xytheta(), *pole_peek_rtheta)
                pole_peek = (pole_peek_xytheta[0], pole_peek_xytheta[1])
                pole_peek_theta = pole_peek_xytheta[2]
                
                if pole_peek is not None and not pole_manager.is_visited(pole_peek):
                    if pole_detected is None:
                        pole_detected = pole_peek
                        pole_detected_acc = 1
                    else:
                        if pole_manager.is_same(pole_peek, pole_detected):
                            pole_detected_acc += 1
                        else:
                            pole_detected = pole_peek
                            pole_detected_acc = 1
                else:
                    pole_detected = None
                    pole_detected_acc = 0
                
                if pole_detected_acc >= 10:
                    rospy.loginfo(f"Detected pole at {pole_peek_xytheta}!")
                    pole_manager.publish_pole(pole_detected)
                    
                    # navigate to pole_detected
                    client.cancel_goal()
                    # TODO
                    
                    # add pole_detected to visited
                    pole_manager.visited.append(pole_detected)
                    
                    # resume navigation
                    client.navigate_by_xytheta(*goal, blocking = False)
                    continue
    
    except rospy.ROSInterruptException:
        rospy.loginfo('ROSInterruptException')
    except KeyboardInterrupt:
        rospy.loginfo('KeyboardInterrupt')

