#!/usr/bin/env python

import numpy as np

import rospy
import actionlib

from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal


class MovebaseClient:
    def __init__(self):
        self.client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        rospy.loginfo("Waiting for move_base action server ...")
        self.client.wait_for_server()
        rospy.loginfo("Connected to move_base action server.")
    
    def navigate_by_xytheta(self, x, y, theta_deg, block_fn = lambda **args: rospy.loginfo("Navigating ..."), block_fn_interval = 0.1):
        self.client.wait_for_server()
        theta_rad = np.deg2rad(theta_deg)
        
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = 'map'
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose.position.x = x
        goal.target_pose.pose.position.y = y
        goal.target_pose.pose.orientation.z = np.sin(theta_rad / 2)
        goal.target_pose.pose.orientation.w = np.cos(theta_rad / 2)
        self.client.send_goal(goal)
        
        rospy.loginfo(f"Navigating to ({x}, {y}, {theta_deg}) ...")
        if block_fn is not None:
            while not rospy.is_shutdown() and not self.client.wait_for_result(rospy.Duration(block_fn_interval)):
                # not rospy.is_shutdown() for Ctrl+C (?)
                block_fn() # TODO: pass args
            return self.client.get_result()
        else:
            return None


if __name__ == '__main__':
    try:
        rospy.init_node('goals_new')
        
        client = MovebaseClient()
        
        result = client.navigate_by_xytheta(3.166435534256725, 1.941716333329415, 180) # P1
        result = client.navigate_by_xytheta(1.892370461567939, 1.2399054878040552, 180) # P1-2, in
        result = client.navigate_by_xytheta(-0.8659202038331747, 2.264093551514219, -90) # P2, TODO
        result = client.navigate_by_xytheta(0.8941441783576553, -0.9154355192352538, -90) # P12-34
        result = client.navigate_by_xytheta(-1.1849005393302707, -1.7624668049043297, 180) # P3, x-0.08, y-0.05, TODO
        result = client.navigate_by_xytheta(2.899912171347615, -2.0176318836143958, 180) # P4, y-0.05
        
        # result = client.navigate_by_xytheta(0.8941441783576553, -0.9154355192352538, 90) # P12-34
        result = client.navigate_by_xytheta(1.3058982355526936, 1.3136307594119567, 0) # P1-2, out
        result = client.navigate_by_xytheta(3.166435534256725, 1.941716333329415, 0) # P1, TODO
    except rospy.ROSInterruptException:
        pass

