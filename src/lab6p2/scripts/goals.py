#!/usr/bin/env python

import rospy
import actionlib

from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal


def move_to_goal(x, y):
    client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
    
    rospy.loginfo("Waiting for server ...")
    client.wait_for_server()
    
    rospy.loginfo(f"Sending goal to ({x}, {y}) ...")
    goal = MoveBaseGoal()
    goal.target_pose.header.frame_id = "map"
    goal.target_pose.header.stamp = rospy.Time.now()
    goal.target_pose.pose.position.x = x
    goal.target_pose.pose.position.y = y
    goal.target_pose.pose.orientation.w = 1.0
    client.send_goal(goal)

    rospy.loginfo("Waiting for result ...")
    wait = client.wait_for_result()
    if not wait:
        rospy.logerr("Action server not available!")
        rospy.signal_shutdown("Action server not available!")
    else:
        rospy.loginfo("Goal reached!")
        return client.get_result()

if __name__ == '__main__':
    try:
        rospy.init_node('goals')
        
        client = actionlib.SimpleActionClient('move_base', MoveBaseAction)

        result = move_to_goal(0.05906636667735339, 0.5282931145835327) # P1
        result = move_to_goal(1.6232298292367529, 0.29091046522267017) # P1.1
        result = move_to_goal(4.090537292440351, 0.010584508509484562) # P2 x+2
        result = move_to_goal(1.6232298292367529, 0.29091046522267017) # P1.1
        # result = move_to_goal(2.8596579848240773, 3.403375831152019) # P3.2 y-2
        result = move_to_goal(4.520944884067175, 4.058297820749271) # P3 x+2
        result = move_to_goal(3.4100635639474732, 3.3839259020153683) # P3.1 y-2
        result = move_to_goal(0.4148001240322666, 4.535431950921567) # P4 y+10
        # result = move_to_goal(, )
    except rospy.ROSInterruptException:
        pass

