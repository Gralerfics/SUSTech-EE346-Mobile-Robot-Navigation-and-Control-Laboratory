#! /usr/bin/env python

import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan

import numpy as np


class PoleParker:
    def __init__(self, k_rho, k_alpha, k_p, k_d, k_i, allow_reverse_parking):
        self.k_rho = k_rho
        self.k_alpha = k_alpha
        
        self.k_p = k_p
        self.k_d = k_d
        self.k_i = k_i
        
        self.allow_reverse_parking = allow_reverse_parking
        
        self.pub_cmd_vel = rospy.Publisher('/cmd_vel', Twist, queue_size = 10)
    
    def circular_median_filter(self, values, window_radius):
        return np.median([np.roll(values, x) for x in range(1 - window_radius, window_radius)], axis = 0).tolist()
    
    def get_ranges_filtered(self, window_radius = 3):
        scan = rospy.wait_for_message('/scan', LaserScan)
        # ranges_filtered = np.array(self.circular_median_filter(scan.ranges, window_radius))
        ranges_filtered = np.array(scan.ranges)
        ranges_filtered[ranges_filtered < 0.1] = 1000.0
        return ranges_filtered.tolist()
    
    def detect_pole(self):
        ranges = self.get_ranges_filtered()
        nearest_idx = np.argmin(ranges)
        alpha = (nearest_idx - 360 if nearest_idx >= 180 else nearest_idx) / 180.0 * np.pi
        return alpha, ranges[nearest_idx]
    
    def stop(self):
        self.pub_cmd_vel.publish(Twist())
    
    def run(self):
        v = 0
        omega = 0
        
        integral_v = 0
        integral_omega = 0
        
        delta_v = 0
        delta_omega = 0
        
        while not rospy.is_shutdown():
            try:
                alpha, rho = self.detect_pole()
                if rho < 0.15:
                    self.stop()
                    break
                
                if self.allow_reverse_parking and abs(alpha) > np.pi / 2:
                    alpha = alpha + np.pi if alpha < 0 else alpha - np.pi
                    rho = -rho
                
                target_v = np.clip(self.k_rho * rho * (1 if abs(alpha) < np.pi / 2 else -1), -0.22, 0.22)
                target_omega = np.clip(self.k_alpha * alpha, -2.84, 2.84)
                
                error_v = target_v - v
                error_omega = target_omega - omega
                
                integral_v += error_v
                integral_omega += error_omega
                
                delta_v = self.k_p * error_v - self.k_d * delta_v + self.k_i * integral_v
                delta_omega = self.k_p * error_omega - self.k_d * delta_omega + self.k_i * integral_omega
                
                v = np.clip(v + delta_v, -0.22, 0.22)
                omega = np.clip(omega + delta_omega, -2.84, 2.84)
                
                twist = Twist()
                twist.linear.x = v
                twist.angular.z = omega
                
                # print(alpha / np.pi * 180, rho)
                # print(twist.linear.x, twist.angular.z, target_v, target_omega, v, omega)
                
                self.pub_cmd_vel.publish(twist)
            except KeyboardInterrupt:
                self.stop()
                break


def main():
    rospy.init_node('pole_parker', anonymous = True)
    
    pole_parker = PoleParker(0.6, 0.8, 0.25, 0.02, 0.01, False) # Stable, not so fast
    # pole_parker = PoleParker(0.8, 0.8, 0.3, 0.02, 0.01, False)
    pole_parker.run()

    rospy.spin()

if __name__ == '__main__':
    main()

