import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/angushu/CSCI-39536-Intro-to-Robotics/install/lab3_pkg'
