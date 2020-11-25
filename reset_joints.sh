#!/bin/bash
rostopic pub -1 /robot/joint1_position_controller/command std_msgs/Float64 "data:  0" & 
rostopic pub -1 /robot/joint2_position_controller/command std_msgs/Float64 "data:  0" &
rostopic pub -1 /robot/joint3_position_controller/command std_msgs/Float64 "data:  0" &
rostopic pub -1 /robot/joint4_position_controller/command std_msgs/Float64 "data:  0"
