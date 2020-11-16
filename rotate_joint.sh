#!/bin/bash
echo "$1 * 57" | bc
rostopic pub -1 /robot/joint$1_position_controller/command std_msgs/Float64 "data:  $2"
