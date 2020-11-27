#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

$DIR/rotate_joint.sh 1 $1 &
$DIR/rotate_joint.sh 2 $2 &
$DIR/rotate_joint.sh 3 $3 &
$DIR/rotate_joint.sh 4 $4