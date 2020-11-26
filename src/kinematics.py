#!/usr/bin/env python3

import roslib
import sys
import rospy
import numpy as np
from sympy import *
import os


# creates lambda functions which give the forward kinematics, jacobian and velocity kinematics equations
def calculate_all():
    q1,q2,q3,q4 = symbols("q_{1}"),symbols("q_{2}"),symbols("q_{3}"),symbols("q_{4}")

    q1_der,q2_der,q3_der,q4_der = symbols("\dot{q_{1}}"),symbols("\dot{q_{2}}"),symbols("\dot{q_{3}}"),symbols("\dot{q_{4}}")

    t1,t2,t3,t4 = getTransformationsFromDHTable(np.array([
    [ -pi/2 ,      0,  2.5,   -pi/2 + q1],
    [ pi/2  ,      0,    0,  -pi/2 + q2 ],
    [ -pi/2 ,   3.5,     0,      q3     ],
    [ 0     ,      3,    0,      q4    ]]))


    # print((t1*Matrix([[0],[0],[0],[1]]).subs((q1,))))
    # print(t1*t2*Matrix([[0],[0],[0],[1]]))

    full_fk=trigsimp((t1*t2*t3*t4),ratio=1)
    fk_translation = full_fk[0:3,3]

    # print output    
    jacobian = fk_translation.jacobian([q1,q2,q3,q4])
    qderivative = Matrix([[q1_der],
                        [q2_der],
                        [q3_der],
                        [q4_der]])
                        
    formatOutput(full_fk,jacobian)
    

    return (lambdify([q1,q2,q3,q4],fk_translation,"numpy"),
            lambdify([q1,q2,q3,q4],jacobian,"numpy"),
            lambdify([q1,q2,q3,q4,q1_der,q2_der,q3_der,q4_der],jacobian*qderivative,"numpy"))

def getTransformationsFromDHTable(dhTable):

    transformations = []
    for row in dhTable:
        th = row[3]
        d = row[2]
        a = row[1]
        al = row[0]

        transformation = Matrix([
            [cos(th),   -sin(th)*cos(al),   sin(th)*sin(al),    a*cos(th)],
            [sin(th),   cos(th)*cos(al),    -cos(th)*sin(al),   a*sin(th)],
            [0, sin(al),    cos(al),    d],
            [0, 0,  0,  1]])


        transformations.append(simplify(transformation,ratio=1))
    return transformations

def formatOutput(fk,jacobian):

    base_dir = os.path.dirname(os.path.realpath(__file__)) + "/latex"

    translationMatrix = fk[0:4,3]
    rotationMatrix = fk[0:3,0:3]
    f = open(base_dir + "/translation.tex","w")
    f.write(latex(translationMatrix,
        mode="equation",
        mat_delim="(",
        mat_str="array")
    )

    f = open(base_dir + "/rotation.tex","w")
    f.write(latex(rotationMatrix,
	mode="equation",
        mat_delim="(",
        mat_str="array")
    )

    f = open(base_dir + "/fk.tex","w")
    f.write(latex(fk,
	mode="equation",
        mat_delim="(",
        mat_str="array")
    )

    f = open(base_dir + "/jacobian.tex","w")
    f.write(latex(jacobian,
	mode="equation",
        mat_delim="(",
        mat_str="array")
    )




if __name__ == "__main__":
    fk,jac,vk = calculate_all()
    print("success")
