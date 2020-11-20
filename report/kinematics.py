from sympy import *
import numpy as np 
import sys
from sympy.printing.pycode import NumPyPrinter

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

def formatOutput(fk,jacobian, qderivative):

    translationMatrix = fk[0:4,3]
    rotationMatrix = fk[0:3,0:3]
    f = open("translation.tex","w")
    f.write(latex(translationMatrix,
        mode="equation",
        mat_delim="(",
        mat_str="array")
    )

    f = open("translation.py","w")
    f.write(NumPyPrinter(rotationMatrix))

    f = open("rotation.tex","w")
    f.write(latex(rotationMatrix,
	mode="equation",
        mat_delim="(",
        mat_str="array")
    )

    f = open("fk.tex","w")
    f.write(latex(fk,
	mode="equation",
        mat_delim="(",
        mat_str="array")
    )

    f = open("jacobian.tex","w")
    f.write(latex(jacobian,
	mode="equation",
        mat_delim="(",
        mat_str="array")
    )

    f = open("vk.tex","w")
    f.write(latex(jacobian * qderivative,
	mode="equation",
        mat_delim="(",
        mat_str="array")
    )



def calc_jacobian(fkMatrix,q1,q2,q3,q4):
    jacobian = fkMatrix.jacobian([q1,q2,q3,q4])
    return jacobian

if __name__ == "__main__":

    q1,q2,q3,q4 = symbols("q_{1}"),symbols("q_{2}"),symbols("q_{3}"),symbols("q_{4}")
    q1_der,q2_der,q3_der,q4_der = symbols("\dot{q_{1}}"),symbols("\dot{q_{2}}"),symbols("\dot{q_{3}}"),symbols("\dot{q_{4}}")

    t1,t2,t3,t4 = getTransformationsFromDHTable(np.array([
    [ -pi/2, 0,  2.5,   -pi/2 + q1],
    [ pi/2, 0,  0,  -pi/2 + q2],
    [ -pi/2, 3.5,    0,  q3],
    [ 0,3, 0, q4]]))

    finalTransformation=trigsimp((t1*t2*t3*t4),ratio=1)

    if len(sys.argv) == 5:
        finalTransformation = finalTransformation.subs("q_{1}",parse_expr(sys.argv[1]))
        finalTransformation = finalTransformation.subs("q_{2}",parse_expr(sys.argv[2]))
        finalTransformation = finalTransformation.subs("q_{3}",parse_expr(sys.argv[3]))
        finalTransformation = finalTransformation.subs("q_{4}",parse_expr(sys.argv[4]))

    # print output
    jacobian = calc_jacobian(finalTransformation[0:3,3],q1,q2,q3,q4)
    formatOutput(finalTransformation,jacobian,Matrix([[q1_der],
                                                      [q2_der],
                                                      [q3_der],
                                                      [q4_der]]))


    # show the final orientation and translation for the given parameters 
    translation =(finalTransformation*Matrix([[0],[0],[0],[1]])).evalf()
    orientation =Matrix([[atan2(finalTransformation[1,2],finalTransformation[0,2])],
                         [atan2(sqrt(finalTransformation[0,2]**2 + finalTransformation[1,2]**2),finalTransformation[2,2])],
                         [atan2(finalTransformation[2,1],-finalTransformation[2,0])]]).evalf()



    preview(translation, output="png", filename='output.png')
    preview(orientation,output="png", filename='orient.png')
