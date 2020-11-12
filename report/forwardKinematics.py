from sympy import *
import numpy as np 
import sys

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




if __name__ == "__main__":

    t1,t2,t3,t4 = getTransformationsFromDHTable(np.array([
    [ -pi/2, 0,  2.5,   -pi/2 + symbols("q_{1}")],
    [ pi/2, 0,  0,  -pi/2 + symbols("q_{2}")],
    [ -pi/2, 3.5,    0,  symbols("q_{3}")],
    [ 0,3, 0, symbols("q_{4}")]]))

    finalTransformation=trigsimp((t1*t2*t3*t4),ratio=1)

    if len(sys.argv) == 5:
        finalTransformation = finalTransformation.subs("q_{1}",parse_expr(sys.argv[1]))
        finalTransformation = finalTransformation.subs("q_{2}",parse_expr(sys.argv[2]))
        finalTransformation = finalTransformation.subs("q_{3}",parse_expr(sys.argv[3]))
        finalTransformation = finalTransformation.subs("q_{4}",parse_expr(sys.argv[4]))


    f = open("latex.tex","w")
    f.write(latex(finalTransformation,
        mode="equation",
        mat_delim="(",
        mat_str="array")
    )

    translation =(finalTransformation*Matrix([[0],[0],[0],[1]])).evalf()
    orientation =Matrix([[atan2(finalTransformation[1,2],finalTransformation[0,2])],
                         [atan2(sqrt(finalTransformation[0,2]**2 + finalTransformation[1,2]**2),finalTransformation[2,2])],
                         [atan2(finalTransformation[2,1],-finalTransformation[2,0])]]).evalf()

    preview(translation, output="png", filename='output.png')
    preview(orientation,output="png", filename='orient.png')
