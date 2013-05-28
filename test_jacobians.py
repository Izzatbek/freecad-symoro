#!/usr/bin/python
# -*- coding: utf-8 -*-

from math import cos, sin
from numpy import matrix
from numpy import allclose
from numpy import zeros
from numpy import concatenate
from kinematics import Kinematics
from jacobians import serialKinematicJacobian as jacobian
from robots import table_rx90

kin = Kinematics(table_rx90)
kin.set_q([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
jnts = kin.joints

# This jacobian is obtained from the course 'Modeling and Control of Manipulators' of Wisama Khalil
numer_jac = matrix([[-10.2079, -408.5824, -349.2793,  0,       0,       0],
                     [101.7389, -40.9950,  -35.0448,   0,       0,       0],
                     [0,         102.2498, -191.7702,  0,       0,       0],
                     [0,         0.0998,    0.0998,   -0.4770,  0.4320, -0.7856],
                     [0,        -0.9950,   -0.9950,   -0.0479, -0.8823, -0.2665],
                     [1.0000,    0.0000,    0.0000,    0.8776,  0.1867,  0.5584]
                    ])
J = jacobian(jnts)
assert allclose(numer_jac, J, rtol=8e-04)

# From "Robots-manipulateurs de type série" - Wisama Khalil, Etienne Dombre, p.12.
# http://www.gdr-robotique.org/cours_de_robotique/?id=fd80a49ceaa004030c95cdacb020ec69.
# In French.
q0, q1, q2, q3, q4, q5 = kin.get_q()
c1 = cos(q0)
c2 = cos(q1)
c3 = cos(q2)
c4 = cos(q3)
c5 = cos(q4)
s1 = sin(q0)
s2 = sin(q1)
s3 = sin(q2)
s4 = sin(q3)
s5 = sin(q4)
c23 = cos(q1 + q2)
s23 = sin(q1 + q2)

# From the lectures of Wisama Khalil. Handbook of the course 'Modeling and Control of Manipulators', p. 45
R01 = matrix([[c1, -s1, 0],
              [s1,  c1, 0],
              [0,   0,  1]])

R13 = matrix([[c23, -s23, 0],
              [0,    0,  -1],
              [s23,  c23, 0]])

R03 = R01 * R13

# From the lectures of Wisama Khalil. Handbook of the course 'Modeling and Control of Manipulators', p. 96
D3 = table_rx90[2][7]
RL4 = table_rx90[3][9]
J36_Khalil = matrix([[0, -RL4 + s3 * D3, -RL4, 0, 0, 0],
                     [0, c3 * D3, 0, 0, 0, 0],
                     [s23 * RL4 - c2 * D3, 0, 0, 0, 0, 0],
                     [s23, 0, 0, 0, s4, -s5 * c4],
                     [c23, 0, 0, 1, 0, c5],
                     [0, 1, 1, 0, c4, s5 * s4]])

analyt_jac = matrix(concatenate((concatenate((R03, zeros((3, 3))), 1), concatenate((zeros((3, 3)), R03), 1)))) * J36_Khalil
assert allclose(analyt_jac, J)

print("Success!")