#!/usr/bin/python
# -*- coding: utf-8 -*-

from loop_solver import frame_diff
from numpy import allclose
from numpy.matlib import identity
T0 = identity(4)
T1 = identity(4)
assert(allclose(frame_diff(T0, T1), 0))

from robots import table_sr400
from kinematics import Kinematics
from loop_solver import LoopSolver
kin = Kinematics(table_sr400)
jnts = kin.joints
ls = LoopSolver(jnts, jnts[0], jnts[8])

assert(ls.get_end_joints() == [jnts[8], jnts[9]])

assert(ls.get_chains() == [[jnts[6], jnts[7], jnts[8]],
    [jnts[1], jnts[2], jnts[9]]])

J = ls.get_cjoint_jac()

ls.solve()
