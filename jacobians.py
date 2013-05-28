
#***************************************************************************
#*                                                                         *
#*   Copyright (c) 2012 Gael Ecorchard <galou_breizh@yahoo.fr>             *
#*                                                                         *
#*   This program is free software; you can redistribute it and/or modify  *
#*   it under the terms of the GNU Lesser General Public License (LGPL)    *
#*   as published by the Free Software Foundation; either version 2 of     *
#*   the License, or (at your option) any later version.                   *
#*   for detail see the LICENCE text file.                                 *
#*                                                                         *
#*   This program is distributed in the hope that it will be useful,       *
#*   but WITHOUT ANY WARRANTY; without even the implied warranty of        *
#*   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         *
#*   GNU Library General Public License for more details.                  *
#*                                                                         *
#*   You should have received a copy of the GNU Library General Public     *
#*   License along with this program; if not, write to the Free Software   *
#*   Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  *
#*   USA                                                                   *
#*                                                                         *
#***************************************************************************

__title__ = "FreeCAD Symoro+ Workbench - jabobians"
__author__ = "Gael Ecorchard <galou_breizh@yahoo.fr>"
__url__ = ["http://free-cad.sourceforge.net"]


def serialKinematicJacobian(joints):
    """Return the kinematic Jacobian of a serial chain

    Passive joints are considered to be as if they were actuated. The end
    joint is the last joint in the list and the base joint is its farest
    antecedent.
    """
    from chain import Chain
    from chain import get_subchain_to

    chain = Chain(get_subchain_to(joint=joints[-1], joints=joints))

    # We need the position of last joint frame in the base joints frame
    from serialmechanism import end_transform
    Tend = end_transform(chain.joints)
    Pend = Tend[0:3, 3]

    from numpy.matlib import zeros, identity, cross
    jacobian = zeros((6, len(chain.get_mjoints())))
    T = identity(4)
    k = 0
    for jnt in chain.joints:
        T *= jnt.T
		#Take P and a vectors
        P = T[0:3, 3]
        a = T[0:3, 2]
        if jnt.isrevolute():
			#revolute: [ahat*(Pend-P), a]
            DP = Pend - P
            a_hat = zeros((3,3))
            a_hat[0, 1] = -a[2]
            a_hat[0, 2] = a[1]
            a_hat[1, 0] = a[2]
            a_hat[1, 2] = -a[0]
            a_hat[2, 0] = -a[1]
            a_hat[2, 1] = a[0]
            jacobian_trans = a_hat * DP
            #jacobian_trans = cross(P, DP, axis=0)
            jacobian_rot = a
        if jnt.isprismatic():
			#prismatic: [a, 0]
            jacobian_trans = a
            jacobian_rot = zeros((3, 1))
		#skip fixed joints
        if not(jnt.isfixed()):
            jacobian[:3, k] = jacobian_trans
            jacobian[3:, k] = jacobian_rot
            k += 1
    # If the last joint is fixed, apply the Jacobian transposition
    if (joints[-1].isfixed() and not joints[-1].sameas == None):
        P = jnt.T[0:3, 2]
        jac_transposition = identity(6)
        jac_transposition[3, 1] = -P[2]
        jac_transposition[3, 2] = P[1]
        jac_transposition[4, 0] = P[2]
        jac_transposition[4, 2] = -P[0]
        jac_transposition[5, 0] = -P[1]
        jac_transposition[5, 1] = P[0]

        jacobian = jac_transposition * jacobian
    return jacobian


def serialKinematicJacobianPassive(joints):
    """Return the kinematic Jacobian of a serial chain

    Passive joints are considered to be as if they were actuated. Actuated
    joints are considered to be fixed. The end joint is the last joint in
    the list and the base joint is its farest antecedent.
    """
    from copy import deepcopy
    from joint import ACTUATED_JOINT, FIXED_JOINT

    ch_joints = deepcopy(joints)

    for jnt in ch_joints:
        if jnt.ispassive():
            # HERE was sigma instead
            jnt.mu = ACTUATED_JOINT
        else:
            jnt.sigma = FIXED_JOINT

    # Jacobians with passive and actuated joints
    jac_all = serialKinematicJacobian(ch_joints)

    # List of indexes of passive joints
    #p_indexes = [i for i, jnt in enumerate(joints) if jnt.ispassive()]
    return jac_all
