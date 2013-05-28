
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

__title__ = "FreeCAD Symoro+ Workbench - LoopSolver"
__author__ = "Gael Ecorchard <galou_breizh@yahoo.fr>"
__url__ = ["http://free-cad.sourceforge.net"]


def remove_upto_root(chain, root):
    """Remove all joints before root and root itself in a chain"""
    for jnt in chain:
        if not(jnt is root):
            chain.pop(0)
        else:
            break
    # Also remove root
    chain.pop(0)


def l_from_l_of_l(ll):
    """Return a list from a list of lists"""
    return [item for sublist in ll for item in sublist]


def frame_diff(T1, T2):
    """Return the difference between two frames given their transformation

    Return the difference (translation and rotation) between two frames given
    their homogeneous transformation matrices T1 and T2 from a common frame.
    The rotation is expressed by a vector representing the axis of rotation,
    and which norm is the rotation angle.
    """
    # TODO: add clear reference to Khalil
    dtrans = T2[0:3, 3] - T1[0:3, 3]

    # TODO: write (find?) a module for homogeneous transforms
    # Transpose of the rotational part of T1
    Rt = T1[0:3, 0:3].transpose()
    from numpy.matlib import identity
    invT1 = identity(4)
    invT1[0:3, 0:3] = Rt
    invT1[0:3, 3] = -Rt * T1[0:3, 3]
    # Transform from 1 to 2
    T = invT1 * T2

    sx = T[0, 0]
    sy = T[1, 0]
    sz = T[2, 0]
    nx = T[0, 1]
    ny = T[1, 1]
    nz = T[2, 1]
    ax = T[0, 2]
    ay = T[1, 2]
    az = T[2, 2]

    from math import sqrt, atan2
    cos_theta = 0.5 * (sx + ny + az - 1)
    sin_theta = 0.5 * sqrt(
            (nz - ay) ** 2 +
            (ax - sz) ** 2 +
            (sy - nx) ** 2)
    theta = atan2(sin_theta, cos_theta)

    from numpy import matrix
    if abs(sin_theta) < 1e-8:
        u = matrix([0.0, 0.0, 1.0]).transpose()
    else:
        u = matrix([(nz - ay) / (2 * sin_theta),
            (ax - sz) / (2 * sin_theta),
            (sy - nx) / (2 * sin_theta)]).transpose()
    drot = theta * u

    from numpy.matlib import zeros
    dx = zeros((6, 1))
    dx[0:3] = dtrans
    dx[3:6] = drot
    return dx


class LoopSolver():
    def __init__(self, joints, root, end):
        self.joints = joints
        self.root = root
        self.end = end
        if not(self.root_ok()):
            raise ValueError('Root joint is not correct')
        if not(self.end_ok()):
            raise ValueError('End joint is not correct')
        self.end_joints = self.get_end_joints()
        self.nloops = len(self.end_joints)
        self.chains = self.get_chains()
        self.ajoints = self.get_ajoints()
        self.pjoints = self.get_pjoints()

    def root_ok(self):
        """Check that the root is opening the loop"""
        return True

    def end_ok(self):
        """Check that the end joint is closing the loop"""
        # TODO: also allow that self.end has attribute sameas not None by
        # finding the joint which has sameas equal to the given end.
        if not(self.end.sameas is None):
			if not(self.end.sameas in self.joints and self.end.sameas.sameas is self.end):
				return False
        check_ok = False
        for jnt in self.joints:
            if jnt.sameas is self.end:
                check_ok = True
                break
        # TODO: put this check somewhere else
        for jnt in self.joints:
            if (not(jnt.sameas is None) and
                    not(jnt.sameas is self.end)):
                check_ok = False
        return check_ok

    def get_end_joints(self):
        """Return the list of self.end and all joints with sameas == end."""
        l = ([self.end] +
                [jnt for jnt in self.joints if jnt.sameas is self.end])
        return l

    def get_chains(self):
        """Return a list of serial chains constituting the mechanism

        Each serial chain starts with the joint following self.root and
        ends with self.end or a joint with sameas == self.end.
        """
        from chain import Chain
        chain = Chain(self.joints)

        chains = []
        for end_joint in self.end_joints:
            subchain = chain.get_subchain_to(end_joint)
            remove_upto_root(subchain, self.root)
            chains.append(subchain)
        return chains

    def get_ajoints(self):
        """Return a list of actuated joints in each subchain
        """
        ajoints = []
        for subchain in self.chains:
            jnts = []
            for jnt in subchain:
                if jnt.isactuated():
                    jnts.append(jnt)
            ajoints.append(jnts)
        return ajoints

    def get_pjoints(self):
        """Return a list of passive joints in each subchain
        """
        pjoints = []
        for subchain in self.chains:
            jnts = []
            for jnt in subchain:
                if jnt.ispassive():
                    jnts.append(jnt)
            pjoints.append(jnts)
        return pjoints

    def random_qinit(self):
        """Set random values for joint.qinit"""
        from numpy.random import random
        from math import pi

        for jnt in self.joints:
            if (jnt.ispassive()):
                if jnt.isprismatic():
                    jnt.qinit = 300 * random() - 150
                else:
                    jnt.qinit = 2 * pi * random() - pi

    def get_cjoint_diff(self):
        """Return the frame difference at a cut joint"""
        # The first cut joint will be used as reference.
        from serialmechanism import end_transform
        T0 = end_transform(self.chains[0])
        # The other cut joints will be used to make a difference from the
        # first cut joint.
        from numpy.matlib import zeros
        dx = zeros((6 * (len(self.chains) - 1), 1))
        for k, chain in enumerate(self.chains[1:]):
            T = end_transform(chain)
            dx[(k * 6):(k * 6 + 6)] = frame_diff(T0, T)
        return dx

    def get_cjoint_jac(self):
        from numpy.matlib import zeros
        from jacobians import serialKinematicJacobianPassive as jacobian

        n_endjoints = len(self.chains)
        from serialmechanism import n_pjoints
        n_pjnts = n_pjoints(l_from_l_of_l(self.pjoints))
        J = zeros((6 * (n_endjoints - 1), n_pjnts))

        # Put J0 more times in the first columns.
        J0 = jacobian(self.chains[0])
        n_pjoints0 = J0.shape[1]
        for i in range(n_endjoints - 1):
            J[(6 * i):(6 * i + 6), :n_pjoints0] = J0

        # Put -J1, -J2, ... in the diagonal of the non-yet-filled rest part
        # of J.
        np = n_pjoints0
        for i in range(1, n_endjoints):
            npi = n_pjoints(self.chains[i]) # MAYBE HERE should be # of passive joints
            Ji = jacobian(self.chains[i])
            J[(6 * (i - 1)):(6 * i), np:(np + npi)] = -Ji
            np += npi
        return J

    def set_dq_pjoints(self, dq):
        pjoints = l_from_l_of_l(self.pjoints)
        for i in range(len(pjoints)):
            jnt = pjoints[i]
            jnt.q += dq[i, 0]
            # TODO: remainder by 2 * pi and check limits.
            # Beware not to limit revolute joints if (q +/- 2*pi) would be
            # within the limits.

    def solve(self):
        kmax = 10
        jmax = 100
        dpos_min = 1e-5

        #from numpy.linalg import pinv

        for k in range(kmax):
            print('Initial values set number: {0}/{1}'.format(k, kmax))
            # Use joint.qinit the first attempt. If the attempt fails, try
            # with random values.
            if (k > 1):
                self.random_qinit()
            # TODO: utiliser scipy.optimize
            for j in range(jmax):
                dx = self.get_cjoint_diff()
                dpos = dx.reshape((-1, 6))[:, 0:3]
                if (abs(dpos).max() < dpos_min):
                    return
                J = self.get_cjoint_jac()
                # TODO: study the use of (dx/2) instead of dx.
                # Demander a Wisama s'il ne serait pas mieux de prendre
                # (dx/2), sinon but pour nouveau X0 c'est atteindre ancien X1
                # et inversement, alors qu'il faudrait avoir but pour nouveau
                # X0 c'est atteindre nouveau X1.
                dq = ((J.T * J) ** -1) * J.T * dx
                #dq = pinv(J) * dx
                self.set_dq_pjoints(dq)
