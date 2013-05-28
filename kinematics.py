
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

__title__ = "FreeCAD Symoro+ Workbench - Kinematics"
__author__ = "Gael Ecorchard <galou_breizh@yahoo.fr>"
__url__ = ["http://free-cad.sourceforge.net"]

from joint import Joint
from chain import Chain


def table_ok(table):
    # Comparator function to order according to the antecedant index,
    # which is the first element in a joint description of the table
    # notation.
    cmp_antc = lambda j1, j2: cmp(j1[0], j2[0])
    sorted_table = sorted(table, cmp=cmp_antc)
    return (sorted_table == list(table))


def Ttomatrix(T):
    # TODO: remove FreeCAD dependency from this module
    from FreeCAD import Base
    # Flatten converts a matrix into an array and then it is converted into a list
    l = T.flatten().tolist()
    return Base.Matrix(*l)


def get_joints_from_table(table):
    joints = []
    if not(table_ok):
        raise ValueError('Antecedants should be sorted in the table')
    for j, row in enumerate(table):
        if (row[0] == 0):
            antc = None
        else:
            antc = joints[row[0] - 1]
        # TODO: redefine sameas by starting with index 0 (-1 <=> None)
        if (row[1] == 0):
            sameas = None
        else:
            sameas = joints[row[1] - 1]
        jnt = Joint(
                j=j + 1,
                antc=antc,
                sameas=sameas,
                mu=row[2],
                sigma=row[3],
                gamma=row[4],
                b=row[5],
                alpha=row[6],
                d=row[7],
                theta=row[8],
                r=row[9],
                )
        joints.append(jnt)
    # TODO: add recursion detection for antecedants
    return joints


def get_looproot(joints, joint):
    """Return the loop root for loop ending at joint

    Argument 2, joint, must be the end joint with non-None sameas attribute.
    The loop root is the last common joints of the two subchains ending at,
    respectively joint and joint.sameas.
    """
    chain = Chain(joints)
    subchain1 = chain.get_subchain_to(joint)
    subchain2 = chain.get_subchain_to(joint.sameas)

    #returns the first common antecedant for both loops
    for jnt in subchain2[::-1]:
        if (jnt in subchain1):
            return jnt
    return None


def get_loops(joints):
    """Returns a list of tuples with roots and ends (cut joints) of loops"""
    ends = []
    roots = []
    for jnt in joints:
        # If jnt's sameas attribute is another joint and that this latter is
        # not already in the list of end joints, add jnt.
        end = jnt.sameas
        if (not(end is None) and
                not(end in ends)):
            ends.append(end)
            roots.append(get_looproot(joints, jnt))
    return zip(roots, ends)


class Kinematics():
    def __init__(self, table, base_t=None, tool_t=None):
        self.table = table
        if base_t is None:
            self.base_t = [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]]
        else:
            self.base_t = base_t
        if tool_t is None:
            self.tool_t = [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]]
        else:
            self.tool_t = tool_t
        self.joints = get_joints_from_table(self.table)
        # List of actuated joints
        self.ajoints = self.get_ajoints()
        # List of passive joints
        self.pjoints = self.get_pjoints()
        # Actuated joints' variables
        self.q = self.get_q()
        # Passive joints' variables
        self.qp = self.get_qp()
        # Chains
        self.chain = Chain(self.joints)
        # Tuples (root, end) for each loop
        self.loops = get_loops(self.joints)
        # List of loop solvers
        from loop_solver import LoopSolver
        self.loopsolvers = [LoopSolver(self.joints, *l) for l in self.loops]

    # TODO: ajoints as property
    def get_ajoints(self):
        """Return the list of active joints, always in the same order"""
        ajoints = []
        for jnt in self.joints:
            if (jnt.isactuated()):
                ajoints.append(jnt)
        return ajoints

    # TODO: pjoints as property
    def get_pjoints(self):
        """Return the list of passive joints, always in the same order"""
        pjoints = []
        for jnt in self.joints:
            if (jnt.ispassive()):
                pjoints.append(jnt)
        return pjoints

    # TODO: cjoints as property
    def get_cjoints(self):
        """Return the list of cut joints, always in the same order"""
        cjoints = []
        for jnt in self.joints:
            if not(jnt.sameas is None):
                cjoints.append(jnt)

    # TODO: define q as property
    def get_q(self, index=None):
        """Returns the list of joint variables of active joints"""
        if (index is None):
            return [jnt.q for jnt in self.ajoints]
        else:
            return self.ajoints[index].q

    # TODO: define qp as property
    def get_qp(self, index=None):
        """Returns the list of joint variables of passive joints"""
        if (index is None):
            return [jnt.q for jnt in self.pjoints]
        else:
            return self.pjoints[index].q

    def set_q(self, q, index=None):
        if (index is None):
            index = range(len(self.ajoints))
        try:
            self.ajoints[index].q = q
        except (AttributeError, TypeError):
            for i in index:
                self.ajoints[i].q = q[i]

    def set_qp(self, q, index=None):
        if (index is None):
            index = range(len(self.pjoints))
        try:
            self.pjoints[index].q = q
        except (AttributeError, TypeError):
            for i in index:
                self.pjoints[i].q = q[i]

    def get_joint_transform(self, joint):
        """Return the transform from base to joint

        Return a tuple (m, Pjminus1), where m is the transformation
        matrix from base (not base joint) to the joint, and Pjminus1 is the
        position of previous joint.
        """
        # Get the list of joints from base to joint

        # The transform from the base to the base joint is given by
        # self.base_t.
        l = self.base_t[0] + self.base_t[1] + self.base_t[2] + self.base_t[3]
        from FreeCAD import Base
        m = Base.Matrix(*l)
        Pj = Base.Vector(0, 0, 0)
        for jnt in self.chain.get_subchain_to(joint):
            # Pjminus1 is Pj from the step before
            Pjminus1 = Pj
            m *= Ttomatrix(jnt.T)
            Pj = Base.Vector(m.A14, m.A24, m.A34)
        return m, Pjminus1

    def solve_loops(self):
        for ls in self.loopsolvers:
            ls.solve()
