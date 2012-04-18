
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

__title__ = "FreeCAD Symoro+ Workbench - Joint"
__author__ = "Gael Ecorchard <galou_breizh@yahoo.fr>"
__url__ = ["http://free-cad.sourceforge.net"]

class Joint(object):
    def __init__(self, **kwargs):
        """
        Parameters
        ----------
        antc: Joint or None
            Antecedent joint.
        mu: 0 or 1
            0 for passive joint, 1 for actuated joint.
        sigma: 0, 1, or 2
            0 for revolute joint, 1 for prismatic one and 2 for fixed.
        gamma, b, alpha, d, theta, r: float
            Parameters of modified Denavit-Hartenberg convention
            (Khalil-Kleinfinger convention).
        qinit: float, defaults to 0
            Start value used for the optimization.
        qmin, qmax: float, defaults to None (i.e. without limit)
            Joint limits.
        q: float, optional
            Current joint value.
        """
        self.j = kwargs.get('j')
        self.antc = kwargs.get('antc')
        self.mu = kwargs.get('mu')
        self.sigma = kwargs.get('sigma')
        self.gamma = kwargs.get('gamma')
        self.b = kwargs.get('b')
        self.alpha = kwargs.get('alpha')
        self.d = kwargs.get('d')
        self.theta = kwargs.get('theta')
        self.r = kwargs.get('r')
        self._qmin = kwargs.get('qmin', -0.5)
        self._qmax = kwargs.get('qmax', 0.5)
        self.qinit = kwargs.get('qinit', 0)
        self._q = kwargs.get('q', 0)
        import numpy as np
        self._T = np.identity(4)
        # A list that holds previous parameter values, so that a change will
        # be seen and _T will be updated.
        self._prev_params = []

    @property
    def q(self):
        """The joint value"""
        return self._q

    @q.setter
    def q(self, value):
        if not(self.qmin is None):
            if (value < self.qmin):
                value = self.qmin
        if not(self.qmax is None):
            if (value > self.qmax):
                value = self.qmax
        self._q = value
        self._set_transform_antc()

    @property
    def T(self):
        """The homogeneous transformation matrix from the antecedant joint"""
        # Recompute _T if needed
        new_params = [
                self.mu,
                self.sigma,
                self.gamma,
                self.b,
                self.alpha,
                self.d,
                self.theta,
                self.r,
                ]
        if (self._prev_params != new_params):
            self._prev_params = new_params
            self._set_transform_antc()
        return self._T

    @property
    def qmin(self):
        """The minimal value allowed for q"""
        return self._qmin

    @qmin.setter
    def qmin(self, value):
        if (value is None):
            self._qmin = None
            return
        if (value > self.qmax):
            value = self.qmax
        self._qmin = value
        if (self.q < value):
            self.q = value

    @property
    def qmax(self):
        """The maximal value allowed for q"""
        return self._qmax

    @qmax.setter
    def qmax(self, value):
        if (value is None):
            self._qmax = None
            return
        if (value < self.qmin):
            value = self.qmin
        self._qmax = value
        if (self.q > value):
            self.q = value

    def __str__(self):
        return str(self.j)

    def ispassive(self):
        return (self.mu == 0)

    def isactuated(self):
        return (self.mu == 1)

    def isrevolute(self):
        return (self.sigma == 0)

    def isprismatic(self):
        return (self.sigma == 1)

    def isfixed(self):
        return (self.sigma == 2)

    def _set_transform_antc(self):
        """Modify the transform from the antecedant joint"""
        if self.isrevolute():
            theta = self.theta + self.q
            r = self.r
        elif self.isprismatic():
            theta = self.theta
            r = self.r + self.q
        else:
            theta = self.theta
            r = self.r

        from math import cos, sin
        ct = cos(theta)
        st = sin(theta)
        ca = cos(self.alpha)
        sa = sin(self.alpha)
        cg = cos(self.gamma)
        sg = sin(self.gamma)

        self._T[0, 0] = cg * ct - sg * ca * st
        self._T[0, 1] = -cg * st - sg * ca * ct
        self._T[0, 2] = sg * sa
        self._T[0, 3] = self.d * cg + r * sg * sa
        self._T[1, 0] = sg * ct + cg * ca * st
        self._T[1, 1] = -sg * st + cg * ca * ct
        self._T[1, 2] = -cg * sa
        self._T[1, 3] = self.d * sg - r * cg * sa
        self._T[2, 0] = sa * st
        self._T[2, 1] = sa * ct
        self._T[2, 2] = ca
        self._T[2, 3] = r * ca + self.b

