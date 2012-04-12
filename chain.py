#!/usr/bin/python
# -*- encoding: utf-8 -*-

def get_bases(joints):
    """Return the base joints, i.e. joints without antecedant"""
    bases = []
    base_found = False
    for jnt in joints:
        if jnt.antc is None:
            base_found = True
            bases.append(jnt)
    if not(base_found):
        raise ValueError('No base is defined')
    return bases

def get_tools(joints):
    """Return the end joints, i.e. joints that are antecedant of nothing"""
    antc = [jnt.antc for jnt in joints]
    # A joint is an end joint if is antecedant of no other joints.
    tools = [jnt for jnt in joints if not(jnt in antc)]
    return tools

def get_children(joints, joint):
    """Return a list of joints which have the given joint as antecedant"""
    return [jnt for jnt in joints if (jnt.antc is joint)]

def is_unique_child(joint, joints):
    """Return True is no other joint has the same antecedant"""
    antc = [jnt.antc for jnt in joints]
    return (antc.count(joint.antc) == 1)

class Chain():
    def __init__(self, joints):
        self.joints = joints
        self.bases = get_bases(joints)
        self.tools = get_tools(joints)

    def get_subchain(self, joint):
        """Return the subchain starting at joint"""
        subjnts = get_children(self.joints, joint)
        if (len(subjnts) == 0):
            return [joint]
        elif (len(subjnts) == 1):
            return [joint] + self.get_subchain(subjnts[0])
        else:
            l = [joint]
            l.append([self.get_subchain(jnt) for jnt in subjnts])
            return l

    def get_chain(self):
        if (len(self.bases) > 1):
            return ValueError('No PKM supported yet')
        base = self.bases[0]
        return [self.bases, self.get_subchain(base)]


