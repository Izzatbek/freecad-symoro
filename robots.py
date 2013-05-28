from math import pi

from joint import REVOLUTE_JOINT, PRISMATIC_JOINT, FIXED_JOINT
from joint import PASSIVE_JOINT as PJOINT
from joint import ACTUATED_JOINT as AJOINT

table1 = (
    # antecedant, sameas, mu, sigma,
    #   gamma, b, alpha, d, theta, r
    (0, 0, AJOINT, REVOLUTE_JOINT,
        0, 0, 100, 0, 50, 0),
    (1, 0, AJOINT, PRISMATIC_JOINT,
        0, 0, 100, 1, 50, 0.5),
    (2, 0, AJOINT, REVOLUTE_JOINT,
        0, 0, 0, 50, pi / 2, 0),
    )

table_tree = (
    # antecedant, sameas, mu, sigma,
    #   gamma, b, alpha, d, theta, r
    (0, 0, AJOINT, REVOLUTE_JOINT,
        0, 0, 0, 0, 50, 0),
    (1, 0, AJOINT, REVOLUTE_JOINT,
        0, 0, 1, 300, 0.5, 50),
    (2, 0, AJOINT, REVOLUTE_JOINT,
        0, 0, 0, 200, pi / 2, 0),
    (3, 0, AJOINT, REVOLUTE_JOINT,
        0, 0, 0, 100, pi / 2, 0),
    (2, 0, AJOINT, REVOLUTE_JOINT,
        1, 0, 0, 300, -pi / 2, 0),
    (5, 0, AJOINT, REVOLUTE_JOINT,
        0, 0, 0, 100, pi / 2, 0),
    )

D3 = 300
RL4 = 400
table_rx90 = (
    # antecedant, sameas, mu, sigma,
    #   gamma, b, alpha, d, theta, r
    (0, 0, AJOINT, REVOLUTE_JOINT,
        0, 0, 0, 0, 0, 0),
    (1, 0, AJOINT, REVOLUTE_JOINT,
        0, 0, pi / 2, 0, 0, 0),
    (2, 0, AJOINT, REVOLUTE_JOINT,
        0, 0, 0, D3, 0, 0),
    (3, 0, AJOINT, REVOLUTE_JOINT,
        0, 0, -pi / 2, 0, 0, RL4),
    (4, 0, AJOINT, REVOLUTE_JOINT,
        0, 0, pi / 2, 0, 0, 0),
    (5, 0, AJOINT, REVOLUTE_JOINT,
        0, 0, -pi / 2, 0, 0, 0)
        )

table_stanford = (
    # antecedant, sameas, mu, sigma,
    #   gamma, b, alpha, d, theta, r
    (0, 0, AJOINT, REVOLUTE_JOINT,
        0, 0, 0, 0, 0, 0),
    (1, 0, AJOINT, REVOLUTE_JOINT,
        0, 0, -pi / 2, 0, 0, 200),
    (2, 0, AJOINT, PRISMATIC_JOINT,
        0, 0, pi / 2, 0, 0, 400),
    (3, 0, AJOINT, REVOLUTE_JOINT,
        0, 0, 0, 0, 0, 0),
    (4, 0, AJOINT, REVOLUTE_JOINT,
        0, 0, -pi / 2, 0, 0, 0),
    (5, 0, AJOINT, REVOLUTE_JOINT,
        0, 0, pi / 2, 0, 0, 0)
    )

# SR400
d2 = 400
d3 = 800
d4 = 200
rl4 = 1000
d8 = 1000
table_sr400 = (
    # antecedant, sameas, mu, sigma,
    #   gamma, b, alpha, d, theta, r
    (0, 0, AJOINT, REVOLUTE_JOINT,
        0, 0, 0, 0, 0, 0),  # 1
    (1, 0, AJOINT, REVOLUTE_JOINT,
        0, 0, -pi / 2, d2, 0, 0),  # 2
    (2, 0, PJOINT, REVOLUTE_JOINT,
        0, 0, 0, d3, 0, 0),  # 3
    (3, 0, PJOINT, REVOLUTE_JOINT,
        0, 0, -pi / 2, d4, 0, rl4),  # 4
    (4, 0, AJOINT, REVOLUTE_JOINT,
        0, 0, pi / 2, 0, 0, 0),  # 5
    (5, 0, AJOINT, REVOLUTE_JOINT,
        0, 0, -pi / 2, 0, 0, 0),  # 6
    (1, 0, AJOINT, REVOLUTE_JOINT,
        0, 0, -pi / 2, d2, 0, 0),  # 7
    (7, 0, PJOINT, REVOLUTE_JOINT,
        0, 0, 0, d8, 0, 0),  # 8
    (8, 0, PJOINT, REVOLUTE_JOINT,
        0, 0, 0, d3, 0, 0),  # 9
    (3, 9, 0, FIXED_JOINT,
        pi / 2, 0, 0, -d8, 0, 0),  # 10
        )

table_testloop_r4 = (
    # antecedant, sameas, mu, sigma,
    #   gamma, b, alpha, d, theta, r
    (0, 0, 0, FIXED_JOINT,
        0, 0, 0, 0, 0, 0),              # 1
    (1, 0, AJOINT, REVOLUTE_JOINT,
        0, 0, 0, 0, 0, 0),              # 2
    (2, 0, PJOINT, REVOLUTE_JOINT,
        0, 0, 0, 100, 0, 0),            # 3
    (1, 0, PJOINT, REVOLUTE_JOINT,
        0, 0, 0, 100, 0, 0),            # 4
    (3, 0, PJOINT, REVOLUTE_JOINT,
        0, 0, 0, 100, 0, 0),            # 5
    (4, 5, 0, FIXED_JOINT,
        pi/2, 0, 0, 100, 0, 0)          # 6
)