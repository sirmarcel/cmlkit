"""Implements the k-body MBTRs."""


from .mbtr import MBTR


class MBTR1(MBTR):
    """One-body MBTR.

    Allowed geomf:
        "unity": Returns 1.
        "count": Counts occurence of elements.
    Allowed weightf:
        "unity" : Returns the constant 1. (Typically the only useful choice.)
        "1/count": Returns 1/c, where c is number of atoms with same element type. Only k = 1.
        "identity": Returns result x of geometry function.
        "identity^2": Returns squared result x^2 of geometry function.
        "identity_root": Return square root sqrt(x) of geometry function.
        "1/identity": Returns inverse 1/x of geometry function.
        "delta_1/identity": Returns 1 if 1/x <= cutoff, 0 otherwise


    Parametrized weightf:
        "exp_-1/identity": Parameterized by length scale ls of exponential.
            Returns inverse exponentiated result of geometry function, exp( - 1 / x*ls )
        "exp_-1/identity^2": Parameterized by length scale ls of exponential.
            Returns inverse exponentiated result of squared geometry function,
            exp( - 1 / x^2*ls )

        For these, the syntax is {"name": {"ls": 1.0}}.

    For k=1, we usually only use count and unity.

    """
    kind = "mbtr_1"
    k = 1


class MBTR2(MBTR):
    """Two-body MBTR.

    Allowed geomf:
        "1/distance": Returns inverse distance 1 / ||R_a - R_b||.
        "1/dot": Returns squared inverse distance 1 / ||R_a - R_b||^2.

    Allowed weightf:
        "unity" : Returns the constant 1.
            (Will cause infinite computaiton time on periodic system.)
        "identity": Returns result x of geometry function.
        "identity^2": Returns squared result x^2 of geometry function.
        "identity_root": Return square root sqrt(x) of geometry function.
        "1/identity": Returns inverse 1/x of geometry function.
        "delta_1/identity": Returns 1 if 1/x <= cutoff, 0 otherwise

    Parametrized weightf:
        "exp_-1/identity": Parameterized by length scale ls of exponential.
            Returns inverse exponentiated result of geometry function, exp( - 1 / x*ls )
        "exp_-1/identity^2": Parameterized by length scale ls of exponential.
            Returns inverse exponentiated result of squared geometry function,
            exp( - 1 / x^2*ls )

        For these, the syntax is {"name": {"ls": 1.0}}.

    """

    kind = "mbtr_2"
    k = 2


class MBTR3(MBTR):
    """Three-body MBTR.

    Allowed geomf:
        "angle": Returns angle in radians between R_a - R_b and R_c - R_b.
        "cos_angle": Returns cos of angle between R_a - R_b and R_c - R_b.
        "dot/dotdot": Returns <u,v>/<u,u><v,v> for u = R_a - R_b and v = R_c - R_b.

    Allowed weightf:
        "unity" : Returns the constant 1.
            (Will cause infinte computation time for periodic systems.)
        "identity": Returns result x of geometry function.
        "identity^2": Returns squared result x^2 of geometry function.
        "identity_root": Return square root sqrt(x) of geometry function.
        "1/identity": Returns inverse 1/x of geometry function.
        "delta_1/identity": Returns 1 if 1/x <= cutoff, 0 otherwise.
        "1/normnorm": Returns 1/pq, where p and q are norms of
            R_a - R_b and R_c - R_b, respectively.
        "1/dotdot": Returns 1/pq, where p and q are squared norms of
            R_a - R_b and R_c - R_b, respectively.
        "1/normnormnorm": Returns 1/pqr, where p, q, r are norms of
            R_a - R_b, R_c - R_a, R_b - R_c, respectively.
        "1/dotdotdot": Returns 1/pqr, where p, q, r are squared norms of
            R_a - R_b, R_c - R_a, R_b - R_c, respectively.

    Parametrized weightf:
        "exp_-1/identity": Parameterized by length scale ls of exponential.
            Returns inverse exponentiated result of geometry function, exp( - 1 / x*ls )
        "exp_-1/identity^2": Parameterized by length scale ls of exponential.
            Returns inverse exponentiated result of squared geometry function,
            exp( - 1 / x^2*ls )
        "exp_-1/normnormnorm": Returns inverse exponentiated summed distance
            exp( - pqr / ls ), where p,q,r are norms of
            R_a - R_b, R_c - R_a, R_b - R_c, respectively.
        "exp_-1/norm+norm+norm": Returns inverse exponentiated summed distance
            exp( - (p+q+r) / ls ), where p,q,r are norms of
            R_a - R_b, R_c - R_a, R_b - R_c, respectively.

        For these, the syntax is {"name": {"ls": 1.0}}.

    """

    kind = "mbtr_3"
    k = 3


class MBTR4(MBTR):
    """Four-body MBTR. Here be dragons.

    Allowed geomf:
        "unity": Returns 1.
        "dihedral": Returns dihedral angle between four atoms.
        "cos_dihedral": Returns cos of dihedral angle between four atoms.

    Allowed weightf:
        "unity" : Returns the constant 1. (Typically the only useful choice.)
        "identity": Returns result x of geometry function.
        "identity^2": Returns squared result x^2 of geometry function.
        "identity_root": Return square root sqrt(x) of geometry function.
        "1/identity": Returns inverse 1/x of geometry function.
        "delta_1/identity": Returns 1 if 1/x <= cutoff, 0 otherwise

    """

    kind = "mbtr_4"
    k = 4
