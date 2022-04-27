import numpy as np


def compound(r, n, T, n_infinity):
    """Calculate investment returns "added" into principal as time goes on

    The problem is that an interest rate is not well defined without a
    compounding frequency. They result in quintessentially differenet solutions.
    This process is the opposite of discounting. There are two equations below
    because the solution depends on the size of n that we choose. If we choose
    an extremely large n, then the equation converges to a value that can be
    represented in simpler terms.

    Args:
        r (int): the interest rate
        n (int): the compounding frequency
        T (int): the horizon
        n_infinity (logical): if n approaches infinity

    Returns:
        int: The value that will be reached at the end of the compounding
    """
    if n_infinity:
        ret = np.exp(r * T)
    else:
        ret = (1 + r / n) ** (n * T)
    return ret


def discount(r, n, T, n_infinity):
    """Find value needed to receive interest rate over time T

    The question, "what is the amount we have to invest today to have $1
    dollar in the future, given a rate r compounded n times per year? This
    process is the opposite of compounding. Similar to the explanation above,
    one equation is for the normal case and the other as the solution converges
    as n approaches infinity.

    Args:
        r (int): the interest rate
        n (int): the compounding frequency
        T (int): the horizon
        n_infinity (logical): if n approaches infinity

    Returns:
        int: The discount rate
    """
    if n_infinity:
        res = np.exp(-(r ** T))
    else:
        res = 1 / (1 + r / n) ** (n * T)

    return res


def rate_from_discount(n, Z_T, T, n_infinity):
    """Given discount Z_T, derive the rate r_n for any compounding frequency n.

    We can then choose the frequencies ourselves to get us there. This will
    allow us to enter into contracts ahead of time without just "guessing"
    which one will get us to our final solution

    Args:
        n (int): the compounding frequency
        Z_T (int): The discount rate
        T (_type_): The horizon
        n_infinity (logical): if n approaches infinity

    Returns:
        int: Interest rate derviced from values
    """
    if n_infinity:
        res = (-1 / T) * np.log(Z_T)
    else:
        res = n * (1 / (Z_T ^ (1 / (n * T))) - 1)
    return res


def discount_factor(r, t, T):
    """Find the discount rate for different maturities.

    When we discount future cash flows, the discount factor at t for a dollar
    to be received at time T is given by this function. Not that from the
    variables, we can also find the time to maturity = T -t.

    Args:
        r (ndarray): array of continuously compounded yield at t for an
        investment up to T
        t (int): the calendar date when the discounting is made
        T (int): the maturity date

    Returns:
        int: The discount factor at time t for (T - t) periods
    """

    ret = np.exp(-r[t, T] * (T - t))
    return ret
