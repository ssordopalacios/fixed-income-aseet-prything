import numpy as np


def compound(r, n, T, n_infinity):
    """Calculate investment returns "added" into principal as time goes on

    Note
    ----
    An interest rate is not well defined without a compounding frequency. The
    range of possible outcomes between frequencies is infinite. This process,
    which is the opposite of discounting, will calculate the result of investing
    $1 right now (at time t = 0) at an interest rate, r, for T periods where
    interest is compounded n times per period.

    There are two equations below because the solution depends on the size of
    n that we choose. If we choose an extremely large n, then the equation
    converges to a value that can be represented in simpler terms.

    The value returned is then a percent that can be used to "scale" your value
    (i.e., the value you want to correct) to the correct amount which allows
    you to do a comparison in units that you can understand.

    Parameters
    ----------
    r : float
        the interest rate being invested at (e.g., 0.038 = 3.8%)
    n : int
        the compounding frequency (e.g., n payments per year)
    T : int
        the periods (e.g., T years) of the investment
    n_infinity : logical
        whether n approaches infinity

    Returns
    -------
    float
        The value of $1 at the end of compounding
    """

    # Use the first option if we chose n to go to infinity
    # $\lim_{n \to \infty}(1+\frac{r}{n}))^{n \cdot T} = e^{r \cdot T}$

    if n_infinity:
        # $e^{r \cdot T}$
        # Think about it this way:
        ret = np.exp(r * T)
    else:
        # $(1 + \frac{r}{n})^{n \cdot t}
        # Think about it this way: (1 + r/n) gives you the amount that you get
        # for each payment. And you get n payments over T periods. So to get
        # the final answer you take the multiplication of all of them to the
        # power of n times T
        ret = (1 + r / n) ** (n * T)
    return ret


def discount(r, n, T, n_infinity):
    """Find value needed to receive interest rate over time T

    The amount investment now that results in $1 given a rate r compounded n
    times per period

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

