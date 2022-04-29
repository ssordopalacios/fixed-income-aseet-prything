import numpy as np


def compound(r: float, n: int, T: int, n_infinity=True) -> float:
    """Calculate investment returns "added" into principal as time goes on

    Note
    ----
    An interest rate is not well defined without a compounding frequency.
    This process, which is the opposite of discounting, will calculate the
    result of investing $1 right now (at time t = 0) at an interest rate, r,
    for T periods where interest is compounded n times per period.

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
        whether n approaches infinity, defaults to true because we use
        continuous compounding

    Returns
    -------
    float
        The value of $1 at the end of compounding
    """

    if n_infinity:
        # $e^{r \cdot T}$
        # Think about it this way: There comes a certain r (or T, but I do not)
        # take that approach here where n gets so large that the only thing that
        # matters is the multiplication of r and T to decide the function
        ret = np.exp(r * T)
    else:
        # $(1 + \frac{r}{n})^{n \cdot t}
        # Think about it this way: (1 + r/n) gives you the amount that you get
        # for each payment. And you get n payments over T periods. So to get
        # the final answer you take the multiplication of all of them to the
        # power of n times T
        ret = np.power(1 + r / n, n * T)
    return ret


def discount(r, n, T, n_infinity=True):
    """Find value needed to receive interest rate over time T

    The amount investment now that results in $1 given a rate r compounded n
    times per period. What is the amount we have to invest today to have $1
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
        res = np.exp(-r * T)
    else:
        res = 1 / ((1 + r / n) ** (n * T))

    return res


def rate_from_discount(n, Z_T, T, n_infinity=True):
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
        res = n * (1 / (Z_T ** (1 / (n * T))) - 1)
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


def coupon_bond_price(r, t, T, c, freq=2, principal=100):
    """The price of a coupon bond at time t

    Parameters
    ----------
    r : ndarray
        array of continuously compounded yield at time for an
        investment up to time T
    T : ndarray
        array of the maturities
    c : int
        The coupon amount of the bond semi-annually
    freq : int
        The coupon payment frequency, by default 2
    principal : int, optional
        The principal amount of the bond, by default 100
    """

    # Find the number of maturities
    n = len(T)

    # Calculate the value of each coupon payment
    coupon_pmt = 0
    for i in range(1, n):
        coupon_pmt += (c / freq) / discount(
            r[t, T[i]], freq, T[i] - t, n_infinity=False
        )

    # Calculate the value of the principal payment
    principal_pmt = principal / discount(
        r[t, T[n]], freq, T[n] - t, n_infinity=False
    )

    # Return the price
    return coupon_pmt + principal_pmt


if __name__ == "__main__":
    r = np.array([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]])
    T = np.array([1, 2, 3, 4, 5])
    coupon_bond_price(r, 1, T, 0.03)

