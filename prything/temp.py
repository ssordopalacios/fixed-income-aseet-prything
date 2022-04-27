def coupon_bond_price(c, r, t, T):

    # You really DO need ot pass the maturities
    n = len(T)

    # Calculate value of one coupon payment at time T_i
    discounted_pmt = lambda num, r, c, T, i: num / ((1 + r[t, T[i]] / 2)) ** (
        2 * (T[i] - t)
    )

    # Sum up the value of all the coupon payments from 0 to n
    sum_coupon_pmts = [coupon_sum_fun(c, r, T, i) for i in range(n)]

    # Calculate the value of the final single principal repayment
    principal_pmt = ()
