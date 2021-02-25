# Authors: Michael L. Li (mlli@mit.edu),


from pyliferisk import Actuarial, Ax, annuity
from pyliferisk.mortalitytables import *
from params import VSLY, DEATHS_DIST, DEATHS_ACTUARIAL_TABLE


def deaths_costs(num_deaths, country = "GM"):
    if country in DEATHS_DIST and country in VSLY and country in DEATHS_ACTUARIAL_TABLE:
        deaths_dist = DEATHS_DIST[country]
        vsly = VSLY[country]
        life_table = DEATHS_ACTUARIAL_TABLE[country]
        mt = Actuarial(nt=life_table , i =0)
        cost_pp = 0
        for key in deaths_dist:
            percentage = deaths_dist[key]
            age_range = key.split("-")
            lower_bound = int(age_range[0])
            upper_bound = int(age_range[1])
            for i in range(lower_bound,upper_bound):
                cost_pp += annuity(mt, i, "w",1) * percentage * 1/ (upper_bound - lower_bound) * vsly
        total_cost = num_deaths * cost_pp
        return total_cost
        
    else:
        raise ValueError("Country Not Implemented")
    