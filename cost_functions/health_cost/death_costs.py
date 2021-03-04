# Authors: Michael L. Li (mlli@mit.edu), Saksham Soni, Baptiste Rossi


from pyliferisk import Actuarial, annuity
from pyliferisk.mortalitytables import *
from cost_functions.health_cost.health_data.health_params import VSLY, DEATHS_DIST, DEATHS_ACTUARIAL_TABLE

### should potentially consider comorbidity adjustments
def death_costs(pandemic):
    region = pandemic.region
    if region in DEATHS_DIST and region in VSLY and region in DEATHS_ACTUARIAL_TABLE:
        deaths_dist = DEATHS_DIST[region]
        vsly = VSLY[region]
        life_table = DEATHS_ACTUARIAL_TABLE[region]
        mt = Actuarial(nt=life_table , i =0)
        cost_pp = 0
        for key in deaths_dist:
            percentage = deaths_dist[key]
            age_range = key.split("-")
            lower_bound = int(age_range[0])
            upper_bound = int(age_range[1])
            for i in range(lower_bound,upper_bound):
                cost_pp += annuity(mt, i, "w",1) * percentage * 1/ (upper_bound - lower_bound) * vsly
        total_cost = pandemic.num_deaths * cost_pp
        return total_cost
        
    else:
        raise ValueError("Country Not Implemented")
    