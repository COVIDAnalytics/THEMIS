from policy_functions.policy import Policy
from cost_functions.health_cost.death_costs import death_costs
from cost_functions.health_cost.hospitalization_costs import hospitalization_costs
from cost_functions.health_cost.mental_health_costs import mental_health_costs
from cost_functions.economic_cost.short_term_economic_costs import short_term_economic_costs 
from cost_functions.economic_cost.long_term_economic_costs import long_term_economic_costs

class PandemicCost:
    """Wrapper class to calculate and store all cost components for a pandemic scenario in a given region"""
    def __init__(self, pandemic):
        """Function that initializes the object by calculating all cost components"""
        self.st_term_unemployment_costs, self.st_term_gdp_impact, st_term_sick_worker_cost_ci = short_term_economic_costs(pandemic)
        self.st_term_sick_worker_cost, self.st_term_sick_worker_cost_lb, self.st_term_sick_worker_cost_ub = tuple(st_term_sick_worker_cost_ci)
        # short term economic cost, result is an np.array of size 3 (value, lower bound, upper bound)
        st_economic_costs_ci = self.st_term_unemployment_costs + self.st_term_gdp_impact + st_term_sick_worker_cost_ci
        self.st_economic_costs, self.st_economic_costs_lb, self.st_economic_costs_ub = tuple(st_economic_costs_ci)
        self.lt_economic_costs = long_term_economic_costs(pandemic)
        self.d_costs, self.d_costs_lb, self.d_costs_ub  = death_costs(pandemic)
        self.h_costs = hospitalization_costs(pandemic)
        self.mh_costs, self.mh_costs_lb, self.mh_costs_ub = mental_health_costs(pandemic)
        for a in pandemic.__dict__:
            setattr(self,a, pandemic.__dict__[a])
