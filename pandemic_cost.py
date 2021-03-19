
from policy_functions.policy import Policy
from cost_functions.health_cost.death_costs import death_costs
from cost_functions.health_cost.hospitalization_costs import hospitalization_costs
from cost_functions.health_cost.mental_health_costs import mental_health_costs
from cost_functions.economic_cost.short_term_economic_costs import short_term_economic_costs 
from cost_functions.economic_cost.long_term_economic_costs import long_term_economic_costs

class PandemicCost:

    def __init__(self, pandemic):

        self.st_economic_costs = short_term_economic_costs(pandemic)
        self.lt_economic_costs = long_term_economic_costs(pandemic)
        self.d_costs = death_costs(pandemic)
        self.h_costs = hospitalization_costs(pandemic)
        self.mh_costs = mental_health_costs(pandemic)
        for a in pandemic.__dict__:
            setattr(self,a, pandemic.__dict__[a])
