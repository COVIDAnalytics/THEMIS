
from policy_functions.policy import Policy
from cost_functions.health_cost.death_costs import death_costs
from cost_functions.health_cost.hospitalization_costs import hospitalization_costs
from cost_functions.health_cost.mental_health_costs import mental_health_costs


from cost_functions.economic_cost.short_term_economic_costs import short_term_economic_costs 
from cost_functions.economic_cost.long_term_economic_costs import long_term_economic_costs

class PandemicCost:
    
    def __init__(self,st_economic_costs, lt_economic_costs,d_costs,h_costs,mh_costs):
        
        self.st_economic_costs = st_economic_costs
        self.lt_economic_costs = lt_economic_costs
        self.d_costs = d_costs
        self.h_costs = h_costs
        self.mh_costs = mh_costs
        


def get_pandemic_cost(policy):
# apply it to the various costs we have
    st_economic_costs = short_term_economic_costs(policy)
    lt_economic_costs = long_term_economic_costs(policy)
    d_costs = death_costs(policy)
    h_costs = hospitalization_costs(policy)
    mh_costs = mental_health_costs(policy)
    pandemic_cost = PandemicCost(st_economic_costs, lt_economic_costs,d_costs,h_costs,mh_costs)
    return pandemic_cost