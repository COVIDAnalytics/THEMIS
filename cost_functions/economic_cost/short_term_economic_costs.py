# Authors: Michael L. Li (mlli@mit.edu), Saksham Soni, Baptiste Rossi


from cost_functions.economic_cost.short_term_unemployment_costs import short_term_unemployment_costs
from cost_functions.economic_cost.short_term_gdp_costs import short_term_gdp_costs



def short_term_economic_costs(pandemic):
    return short_term_unemployment_costs(pandemic) + short_term_gdp_costs(pandemic)
    # gdp_costs = short_term_gdp_costs(pandemic)
    # return {
    #   "short_term_unemployment_costs": short_term_unemployment_costs(pandemic),
    #   "short_term_gdp_impact": gdp_costs[0],
    #   "short_term_sick_worker_cost": gdp_costs[1]
    # }
    