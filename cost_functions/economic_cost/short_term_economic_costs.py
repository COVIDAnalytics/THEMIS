# Authors: Michael L. Li (mlli@mit.edu),


from .short_term_unemployment_costs import short_term_unemployment_costs
from .short_term_gdp_costs import short_term_gdp_costs



def short_term_economic_costs(policy):
    return short_term_unemployment_costs(policy) + short_term_gdp_costs(policy)
    