# Authors: Michael L. Li (mlli@mit.edu), Saksham Soni, Baptiste Rossi


from cost_functions.economic_cost.economic_data.economic_params import TOTAL_GDP,GDP_IMPACT,COVID_SICK_DAYS,TOTAL_WORKING_DAYS,TOTAL_LABOR_FORCE
from dateutil.relativedelta import relativedelta
import pandas as pd
from os.path import isfile
from pathlib import Path


def short_term_gdp_costs(pandemic):
    """
    Wrapper function for returning short term GDP costs
    Parameters:
        - pandemic: Pandemic object containing the information of the region and duration that is being analyzed
    Returns:
        - Tuple (gdp_cost, sick_worker_cost)
    """
    region = pandemic.region
    gdp_data_path = Path(__file__).parent / f"economic_data/gdp/{pandemic.region}.csv"
    if region in TOTAL_GDP and region in GDP_IMPACT and isfile(gdp_data_path):
        # gdp_cost = get_gdp_cost(pandemic)
        # return gdp_cost
        return get_gdp_cost(pandemic)
    else:
        raise ValueError("Country Not Implemented")
    
    
def get_gdp_cost(pandemic):
    """
    This function returns a tuple of short term GDP cost elements, which are, 1. the percentage decrease in GDP 2. cost of sick workers.
    Parameters:
        - pandemic: Pandemic object containing the information of the region and duration that is being analyzed
    Returns:
        - Tuple (gdp_cost, sick_worker_cost)
    """
    if pandemic.policy.policy_type == "actual":
        gdp_data_path = Path(__file__).parent / f"economic_data/gdp/{pandemic.region}.csv"
        gdp_data = pd.read_csv(gdp_data_path)
        gdp_loss = []
        for i in range(pandemic.policy.num_months):
            date = pandemic.policy.start_date + relativedelta(months = i)
            year = date.year
            month = date.month
            # c + i - g + x
            monthly_gdp_loss = (gdp_data[(gdp_data.year == year) & (gdp_data.month == month)].c.values[0] + 
            gdp_data[(gdp_data.year == year) & (gdp_data.month == month)].i.values[0] +
            gdp_data[(gdp_data.year == year) & (gdp_data.month == month)].x.values[0] -
            gdp_data[(gdp_data.year == year) & (gdp_data.month == month)].g.values[0]) * 1e9
            gdp_loss.append(monthly_gdp_loss)
        gdp_cost = -sum(gdp_loss)
        return (gdp_cost, 0)
    else:
        # we are in hypothetical regime
        gdp_loss = []
        for i in range(len(pandemic.policy.policy_vector)):
            this_month_policy = pandemic.policy.policy_vector[i]
            impact_of_policy = GDP_IMPACT[pandemic.region][this_month_policy]
            monthly_gdp_loss = impact_of_policy / 100 * TOTAL_GDP[pandemic.region] / 12
            gdp_loss.append(monthly_gdp_loss)
        gdp_cost = -sum(gdp_loss)
        sick_worker_cost = pandemic.num_cases / TOTAL_LABOR_FORCE[pandemic.region] * COVID_SICK_DAYS[pandemic.region] / TOTAL_WORKING_DAYS[pandemic.region] * TOTAL_GDP[pandemic.region]
        return (gdp_cost, sick_worker_cost)