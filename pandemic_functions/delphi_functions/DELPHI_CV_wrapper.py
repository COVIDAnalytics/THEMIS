# Authors: Saksham Soni (saksham9315soni@gmail.com)

import pandas as pd
import numpy as np
from copy import deepcopy
from itertools import compress
from typing import Union, Type

from datetime import datetime, timedelta
import time
import os

from pandemic_functions.pandemic_params import *
from pandemic_functions.delphi_functions.DELPHI_utils import *
from pandemic_functions.delphi_functions.DELPHI_model_fitting import solve_and_predict_area

def evaluate_delphi_region(region: str, splits: list = ['20200515', '20200715', '20200915']):
    """
    Evaluate DELPHI for a particular region by performing 3 fold timeseries cross validation using MAPE score
    :param region: str, code for the region we are evaluating
    :param splits: list, the dates to use for train-test splits. Number of folds is equal to the length of the list
    :returns: tuple of (average MAPE on cases, average MAPE on deaths)
    """ 
    country, province = region_symbol_country_dict[region]
    country_sub = country.replace(" ", "_")
    province_sub = province.replace(" ", "_")
    assert os.path.exists(f"pandemic_functions/pandemic_data/Cases_{country_sub}_{province_sub}.csv"), \
        f"Cases data not found for region {region}"
    totalcases = pd.read_csv(
            f"pandemic_functions/pandemic_data/Cases_{country_sub}_{province_sub}.csv"
        )
    totalcases = totalcases[
            (totalcases.day_since100 >= 0) &
            (totalcases.date <= str(default_maxT))
            ].reset_index(drop=True)
    
    list_mape_cases = []
    list_mape_deaths = []
    
    for yesterday in splits:
        _, df_predictions_since_today, _, _ = \
            solve_and_predict_area(region, yesterday, None, totalcases=totalcases)
        testcases = totalcases[totalcases.date > str(pd.to_datetime(yesterday))]
        mape_cases, mape_deaths = get_mape_test_data(testcases.case_cnt, testcases.death_cnt,
                df_predictions_since_today['Total Detected'], df_predictions_since_today['Total Detected Deaths'])
        list_mape_cases.append(mape_cases)
        list_mape_deaths.append(mape_deaths)

    return np.mean(list_mape_cases), np.mean(list_mape_deaths)

def evaluate_delphi(regions: list, splits: list = ['20200515', '20200715', '20200915']):
    """
    Wrapper function to perform N fold time-series cross validation using the delphi model with dual annealing optimizer
    :param regions: list, codes for all the regions we will run the evaluation for
    :param splits: list, the dates to use for train-test splits. Number of folds is equal to the length of the list
    :returns: data frame with 3 columns -> region, mape_cases, mape_deaths
    """
    cv_results = pd.DataFrame(columns=['region', 'mape_cases', 'mape_deaths'])
    for region in regions:
        mape_cases, mape_deaths = evaluate_delphi_region(region, splits=splits)
        cv_results = cv_results.append({'region': region, 'mape_cases': mape_cases, 'mape_deaths': mape_deaths}, 
                        ignore_index=True)

    return cv_results