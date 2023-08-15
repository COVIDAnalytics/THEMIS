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

def evaluate_delphi_region(region: str, folds: list = ['20200515', '20200715', '20200915']):
    """
    Evaluate DELPHI for a particular region by performing 3 fold timeseries cross validation using MAPE score
    :param region: str, code for the region we are evaluating
    :param folds: list, the dates to use for train-test split at different folds. Number of folds is equal to the length of the list
    :returns: list of tuples of (MAPE on cases, MAPE on deaths) for each CV split
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
    
    cv_results = []
    
    for yesterday in folds:
        _, df_predictions_since_today, _, _ = \
            solve_and_predict_area(region, yesterday, None, totalcases=totalcases)
        testcases = totalcases[totalcases.date > str(pd.to_datetime(yesterday))]
        mape_cases, mape_deaths = get_mape_test_data(testcases.case_cnt, testcases.death_cnt,
                df_predictions_since_today['Total Detected'], df_predictions_since_today['Total Detected Deaths'])
        cv_results.append((mape_cases, mape_deaths))

    return cv_results

def evaluate_delphi(folds: list = ['20200515', '20200715', '20200915']):
    """
    Wrapper function to perform N fold cross validation using the delphi model with dual annealing optimizer
    :param folds: list, the dates to use for train-test split at different folds. Number of folds is equal to the length of the list
    :returns: list of results for every region, where every element is another list of tuples of (MAPE on cases, MAPE on deaths) for each CV split
    """
    cv_results = []
    for region in region_symbol_country_dict.keys():
        cv_results_region = evaluate_delphi_region(region, folds=folds)
        cv_results.append(cv_results_region)

    return cv_results