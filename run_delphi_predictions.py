### Get predictions for a prior date to use for CI calculations

#%%
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import os
from pandemic_functions.pandemic_params import *
from pandemic_functions.delphi_functions.DELPHI_utils import *
from pandemic_functions.delphi_functions.DELPHI_model_fitting import solve_and_predict_area

yesterday = '2020-06-02'
today = str((pd.to_datetime(yesterday) + timedelta(days=1)).date())
regions = ['DE', 'US-FL', 'US-NY', 'ES', 'BR', 'SG']
parameters_list = []
predictions_list = []

#%%
st = time.time()
print(f"Running DELPHI prediction for {len(regions)} region")
for region in regions:
    country, province = region_symbol_country_dict[region]
    country_sub = country.replace(" ", "_")
    province_sub = province.replace(" ", "_")
    totalcases = pd.read_csv(
            f"pandemic_functions/pandemic_data/Cases_{country_sub}_{province_sub}.csv"
        )
    totalcases = totalcases[
            (totalcases.day_since100 >= 0) &
            (totalcases.date <= str(default_maxT))
            ].reset_index(drop=True)
    
    df_parameters, _, df_predictions_since_100, _ = \
            solve_and_predict_area(region, yesterday, None, totalcases=totalcases)
    parameters_list.append(df_parameters)
    predictions_list.append(df_predictions_since_100)

print("Process took -- %s seconds" % (time.time()-st))
#%%
df_all_parameters = pd.concat(parameters_list)
df_all_predictions = pd.concat(predictions_list)
#%%
df_all_parameters.to_csv(f'pandemic_functions/pandemic_data/Parameters_annealing_forCI_{today}.csv', index=False)
df_all_predictions.to_csv(f'pandemic_functions/pandemic_data/DELPHI_predictions_forCI_{today}.csv', index=False)