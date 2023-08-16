#%%
import pandas as pd
import numpy as np
from datetime import datetime
import time
from pandemic_functions.delphi_functions.DELPHI_CV_wrapper import evaluate_delphi

#%%

regions = ['DE', 'US-FL', 'US-NY', 'ES', 'BR', 'SG']
split_dates = ['20200515', '20200715', '20200915']

st = time.time()
print(f"Running DELPHI evaluation for {len(regions)} regions with {len(split_dates)} splits")
cv_results = evaluate_delphi(regions, splits = split_dates)
print("Process took -- %s seconds" % (time.time()-st))
# %%

print("Saving Results...")
cv_results.to_csv('simulation_results/DELPHI_evaluation_metrics.csv')
