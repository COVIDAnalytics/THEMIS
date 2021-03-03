## Authors: Saksham Soni (sakshams@mit.edu)
import numpy as np
from params import DAILY_HOSPITALIZATION_COST 

def hospitalization_costs(active_hospitalized, active_icu, active_ventilated, country="DE"):
    inpatient_daily = DAILY_HOSPITALIZATION_COST[country]["Inpatient"]
    inpatient_daily = inpatient_daily if inpatient_daily is not None else 0
    icu_daily = DAILY_HOSPITALIZATION_COST[country]["ICU bed"]
    icu_daily = icu_daily if icu_daily is not None else 0
    ventilated_daily = DAILY_HOSPITALIZATION_COST[country]["Ventilated ICU bed"]
    ventilated_daily = ventilated_daily if ventilated_daily is not None else 0

    return icu_daily*np.nansum(active_icu) + ventilated_daily*np.nansum(active_ventilated) + inpatient_daily*np.nansum(active_hospitalized)
