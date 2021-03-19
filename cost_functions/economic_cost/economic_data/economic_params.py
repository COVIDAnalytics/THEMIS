from collections import defaultdict

UNEMPLOYMENT_COST = {"GM": 77510,"US": 100000}

# employment_impact in percentages
EMPLOYMENT_IMPACT = {}
EMPLOYMENT_IMPACT["GM"] = defaultdict(lambda:0, { "Lockdown": 2,
        "Social Distancing": 1,
        "No_Measure": 0})

GDP_IMPACT = {}

GDP_IMPACT["GM"] = defaultdict(lambda:0, { "Lockdown": -10,
                                           "Social Distancing": -5,
                                           "No_Measure": 0})


TOTAL_LABOR_FORCE = {"GM": 43356000}


TOTAL_GDP = {"GM": 3.861e12}

COVID_SICK_DAYS = {"GM": 7}

TOTAL_WORKING_DAYS = {"GM": 261}