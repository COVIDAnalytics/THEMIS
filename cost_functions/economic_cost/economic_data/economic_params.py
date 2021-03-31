from collections import defaultdict

UNEMPLOYMENT_COST = {"GM": 77510,"US": 100000}
def not_implemented():
    raise NotImplementedError

# employment_impact in percentages
EMPLOYMENT_IMPACT = {}
EMPLOYMENT_IMPACT["GM"] = defaultdict(not_implemented, { "Lockdown": 2,
          "Restrict_Mass_Gatherings_and_Schools_and_Others": 1.8,
          'Authorize_Schools_but_Restrict_Mass_Gatherings_and_Others': 1.5,
          'Restrict_Mass_Gatherings_and_Schools': 1.2,
        "Restrict_Mass_Gatherings": 1,
        "No_Measure": 0})

GDP_IMPACT = {}

GDP_IMPACT["GM"] = defaultdict(not_implemented, { "Lockdown": -10,
          "Restrict_Mass_Gatherings_and_Schools_and_Others": -8,
          'Authorize_Schools_but_Restrict_Mass_Gatherings_and_Others': -6,
          'Restrict_Mass_Gatherings_and_Schools': -4,
           "Restrict_Mass_Gatherings": -3,
           "No_Measure": 0})


TOTAL_LABOR_FORCE = {"GM": 43356000}


TOTAL_GDP = {"GM": 3.861e12}

COVID_SICK_DAYS = {"GM": 7}

TOTAL_WORKING_DAYS = {"GM": 261}