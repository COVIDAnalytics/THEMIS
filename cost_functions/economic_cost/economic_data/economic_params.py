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
    
EMPLOYMENT_IMPACT["US-NY"] = defaultdict(not_implemented, { "Lockdown": 12.3,
      "Restrict_Mass_Gatherings_and_Schools_and_Others": 10.8,
      'Authorize_Schools_but_Restrict_Mass_Gatherings_and_Others': 9.3,
      'Restrict_Mass_Gatherings_and_Schools': 6.1,
    "Restrict_Mass_Gatherings": 4.8,
    "No_Measure": 0}) 


GDP_IMPACT = {}

GDP_IMPACT["GM"] = defaultdict(not_implemented, { "Lockdown": -10,
          "Restrict_Mass_Gatherings_and_Schools_and_Others": -8,
          'Authorize_Schools_but_Restrict_Mass_Gatherings_and_Others': -6,
          'Restrict_Mass_Gatherings_and_Schools': -4,
           "Restrict_Mass_Gatherings": -3,
           "No_Measure": 0})
    
# https://www.bea.gov/taxonomy/term/461
GDP_IMPACT["US-NY"] = defaultdict(not_implemented, { "Lockdown": -12,
          "Restrict_Mass_Gatherings_and_Schools_and_Others": -10,
          'Authorize_Schools_but_Restrict_Mass_Gatherings_and_Others': -7,
          'Restrict_Mass_Gatherings_and_Schools': -5,
           "Restrict_Mass_Gatherings": -4,
           "No_Measure": 0})


TOTAL_LABOR_FORCE = {"GM": 43356000, "US-NY": 9500000}


TOTAL_GDP = {"GM": 3.861e12, "US-NY": 1.77e12}

COVID_SICK_DAYS = {"GM": 7,"US-NY": 7}

TOTAL_WORKING_DAYS = {"GM": 254, "US-NY": 261}