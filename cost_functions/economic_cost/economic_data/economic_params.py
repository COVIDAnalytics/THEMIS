from collections import defaultdict

# For countries with missing data (including Singapore), assuming roughly 1.5 year of GDP per capita
UNEMPLOYMENT_COST = {"DE": 77510,"US-NY": 100000, "US-FL": 80000, "SG": 120000, "ES": 66312, "BR": 35038 * 1.5}

def not_implemented():
    raise NotImplementedError

# employment_impact in percentages
EMPLOYMENT_IMPACT = {}
EMPLOYMENT_IMPACT["DE"] = defaultdict(not_implemented, { "Lockdown": 2,
          "Restrict_Mass_Gatherings_and_Schools_and_Others": 1.8,
          'Restrict_Mass_Gatherings_and_Schools': 1.5,
          'Authorize_Schools_but_Restrict_Mass_Gatherings_and_Others': 1.2,
        "Restrict_Mass_Gatherings": 1,
        "No_Measure": 0})
    
EMPLOYMENT_IMPACT["US-NY"] = defaultdict(not_implemented, { "Lockdown": 12.3,
      "Restrict_Mass_Gatherings_and_Schools_and_Others": 10.8,
      'Restrict_Mass_Gatherings_and_Schools': 9.3,
      'Authorize_Schools_but_Restrict_Mass_Gatherings_and_Others': 6.1,
    "Restrict_Mass_Gatherings": 4.8,
    "No_Measure": 0}) 

EMPLOYMENT_IMPACT["US-FL"] = defaultdict(not_implemented, { "Lockdown": 10.9,
      "Restrict_Mass_Gatherings_and_Schools_and_Others": 9.3,
      'Restrict_Mass_Gatherings_and_Schools': 8.4,
      'Authorize_Schools_but_Restrict_Mass_Gatherings_and_Others': 3.3,
    "Restrict_Mass_Gatherings": 2,
    "No_Measure": 0
})

EMPLOYMENT_IMPACT["SG"] = defaultdict(not_implemented, { "Lockdown": 1.3,
      "Restrict_Mass_Gatherings_and_Schools_and_Others": 1.0,
      'Restrict_Mass_Gatherings_and_Schools': 0.8,
      'Authorize_Schools_but_Restrict_Mass_Gatherings_and_Others': 0.5,
    "Restrict_Mass_Gatherings": 0.3,
    "No_Measure": 0
    })

EMPLOYMENT_IMPACT["ES"] = defaultdict(not_implemented, { "Lockdown": 2,
      "Restrict_Mass_Gatherings_and_Schools_and_Others": 1.6,
      'Restrict_Mass_Gatherings_and_Schools': 1.4,
      'Authorize_Schools_but_Restrict_Mass_Gatherings_and_Others': 0.6,
    "Restrict_Mass_Gatherings": 0.3,
    "No_Measure": 0
    })

EMPLOYMENT_IMPACT["BR"] =defaultdict(not_implemented, { "Lockdown": -3.0,
          "Restrict_Mass_Gatherings_and_Schools_and_Others": -2.7,
          'Restrict_Mass_Gatherings_and_Schools': -2.5,
          'Authorize_Schools_but_Restrict_Mass_Gatherings_and_Others': -1.9,
           "Restrict_Mass_Gatherings": -1.3,
           "No_Measure": 0})


GDP_IMPACT = {}

GDP_IMPACT["DE"] = defaultdict(not_implemented, { "Lockdown": -10,
          "Restrict_Mass_Gatherings_and_Schools_and_Others": -8,
          'Restrict_Mass_Gatherings_and_Schools': -6,
          'Authorize_Schools_but_Restrict_Mass_Gatherings_and_Others': -4,
           "Restrict_Mass_Gatherings": -3,
           "No_Measure": 0})
    
# https://www.bea.gov/taxonomy/term/461

GDP_IMPACT["US-NY"] = defaultdict(not_implemented, { "Lockdown": -12,
          "Restrict_Mass_Gatherings_and_Schools_and_Others": -10,
          'Restrict_Mass_Gatherings_and_Schools': -7,
          'Authorize_Schools_but_Restrict_Mass_Gatherings_and_Others': -5,
           "Restrict_Mass_Gatherings": -4,
           "No_Measure": 0})

GDP_IMPACT["US-FL"] = defaultdict(not_implemented, { "Lockdown": -4.5,
          "Restrict_Mass_Gatherings_and_Schools_and_Others": -4.1,
          'Restrict_Mass_Gatherings_and_Schools': -3.7,
          'Authorize_Schools_but_Restrict_Mass_Gatherings_and_Others': -1.5,
           "Restrict_Mass_Gatherings": 0.9,
           "No_Measure": 0
})
GDP_IMPACT["SG"] = defaultdict(not_implemented, { "Lockdown": -16,
          "Restrict_Mass_Gatherings_and_Schools_and_Others": -13,
          'Restrict_Mass_Gatherings_and_Schools': -10,
          'Authorize_Schools_but_Restrict_Mass_Gatherings_and_Others': -6,
           "Restrict_Mass_Gatherings": -4,
           "No_Measure": 0})

GDP_IMPACT["ES"] = defaultdict(not_implemented, { "Lockdown": -19.9,
          "Restrict_Mass_Gatherings_and_Schools_and_Others": -15.1,
          'Restrict_Mass_Gatherings_and_Schools': -13.6,
          'Authorize_Schools_but_Restrict_Mass_Gatherings_and_Others': -5.4,
           "Restrict_Mass_Gatherings": -3.3,
           "No_Measure": 0})

GDP_IMPACT["BR"] = defaultdict(not_implemented, { "Lockdown": -11.1,
          "Restrict_Mass_Gatherings_and_Schools_and_Others": -8.5,
          'Restrict_Mass_Gatherings_and_Schools': -6.0,
          'Authorize_Schools_but_Restrict_Mass_Gatherings_and_Others': -4.7,
           "Restrict_Mass_Gatherings": -3.5,
           "No_Measure": 0})

# For US https://data.bls.gov/pdq/SurveyOutputServlet
# For ES: World Bank data https://data.worldbank.org/indicator/SL.TLF.TOTL.IN?locations=ES
# Brazil: https://www.google.com/search?q=brazil+total+labor+force&sxsrf=AOaemvJD58Nn5ymZoi64Re_BOaNYyAGeyA%3A1631390603427&ei=iws9YbapGeahggeHzaLoAg&oq=brazil+total+labor&gs_lcp=Cgdnd3Mtd2l6EAMYADIGCAAQFhAeOgQIIxAnOgQIABBDOgUIABCRAjoECC4QQzoOCC4QgAQQsQMQxwEQowI6BwguELEDEEM6BwgAELEDEEM6CAgAELEDEJECOggIABCABBCxAzoICAAQyQMQkQI6CggAEIAEEIcCEBQ6DQgAEIAEEIcCELEDEBQ6BQgAEIAESgQIQRgAUJJoWM13YKZ_aABwAngAgAHnAYgBmg-SAQYxMC43LjGYAQCgAQHAAQE&sclient=gws-wiz
TOTAL_LABOR_FORCE = {"DE": 43356000, "US-NY": 9500000, "US-FL":10451550, "SG": 3750000, "ES": 22694625, "BR": 107461083}

TOTAL_GDP = {"DE": 3.861e12, "US-NY": 1.77e12, "US-FL": 1.107e12, "SG": 4.6909e11, "ES":1.245e12, "BR": 7.448e12}

COVID_SICK_DAYS = {"DE": 7,"US-NY": 7,"US-FL": 7,"SG": 7, "ES": 7, "BR": 7}

TOTAL_WORKING_DAYS = {"DE": 254, "US-NY": 261, "US-FL": 261, "SG": 261, "ES": 252, "BR": 254}
