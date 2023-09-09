from collections import defaultdict

# For countries with missing data (including Singapore), assuming roughly 1.5 year of GDP per capita
UNEMPLOYMENT_COST = {"DE": 77510,"US-NY": 100000, "US-FL": 80000, "SG": 120000, "ES": 66312, "BR": 35038 * 1.5}

def not_implemented():
    raise NotImplementedError

# employment_impact in percentages
EMPLOYMENT_IMPACT = {
  'BR': defaultdict(not_implemented, {'No_Measure': 0.0,
    'Restrict_Mass_Gatherings': 0.4,
    'Authorize_Schools_but_Restrict_Mass_Gatherings_and_Others': 0.7,
    'Mass_Gatherings_Authorized_But_Others_Restricted': 1.1,
    'Restrict_Mass_Gatherings_and_Schools': 1.8,
    'Restrict_Mass_Gatherings_and_Schools_and_Others': 2.0,
    'Lockdown': 2.6}),
  'DE': defaultdict(not_implemented, {'No_Measure': 0.0,
    'Restrict_Mass_Gatherings': 0.2,
    'Authorize_Schools_but_Restrict_Mass_Gatherings_and_Others': 0.3,
    'Mass_Gatherings_Authorized_But_Others_Restricted': 0.5,
    'Restrict_Mass_Gatherings_and_Schools': 0.8,
    'Restrict_Mass_Gatherings_and_Schools_and_Others': 0.9,
    'Lockdown': 1.2}),
  'ES': defaultdict(not_implemented, {'No_Measure': 0.0,
    'Restrict_Mass_Gatherings': 0.5,
    'Authorize_Schools_but_Restrict_Mass_Gatherings_and_Others': 0.8,
    'Mass_Gatherings_Authorized_But_Others_Restricted': 1.2,
    'Restrict_Mass_Gatherings_and_Schools': 1.9,
    'Restrict_Mass_Gatherings_and_Schools_and_Others': 2.1,
    'Lockdown': 2.8}),
  'SG': defaultdict(not_implemented, {'No_Measure': 0.0,
    'Restrict_Mass_Gatherings': 0.2,
    'Authorize_Schools_but_Restrict_Mass_Gatherings_and_Others': 0.3,
    'Mass_Gatherings_Authorized_But_Others_Restricted': 0.4,
    'Restrict_Mass_Gatherings_and_Schools': 0.7,
    'Restrict_Mass_Gatherings_and_Schools_and_Others': 0.7,
    'Lockdown': 1.0}),
  'US-FL': defaultdict(not_implemented, {'No_Measure': 0.0,
    'Restrict_Mass_Gatherings': 1.8,
    'Authorize_Schools_but_Restrict_Mass_Gatherings_and_Others': 2.9,
    'Mass_Gatherings_Authorized_But_Others_Restricted': 4.7,
    'Restrict_Mass_Gatherings_and_Schools': 7.4,
    'Restrict_Mass_Gatherings_and_Schools_and_Others': 8.2,
    'Lockdown': 10.8}),
  'US-NY': defaultdict(not_implemented, {'No_Measure': 0.0,
    'Restrict_Mass_Gatherings': 1.8,
    'Authorize_Schools_but_Restrict_Mass_Gatherings_and_Others': 2.9,
    'Mass_Gatherings_Authorized_But_Others_Restricted': 4.6,
    'Restrict_Mass_Gatherings_and_Schools': 7.3,
    'Restrict_Mass_Gatherings_and_Schools_and_Others': 8.0,
    'Lockdown': 10.6})
}

# GDP impact in percentage of monthly GDP
GDP_IMPACT = {
  'BR': defaultdict(not_implemented, {'No_Measure': -0.0,
    'Restrict_Mass_Gatherings': -0.5,
    'Authorize_Schools_but_Restrict_Mass_Gatherings_and_Others': -0.8,
    'Mass_Gatherings_Authorized_But_Others_Restricted': -1.2,
    'Restrict_Mass_Gatherings_and_Schools': -1.9,
    'Restrict_Mass_Gatherings_and_Schools_and_Others': -2.1,
    'Lockdown': -2.8}),
  'DE': defaultdict(not_implemented, {'No_Measure': -0.0,
    'Restrict_Mass_Gatherings': -1.4,
    'Authorize_Schools_but_Restrict_Mass_Gatherings_and_Others': -2.2,
    'Mass_Gatherings_Authorized_But_Others_Restricted': -3.5,
    'Restrict_Mass_Gatherings_and_Schools': -5.6,
    'Restrict_Mass_Gatherings_and_Schools_and_Others': -6.2,
    'Lockdown': -8.1}),
  'ES': defaultdict(not_implemented, {'No_Measure': -0.0,
    'Restrict_Mass_Gatherings': -3.5,
    'Authorize_Schools_but_Restrict_Mass_Gatherings_and_Others': -5.7,
    'Mass_Gatherings_Authorized_But_Others_Restricted': -9.2,
    'Restrict_Mass_Gatherings_and_Schools': -14.4,
    'Restrict_Mass_Gatherings_and_Schools_and_Others': -16.0,
    'Lockdown': -21.1}),
  'SG': defaultdict(not_implemented, {'No_Measure': -0.0,
    'Restrict_Mass_Gatherings': -4.0,
    'Authorize_Schools_but_Restrict_Mass_Gatherings_and_Others': -6.5,
    'Mass_Gatherings_Authorized_But_Others_Restricted': -10.5,
    'Restrict_Mass_Gatherings_and_Schools': -16.5,
    'Restrict_Mass_Gatherings_and_Schools_and_Others': -18.2,
    'Lockdown': -24.0}),
  # https://www.bea.gov/taxonomy/term/461
  'US-FL': defaultdict(not_implemented, {'No_Measure': -0.0,
    'Restrict_Mass_Gatherings': -1.3,
    'Authorize_Schools_but_Restrict_Mass_Gatherings_and_Others': -2.1,
    'Mass_Gatherings_Authorized_But_Others_Restricted': -3.5,
    'Restrict_Mass_Gatherings_and_Schools': -5.4,
    'Restrict_Mass_Gatherings_and_Schools_and_Others': -6.0,
    'Lockdown': -7.9}),
  'US-NY': defaultdict(not_implemented, {'No_Measure': -0.0,
    'Restrict_Mass_Gatherings': -1.4,
    'Authorize_Schools_but_Restrict_Mass_Gatherings_and_Others': -2.3,
    'Mass_Gatherings_Authorized_But_Others_Restricted': -3.7,
    'Restrict_Mass_Gatherings_and_Schools': -5.7,
    'Restrict_Mass_Gatherings_and_Schools_and_Others': -6.3,
    'Lockdown': -8.4})
  }

# For US https://data.bls.gov/pdq/SurveyOutputServlet
# For ES: World Bank data https://data.worldbank.org/indicator/SL.TLF.TOTL.IN?locations=ES
# Brazil: https://www.google.com/search?q=brazil+total+labor+force&sxsrf=AOaemvJD58Nn5ymZoi64Re_BOaNYyAGeyA%3A1631390603427&ei=iws9YbapGeahggeHzaLoAg&oq=brazil+total+labor&gs_lcp=Cgdnd3Mtd2l6EAMYADIGCAAQFhAeOgQIIxAnOgQIABBDOgUIABCRAjoECC4QQzoOCC4QgAQQsQMQxwEQowI6BwguELEDEEM6BwgAELEDEEM6CAgAELEDEJECOggIABCABBCxAzoICAAQyQMQkQI6CggAEIAEEIcCEBQ6DQgAEIAEEIcCELEDEBQ6BQgAEIAESgQIQRgAUJJoWM13YKZ_aABwAngAgAHnAYgBmg-SAQYxMC43LjGYAQCgAQHAAQE&sclient=gws-wiz
TOTAL_LABOR_FORCE = {"DE": 43356000, "US-NY": 9500000, "US-FL":10451550, "SG": 3750000, "ES": 22694625, "BR": 107461083}

TOTAL_GDP = {"DE": 3.861e12, "US-NY": 1.77e12, "US-FL": 1.107e12, "SG": 4.6909e11, "ES":1.245e12, "BR": 7.448e12}

COVID_SICK_DAYS = {"DE": 7,"US-NY": 7,"US-FL": 7,"SG": 7, "ES": 7, "BR": 7}

TOTAL_WORKING_DAYS = {"DE": 254, "US-NY": 261, "US-FL": 261, "SG": 261, "ES": 252, "BR": 254}
