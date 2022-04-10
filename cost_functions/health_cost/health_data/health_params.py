# Value of Statistical Life Year from sources
# For those with only VSL, we use the formula on page 14 of https://www.who.int/management/EconomicBenefitofTuberculosisControl.pdf and 3% discount rate to adjust
# Brazil 2013 VSL: https://larrlasa.org/articles/10.25222/larr.61/
# Brazil Life expectancy: https://data.worldbank.org/indicator/SP.DYN.LE00.IN?locations=BR

VSLY = {"DE": 158448, "US": 325000, "US-NY": 325000,
        "US-FL": 325000,
        "SG": 46987 * 114.87/90.,
        # US dollar conversion, first convert US inflation, then convert exchange rate
        "BR": 39939 * 260 / 220 * 5.25,
        "ES": 158448
        }

DEATHS_DIST = {
       "DE": {"0-10": 9/61951,
              "10-20": 4/61951,
              "20-29": 46/61951,
              "30-39": 92/61951,
              "40-49": 342/61951,
              "50-59": 1565/61951,
              "60-69": 4646/61951,
              "70-79": 11849/61951,
              "80-89": 28948/61951,
              "90-100": 14450/61951}, 
       "US": {"0-17": 204/478912,
              "18-29": 1684/478912,
              "30-39": 5030/478912,
              "40-49": 13482/478912,
              "50-64": 70160/478912,
              "65-74": 103451/478912,
              "75-84": 133557/478912,
              "85-100": 151344/478912
                     },
       # https://covid19tracker.health.ny.gov/views/NYS-COVID19-Tracker/NYSDOHCOVID-19Tracker-Fatalities?%3Aembed=yes&%3Atoolbar=no&%3Atabs=n
       "US-NY": {"0-9": 14/40513,
              "10-19": 12/40513,
              "20-29": 137/40513,
              "30-39": 453/40513,
              "40-49": 1156/40513,
              "50-59": 3287/40513,
              "60-69": 7220/40513,
              "70-79": 10531/40513,
              "80-89": 11094/40513,
              "90-100": 6600/40513
                     },
       # http://ww11.doh.state.fl.us/comm/_partners/covid19_report_archive/cases-monitoring-and-pui-information/county-report/county_reports_latest.pdf
       "US-FL": {"0-4": 1/35700,
              "5-14": 5/35700,
              "15-24": 48/35700,
              "25-34": 210/35700,
              "35-44": 566/35700,
              "45-54": 1431/35700,
              "55-64": 3971/35700,
              "65-74": 7629/35700,
              "75-84": 10797/35700,
              "84-100": 11042/35700,
                     },    
       # very crude estimate through cases distribution here https://pubmed.ncbi.nlm.nih.gov/33283840/
       # and death rate estimation here:  https://pubmed.ncbi.nlm.nih.gov/33081241/
       "SG": {"0-9": 0/31,
              "10-19": 0/31,
              "20-29": 0/31,
              "30-39": 0/31,
              "40-49": 0/31,
              "50-59": 4/31,
              "60-69": 3/31,
              "70-100": 24/31,
                     },    
       "ES": {"0-9":0.0,
              "10-19": 0.0,
              "20-29": 0.1,
              "30-39": 0.3,
              "40-49": 1.0,
              "50-59": 3.7,
              "60-69": 9.7,
              "70-79": 22.0,
              "80-100": 63.1,
                     }, 
       # estimate based on estimates from this  https://portal.fiocruz.br/sites/portal.fiocruz.br/files/documentos/boletim_covid_2021-semanas_10-11-red.pdf
       "BR": {"0-9": 0.0015,
              "10-20": 0.0015,
              "20-30": 0.007,
              "30-40": 0.04,
              "40-50": 0.07,
              "50-60": 0.13,
              "60-70": 0.24,
              "70-80": 0.28,
              "80-90": 0.18,
              "90-100": 0.05
              },       
}

DEATHS_ACTUARIAL_TABLE = {"DE": [
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.5785, 1.5951, 1.6006, 1.595, 1.5785, 1.5503, 1.5094, 1.4643, 1.4238,
1.388, 1.3574, 1.3325, 1.3137, 1.3018, 1.2968, 1.2995, 1.3104, 1.3299, 1.3586, 1.397, 1.4454, 1.5045, 1.5754, 1.6591,
1.7566, 1.8694, 1.9983, 2.1445, 2.3096, 2.497, 2.7107, 2.9545, 3.2325, 3.5482, 3.9057, 4.3087, 4.7606, 5.2655, 5.8269,
6.4474, 7.1294, 7.8756, 8.6884, 9.5704, 10.5241, 11.5521, 12.6571, 13.8417, 15.1083, 16.4598, 18.0706, 20.0313, 22.3416,
25.0018, 28.0117, 31.3714, 35.0808, 39.14, 43.549, 48.3078, 53.4163, 58.8745, 64.6826, 70.8404, 77.348, 84.2053,
91.4124, 98.9693, 106.876, 115.1324, 123.7386, 132.6945, 142.0002, 151.6557, 161.6609, 172.016, 182.7207, 193.7753,
205.1796, 216.9337, 229.0375, 241.4911, 254.2945, 267.4477, 280.9506, 294.8032, 309.0057, 323.5579, 338.4599, 353.7116,
369.3131, 385.2644, 401.5655, 418.2163, 435.2169, 452.5672, 470.2673, 488.3172, 506.7169, 525.4663, 544.5654, 564.0144,
583.8131, 603.9616, 624.4598, 1000], 
"US": [0, 5.777,0.38200000000000006,0.24800000000000003,0.193,0.149,0.14100000000000001,0.126,0.114,0.104,0.095,0.093,0.10300000000000001,
 0.133,0.186,0.25799999999999995,0.33799999999999997,0.421,0.51,0.603,0.698,0.795,0.889,0.97,1.0319999999999998,1.08,1.1230000000000002,
 1.165,1.207,1.252,1.3,1.351,1.402,1.454,1.5059999999999998,1.556,1.6149999999999998,1.6789999999999998,1.7399999999999998,1.7979999999999998,
 1.8599999999999999,1.936,2.036,2.16,2.306,2.4699999999999998,2.647,2.846,3.079,3.357,3.6819999999999995,4.029999999999999,4.401000000000001,
 4.82,5.285,5.778,6.284,6.7940000000000005,7.319,7.869,8.456,9.093,9.768,10.466999999999999,11.181000000000001,11.921999999999999,12.71,13.620999999999999,
 14.62,15.77,17.1,18.428,20.317000000000004,22.102,24.194,26.342,29.041999999999998,32.001,35.443000000000005,39.257,43.393,48.163,53.216,59.24,66.564,
 74.045,81.954,90.87899999999999,101.93799999999999,114.07499999999999,127.331,141.733,157.28900000000002,173.98600000000002,191.78799999999998,210.63299999999998,
 230.432,251.066,272.395,294.253,316.456,1000.0],
 "US-NY": [0, 5.777,0.38200000000000006,0.24800000000000003,0.193,0.149,0.14100000000000001,0.126,0.114,0.104,0.095,0.093,0.10300000000000001,
 0.133,0.186,0.25799999999999995,0.33799999999999997,0.421,0.51,0.603,0.698,0.795,0.889,0.97,1.0319999999999998,1.08,1.1230000000000002,
 1.165,1.207,1.252,1.3,1.351,1.402,1.454,1.5059999999999998,1.556,1.6149999999999998,1.6789999999999998,1.7399999999999998,1.7979999999999998,
 1.8599999999999999,1.936,2.036,2.16,2.306,2.4699999999999998,2.647,2.846,3.079,3.357,3.6819999999999995,4.029999999999999,4.401000000000001,
 4.82,5.285,5.778,6.284,6.7940000000000005,7.319,7.869,8.456,9.093,9.768,10.466999999999999,11.181000000000001,11.921999999999999,12.71,13.620999999999999,
 14.62,15.77,17.1,18.428,20.317000000000004,22.102,24.194,26.342,29.041999999999998,32.001,35.443000000000005,39.257,43.393,48.163,53.216,59.24,66.564,
 74.045,81.954,90.87899999999999,101.93799999999999,114.07499999999999,127.331,141.733,157.28900000000002,173.98600000000002,191.78799999999998,210.63299999999998,
 230.432,251.066,272.395,294.253,316.456,1000.0], 
"US-FL": [0, 5.777,0.38200000000000006,0.24800000000000003,0.193,0.149,0.14100000000000001,0.126,0.114,0.104,0.095,0.093,0.10300000000000001,
 0.133,0.186,0.25799999999999995,0.33799999999999997,0.421,0.51,0.603,0.698,0.795,0.889,0.97,1.0319999999999998,1.08,1.1230000000000002,
 1.165,1.207,1.252,1.3,1.351,1.402,1.454,1.5059999999999998,1.556,1.6149999999999998,1.6789999999999998,1.7399999999999998,1.7979999999999998,
 1.8599999999999999,1.936,2.036,2.16,2.306,2.4699999999999998,2.647,2.846,3.079,3.357,3.6819999999999995,4.029999999999999,4.401000000000001,
 4.82,5.285,5.778,6.284,6.7940000000000005,7.319,7.869,8.456,9.093,9.768,10.466999999999999,11.181000000000001,11.921999999999999,12.71,13.620999999999999,
 14.62,15.77,17.1,18.428,20.317000000000004,22.102,24.194,26.342,29.041999999999998,32.001,35.443000000000005,39.257,43.393,48.163,53.216,59.24,66.564,
 74.045,81.954,90.87899999999999,101.93799999999999,114.07499999999999,127.331,141.733,157.28900000000002,173.98600000000002,191.78799999999998,210.63299999999998,
 230.432,251.066,272.395,294.253,316.456,1000.0], 
  # https://www.singstat.gov.sg/-/media/files/publications/population/lifetable17-18.pdf
  "SG": [0, 2.380, 0.12,0.12,0.11,0.10,0.09,0.07,0.07,0.07,0.08,0.09,0.10,0.12,0.14,0.16,0.18,0.19,0.21,0.22,0.23,0.23,0.24,0.25,0.25,0.25,0.25,0.26,0.28,0.29,0.31,
         0.33,0.35,0.37,0.39,0.42,0.44,0.48,0.53,0.60,0.66,0.73,0.82,0.92,1.04,1.15,1.28,1.42,1.60,1.80,2.01,2.22,2.45,2.73,3.03,3.33,3.65,4.00,4.41,4.86,5.31,5.78,
         6.32,6.95,7.63,8.32,9.06,9.97,11.13,12.43,13.78,15.20,16.88,18.93,21.19,23.47,25.93,29.01,33.05,37.67,42.41,47.20,52.35,58.24,64.85,72.01,79.85,88.44,97.83,108.06,
         119.21,131.32,144.44,158.65,173.98,190.49,208.43,227.24,247.55,269.21,1000],
  "ES": [0, 2.593024,  0.205948,  0.135425,  0.099742,  0.099849,  0.058782,  0.055271,  0.06131,  0.049577,  0.06303,  0.076434,  0.077715,  0.045578,  0.10625,  0.105689,  0.102791,  
              0.143248,  0.154397,  0.187431,  0.204627,  0.212298,  0.219429,  0.220012,  0.245406,  0.255186,  0.297293,  0.26258,  0.303547,  0.28201,  0.301968,  0.337795,  0.376289,  
              0.368766,  0.349087,  0.412317,  0.428481,  0.445038,  0.520144,  0.580685,  0.650231,  0.636204,  0.691003,  0.799377,  0.887735,  0.922658,  1.108386,  1.25929,  1.496959,  
              1.664625,  1.907271,  2.058129,  2.411505,  2.658061,  2.926607,  3.330422,  3.570359,  4.06749,  4.312569,  4.513993,  5.144025,  5.625559,  6.21041,  6.630208,  7.132897,  
              7.85391,  8.372769,  9.138525,  9.889097,  10.282475,  11.244173,  12.12326,  13.949964,  14.714084,  16.696368,  18.193695,  20.244111,  22.346994,  23.664038,  27.290029,  
              31.257353,  36.112527,  41.159935,  46.795242,  53.103816,  60.819787,  70.222704,  80.183774,  91.483165,  105.109125,  117.800803,  137.716885,  153.33197,  174.645413,  
              191.999363,  209.158792,  232.176845,  254.090063,  282.003067,  292.16614,  304.572014,  1000],
  # https://www.ibge.gov.br/en/statistics/social/population/17117-complete-life-tables.html?=&t=resultados
  # https://apps.who.int/gho/data/view.main.60220?lang=en
 "BR": [0, 11.9376, 0.796, 0.518, 0.397, 0.328, 0.283, 0.252, 0.231, 0.219, 0.215, 0.219, 0.235, 0.267, 0.321, 0.408, 0.683, 0.850, 0.998, 1.114, 1.203, 1.292, 1.380, 1.440, 1.466, 1.466, 1.455,
           1.448, 1.452, 1.475, 1.514, 1.559, 1.604, 1.654, 1.708, 1.768, 1.838, 1.920, 2.015, 2.122, 2.244, 2.380, 2.533, 2.710, 2.914, 3.143, 3.394, 3.664, 3.953, 4.261, 4.588, 4.942, 5.323, 5.728, 
           6.158, 6.616, 7.114, 7.652, 8.222, 8.825, 9.470, 10.171, 10.943, 11.797, 12.747, 13.799, 14.936, 16.178, 17.578, 19.168, 20.941, 22.855, 24.914, 27.178, 29.675, 32.409, 35.345, 38.500, 
           41.953, 45.753, 49.912, 54.590, 59.879, 64.867, 69.954, 76.859, 83.125, 93.025, 104.95,116.80, 131.21,142.95, 155.91, 165.70, 183.28, 198.21, 217.17, 236.95, 256.71, 277.59, 300.15, 324.55, 1000],
}

# Hospitalization Costs Per Country
DAILY_HOSPITALIZATION_COST = {
       "DE": {"Inpatient": 162.65, "ICU bed": 795, "Ventilated ICU bed": 1539, "Currency": "euro"},
       # https://www.bcbs.com/coronavirus-updates/stories/infographic-covid-19-patients-high-risk-conditions-3x-more-likely-need-the-icu
       # corrected ventilated using  https://pubmed.ncbi.nlm.nih.gov/15942342/ with 3968/3184
       "US-NY": {"Inpatient": 33750/15, "ICU bed": 84375/15, "Ventilated ICU bed": 84375/15*3968/3184, "Currency": "USD"},
       "US-FL": {"Inpatient": 33750/15, "ICU bed": 84375/15, "Ventilated ICU bed": 84375/15*3968/3184, "Currency": "USD"},
       # IFHP - 2015 Comparitive Price Report, accessed through https://www.statista.com/statistics/312022/cost-of-hospital-stay-per-day-by-country/
       # scaled cost of ICU with ventilator using the German costs for ICU v/s ICU with ventilator
       "ES": {"Inpatient": 424/1.2, "ICU bed": 1700, "Ventilated ICU bed": 1700*1.59, "Currency": "euro"},
       # https://blog.seedly.sg/the-true-cost-of-healthcare-in-singapore-that-every-singaporean-should-be-aware-of/
       # https://www.ktph.com.sg/patients/hospital-charges
       "SG": {"Inpatient": 520, "ICU bed": 520 * 3, "Ventilated ICU bed": 520 * 3 *3968/3184, "Currency": "SGD"},
       # https://www.revistas.usp.br/rsp/article/view/189611/175086
       # we only have the average across all patients, so assumed average
       "BR": {"Inpatient": 4864.2/8.2, "ICU bed": 4864.2/8.2, "Ventilated ICU bed": 4864.2/8.2, "Currency": "reals"}
}

# Mental Health Parameters per Country

MENTAL_HEALTH_COST = {
    "DE": {"exposed_health_workers": 892000,
           "gen_population_over14": 72520000,
           "depression_rate_baseline": 7.6/100.,
           "ptsd_rate_baseline": 2.31/100.,
           "depression_rate_inc_sick": 6.7/100.0,
           "depression_rate_inc_hworkers": 6.7/100.,
           "depression_rate_inc_gen_population": 6.7/100.,
           "ptsd_rate_inc_hworkers": 6.7/100.,
           "ptsd_rate_inc_sick": 6.7/100.,
           "depression_cost": 4000.,
           "ptsd_cost": 40000.,
           "lockdown_equivalent_policies": ['Lockdown'],
           "Currency": "euro"
    },
    # https://www.chwsny.org/wp-content/uploads/2018/04/Full_CHWS_NY_Tracking_Report-2018b-1.pdf
    # only count hospitals and nursing home professionals
    "US-NY": {"exposed_health_workers": 600000,
              # https://www.health.ny.gov/statistics/vital_statistics/2018/table01.htm
              "gen_population_over14": 16164571,
              # https://jamanetwork.com/journals/jamanetworkopen/fullarticle/2770146
              "depression_rate_baseline": 8.5/100.,
              # https://www.nimh.nih.gov/health/statistics/post-traumatic-stress-disorder-ptsd
              "ptsd_rate_baseline": 3.6/100.,
              # Assume same as healthcare workers for now
              "depression_rate_inc_sick": 20.2/100.0,
              "depression_rate_inc_hworkers": 20.2/100.,
              "depression_rate_inc_gen_population": 20.2/100.,
              # https://www.psychiatryadvisor.com/home/topics/general-psychiatry/quantifying-the-rates-of-distress-among-health-care-workers-during-the-covid-19-pandemic/
              # https://www.cdc.gov/mmwr/volumes/70/wr/mm7048a6.htm?s_cid=mm7048a6_w#T1_down
              "ptsd_rate_inc_hworkers": 33.2/100.,
              # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7263263/
              "ptsd_rate_inc_sick": 28.2/100.,
              # 2010 dollars, adjusting for inflation
              "depression_cost": 27688*1.37,
              "ptsd_cost": 14857.,
              "lockdown_equivalent_policies": ['Lockdown'],
              "Currency": "USD"
              },
       "US-FL": {
              # https://bhw.hrsa.gov/sites/default/files/bureau-health-workforce/data-research/state-profiles/florida-2018.pdf
              "exposed_health_workers": 624451,
              # projected 2020 population http://edr.state.fl.us/content/population-demographics/data/pop_census_day.pdf
              "gen_population_over14": 17696804,
              # https://jamanetwork.com/journals/jamanetworkopen/fullarticle/2770146
              "depression_rate_baseline": 8.5/100.,
              # https://www.nimh.nih.gov/health/statistics/post-traumatic-stress-disorder-ptsd
              "ptsd_rate_baseline": 3.6/100.,
              # Assume same as healthcare workers for now
              "depression_rate_inc_sick": 20.2/100.0,
              "depression_rate_inc_hworkers": 20.2/100.,
              "depression_rate_inc_gen_population": 20.2/100.,
              # https://www.psychiatryadvisor.com/home/topics/general-psychiatry/quantifying-the-rates-of-distress-among-health-care-workers-during-the-covid-19-pandemic/
              # https://www.cdc.gov/mmwr/volumes/70/wr/mm7048a6.htm?s_cid=mm7048a6_w#T1_down
              "ptsd_rate_inc_hworkers": 33.2/100.,
              # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7263263/
              "ptsd_rate_inc_sick": 28.2/100.,
              # 2010 dollars, adjusting for inflation
              "depression_cost": 27688*1.37,
              "ptsd_cost": 14857.,
              "lockdown_equivalent_policies": ['Lockdown'],
              "Currency": "USD"
              },
      "SG": {"exposed_health_workers": 58000,
              # counting all nurses + doctors https://www.healthhub.sg/a-z/health-statistics/12/health-manpower
              "gen_population_over14": 3456030,
              # Using mental disorder base line as https://pubmed.ncbi.nlm.nih.gov/30947763/, with pandemic rates as https://www.acpjournals.org/doi/full/10.7326/M20-1083
              # assume general population same as nonmedical workers, sick same as medical workers, 12 month prevalence set as baseline
              "depression_rate_baseline": 2.3/100.,
              "ptsd_rate_baseline": 0.0,
              "depression_rate_inc_sick": 8./100.,
              "depression_rate_inc_hworkers": 5.8/100.,
              "depression_rate_inc_gen_population": 8./100.,
              "ptsd_rate_inc_hworkers": 10.9/100.,
              # assume same as healthcare workers
              "ptsd_rate_inc_sick": 10.9/100.,
              # https://pubmed.ncbi.nlm.nih.gov/23977979/ 2008 data, adjusted for medical inflation at 10%[a]
              "depression_cost": 23971,
              # Assuming same cost as depression now
              "ptsd_cost": 23971.,
              "lockdown_equivalent_policies": ['Lockdown'],
              "Currency": "SGD"
              },
       "ES": {"exposed_health_workers": 542140,
              # https://data.worldbank.org/indicator/SP.POP.0014.TO.ZS?locations=ES
              "gen_population_over14": 39756493,
              "depression_rate_baseline": 4.73/100.,
              "ptsd_rate_baseline": 0.56/100.,
              "depression_rate_inc_sick": 0.14,
              "depression_rate_inc_hworkers": 0.14,
              "depression_rate_inc_gen_population": 0.14,
              "ptsd_rate_inc_hworkers": 0.152, # +0.099
              # assume same as healthcare workers
              "ptsd_rate_inc_sick": 0.152,
              # https://www.sciencedirect.com/science/article/pii/S0924977X21002182 adjusted for inflation 
              "depression_cost": 3412,
              # Assuming same cost as depression now
              "ptsd_cost": 1661*1.13,
              "lockdown_equivalent_policies": ['Lockdown'],
              "Currency": "euro"
       },
    
    "BR": {"exposed_health_workers": 2.1 * (2.164 + 10.119) * 100000,
           # counting all nurses + doctorshttps://data.worldbank.org/indicator/SH.MED.PHYS.ZS?locations=BR
           # https://data.worldbank.org/indicator/SH.MED.NUMW.P3?locations=BR
           
           # https://data.worldbank.org/indicator/SP.POP.0014.TO.ZS?locations=BR
           "gen_population_over14": 211 * 1e6 * 0.20709,
           # https://www.sciencedirect.com/science/article/pii/S0033350620305011?casa_token=d-izEpPH_vgAAAAA:91Tivv-Ibhhgzwv0wLcXE3YIr5zsIrbI3DFI8zDnEcHcioZosXYm6LHJdKroU873SL46_jT0Ig
           # https://www.jmir.org/2020/10/e22835
           # assume general population same as nonmedical workers, sick same as medical workers, 12 month prevalence set as baseline
           "depression_rate_baseline": 3.9/100.,
           "ptsd_rate_baseline": 5./100.,
           "depression_rate_inc_sick": 25.2/100,
           "depression_rate_inc_hworkers": 25.2/100,
           "depression_rate_inc_gen_population": 25.2/100,
           # https://www.sciencedirect.com/science/article/pii/S0022395620309870?casa_token=wcHIxnwxO9MAAAAA:ohWeXXqRjeLSfEdGp4ef2axF1E7AJnEld5RfA5nF2XOMjH6FTrasb9jsgn-5Xdj9PzMy8N4HAQ
           # https://www.scielo.br/j/rbp/a/qR3X56ZbwDHPFTpRk5jqs3M/?lang=en#:~:text=Not%20surprisingly%2C%20PTSD%20is%20highly,largest%20metropolitan%20areas%2C%20respectively).
           "ptsd_rate_inc_hworkers": 29.2/100.,
           # assume same as healthcare workers
           "ptsd_rate_inc_sick": 29.2/100.,
           # https://www.scielo.br/j/rbp/a/JQSTrFvqYwH7kZJyhFnrySD/?lang=en 2012 data, adjusted for  inflation at 7%[a]
           "depression_cost": 4100 * (1.07 ** 9),
           # Assuming same cost as depression now
           "ptsd_cost":  4100 * (1.07 ** 9),
           "lockdown_equivalent_policies": ['Lockdown'],
           "Currency": "reals"
           },
}
