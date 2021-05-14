

# Default parameters - TNC & Trust Region
VSLY = {"DE": 158448,"US": 325000, "US-NY": 325000, "US-FL": 325000}
DEATHS_DIST = {"DE": {"0-10": 9/61951,
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
         119.21,131.32,144.44,158.65,173.98,190.49,208.43,227.24,247.55,269.21,1000]         
       
       }

# Hospitalization Costs Per Country
DAILY_HOSPITALIZATION_COST = {
       "DE": {"Inpatient": 162.65, "ICU bed": 795, "Ventilated ICU bed": 1539, "Currency": "euro"},
       # https://www.bcbs.com/coronavirus-updates/stories/infographic-covid-19-patients-high-risk-conditions-3x-more-likely-need-the-icu
       # corrected ventilated using  https://pubmed.ncbi.nlm.nih.gov/15942342/ with 3968/3184
       "US-NY": {"Inpatient": 33750/15, "ICU bed": 84375/15, "Ventilated ICU bed": 84375/15*3968/3184, "Currency": "USD"},
       "US-FL": {"Inpatient": 33750/15, "ICU bed": 84375/15, "Ventilated ICU bed": 84375/15*3968/3184, "Currency": "USD"}
}

# Mental Health Parameters per Country
MENTAL_HEALTH_COST = {
       "DE": {"exposed_health_workers": 300000,
              "gen_population_over14": (83.17-0.77-3.96-5.92) * 1e6,
              "depression_rate_sick": 30./100.0,
              "depression_rate_hworkers_normal": 12./100.,
              "depression_gen_pop": 12./100.,
              "ptsd_rate_hworkers": 7./100.,
              "ptsd_rate_sick": 7./100.,
              "depression_cost":4000.,
              "ptsd_cost": 40000.,
              "lockdown_months": 1,
              "Currency": "euro"
              },
       # https://www.chwsny.org/wp-content/uploads/2018/04/Full_CHWS_NY_Tracking_Report-2018b-1.pdf
       # only count hospitals and nursing home professionals
       "US-NY": {"exposed_health_workers": 600000,
          # https://www.health.ny.gov/statistics/vital_statistics/2018/table01.htm
              "gen_population_over14": 16164571,
                  # Assume same as healthcare workers for now
              "depression_rate_sick": 17/100.0,
         # https://www.psychiatryadvisor.com/home/topics/general-psychiatry/quantifying-the-rates-of-distress-among-health-care-workers-during-the-covid-19-pandemic/
              "depression_rate_hworkers_normal": 17./100.,
              "depression_gen_pop": 10.2/100.,
    # https://www.psychiatryadvisor.com/home/topics/general-psychiatry/quantifying-the-rates-of-distress-among-health-care-workers-during-the-covid-19-pandemic/
              "ptsd_rate_hworkers": 14./100.,
    # assume same as healthcare workers
              "ptsd_rate_sick": 14./100.,
              # 2010 dollars, adjusting for inflation
              "depression_cost": 27688*1.37,
              "ptsd_cost": 14857.,
              "lockdown_months": 1,
              "Currency": "USD"
              },
       "US-FL": {
          # https://bhw.hrsa.gov/sites/default/files/bureau-health-workforce/data-research/state-profiles/florida-2018.pdf
              "exposed_health_workers": 624451,
          # projected 2020 population http://edr.state.fl.us/content/population-demographics/data/pop_census_day.pdf
              "gen_population_over14": 17696804,
                  # Assume same as healthcare workers for now
              "depression_rate_sick": 17/100.0,
         # https://www.psychiatryadvisor.com/home/topics/general-psychiatry/quantifying-the-rates-of-distress-among-health-care-workers-during-the-covid-19-pandemic/
              "depression_rate_hworkers_normal": 17./100.,
              "depression_gen_pop": 10.2/100.,
    # https://www.psychiatryadvisor.com/home/topics/general-psychiatry/quantifying-the-rates-of-distress-among-health-care-workers-during-the-covid-19-pandemic/
              "ptsd_rate_hworkers": 14./100.,
    # assume same as healthcare workers
              "ptsd_rate_sick": 14./100.,
              # 2010 dollars, adjusting for inflation
              "depression_cost": 27688*1.37,
              "ptsd_cost": 14857.,
              "lockdown_months": 2,
              "Currency": "USD"},
      "SG": {"exposed_health_workers": 58000,
          # counting all nurses + doctors https://www.healthhub.sg/a-z/health-statistics/12/health-manpower
              "gen_population_over14": 3456030,
                  # Using mental disorder base line as https://pubmed.ncbi.nlm.nih.gov/30947763/, with pandemic rates as https://www.acpjournals.org/doi/full/10.7326/M20-1083
                  # assume general population same as nonmedical workers, sick same as medical workers, 12 month prevalence set as baseline
                "depression_rate_sick": 0.058,
              "depression_rate_hworkers_normal": 0.058,
              "depression_gen_pop": 0.08,
              "ptsd_rate_hworkers": 5.7/100.,
    # assume same as healthcare workers
              "ptsd_rate_sick": 5.7/100.,
              # https://pubmed.ncbi.nlm.nih.gov/23977979/ 2008 data, adjusted for medical inflation at 10%[a]
              "depression_cost": 23971,
              # Assuming same cost as depression now
              "ptsd_cost": 23971.,
              "lockdown_months": 2,
              "Currency": "SGD"
              }
}
      