

# Default parameters - TNC & Trust Region
VSLY = {"GM": 158448,"US": 325000}
DEATHS_DIST = {"GM": {"0-10": 9/61951,
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
                              } }

DEATHS_ACTUARIAL_TABLE = {"GM": [
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.5785, 1.5951, 1.6006, 1.595, 1.5785, 1.5503, 1.5094, 1.4643, 1.4238,
1.388, 1.3574, 1.3325, 1.3137, 1.3018, 1.2968, 1.2995, 1.3104, 1.3299, 1.3586, 1.397, 1.4454, 1.5045, 1.5754, 1.6591,
1.7566, 1.8694, 1.9983, 2.1445, 2.3096, 2.497, 2.7107, 2.9545, 3.2325, 3.5482, 3.9057, 4.3087, 4.7606, 5.2655, 5.8269,
6.4474, 7.1294, 7.8756, 8.6884, 9.5704, 10.5241, 11.5521, 12.6571, 13.8417, 15.1083, 16.4598, 18.0706, 20.0313, 22.3416,
25.0018, 28.0117, 31.3714, 35.0808, 39.14, 43.549, 48.3078, 53.4163, 58.8745, 64.6826, 70.8404, 77.348, 84.2053,
91.4124, 98.9693, 106.876, 115.1324, 123.7386, 132.6945, 142.0002, 151.6557, 161.6609, 172.016, 182.7207, 193.7753,
205.1796, 216.9337, 229.0375, 241.4911, 254.2945, 267.4477, 280.9506, 294.8032, 309.0057, 323.5579, 338.4599, 353.7116,
369.3131, 385.2644, 401.5655, 418.2163, 435.2169, 452.5672, 470.2673, 488.3172, 506.7169, 525.4663, 544.5654, 564.0144,
583.8131, 603.9616, 624.4598, 1000], 
"US": [0, 5.777,
 0.38200000000000006,
 0.24800000000000003,
 0.193,
 0.149,
 0.14100000000000001,
 0.126,
 0.114,
 0.104,
 0.095,
 0.093,
 0.10300000000000001,
 0.133,
 0.186,
 0.25799999999999995,
 0.33799999999999997,
 0.421,
 0.51,
 0.603,
 0.698,
 0.795,
 0.889,
 0.97,
 1.0319999999999998,
 1.08,
 1.1230000000000002,
 1.165,
 1.207,
 1.252,
 1.3,
 1.351,
 1.402,
 1.454,
 1.5059999999999998,
 1.556,
 1.6149999999999998,
 1.6789999999999998,
 1.7399999999999998,
 1.7979999999999998,
 1.8599999999999999,
 1.936,
 2.036,
 2.16,
 2.306,
 2.4699999999999998,
 2.647,
 2.846,
 3.079,
 3.357,
 3.6819999999999995,
 4.029999999999999,
 4.401000000000001,
 4.82,
 5.285,
 5.778,
 6.284,
 6.7940000000000005,
 7.319,
 7.869,
 8.456,
 9.093,
 9.768,
 10.466999999999999,
 11.181000000000001,
 11.921999999999999,
 12.71,
 13.620999999999999,
 14.62,
 15.77,
 17.1,
 18.428,
 20.317000000000004,
 22.102,
 24.194,
 26.342,
 29.041999999999998,
 32.001,
 35.443000000000005,
 39.257,
 43.393,
 48.163,
 53.216,
 59.24,
 66.564,
 74.045,
 81.954,
 90.87899999999999,
 101.93799999999999,
 114.07499999999999,
 127.331,
 141.733,
 157.28900000000002,
 173.98600000000002,
 191.78799999999998,
 210.63299999999998,
 230.432,
 251.066,
 272.395,
 294.253,
 316.456,
 1000.0]}

# Hospitalization Costs Per Country
DAILY_HOSPITALIZATION_COST = {
       "DE": {"Inpatient": None, "ICU bed": 795, "Ventilated ICU bed": 1539, "Currency": "euro"}
}