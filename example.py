from pandemic_cost import get_pandemic_cost
from policy_functions.policy import Policy




# create a policy that uses the current policy for Germany, length 3 months startging in 2020-03-01, 
policy = Policy(region="GM", policy_type = "actual", start_date = "2020-03-01", policy_length = 3)

# select a country

cost_of_pandemic = get_pandemic_cost(policy)

print(cost_of_pandemic.d_costs)
