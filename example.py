from pandemic_cost import get_pandemic_cost
from policy_functions.policy import Policy
from pandemic_functions.pandemic import Pandemic



# create a policy that uses the current policy for Germany, length 3 months starting in 2020-03-01, 
policy = Policy(policy_type = "actual", start_date = "2020-03-01", policy_length = 3)
# simulate the pandemic using such policy
pandemic = Pandemic(policy,region="GM")
cost_of_pandemic = get_pandemic_cost(pandemic)
# Should have 5 items in the dict, each representing costs. Economic costs and death costs should be on the order of billions to tens of billions at least. 
print(cost_of_pandemic.__dict__)

# create a policy that a hypothetical policy for Germany, length 3 months starting in 2020-03-01, 
policy2 = Policy(policy_type = "hypothetical", start_date = "2020-03-01", policy_vector = ["No_Measure","No_Measure","No_Measure"])
# simulate the pandemic using such policy
pandemic2 = Pandemic(policy2,region="GM")
cost_of_pandemic2 = get_pandemic_cost(pandemic2)
# Should have 5 items in the dict, each representing costs. Short term economic costs should be 0. Health costs should be higher than above. 
print(cost_of_pandemic2.__dict__)

# create a policy that uses a hypothetical policy for Germany, length 3 months starting in 2020-03-01, 
policy3 = Policy(policy_type = "hypothetical", start_date = "2020-03-01", policy_vector = ["Lockdown","Lockdown","Lockdown"])
# simulate the pandemic using such policy
pandemic3 = Pandemic(policy3,region="GM")
cost_of_pandemic3 = get_pandemic_cost(pandemic3)
# Should have 5 items in the dict, each representing costs. Short term economic costs should be 0. Health costs should be higher than above. 
print(cost_of_pandemic3.__dict__)