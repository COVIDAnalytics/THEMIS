
from dateparser import parse
from dateutil.relativedelta import relativedelta

class Policy:
    
    def __init__(self, region, policy_type, start_date, policy_length=None, policy_vec = None):
    # This is the fundamental policy that determines the cost of the pandemic. Here we consider a policy in whole number of months.
    # the region is the country/state the policy is for
    # policy_type can be "hypothetical" or "actual"
    # if actual, we need start_date and policy_length (in months) to know the period you are modeling the policy for
    # if hypothetical, we need start_date and a policy vector (policy_vec) that implements the hypothetical policy per month. 
    # E.g. ["Lockdown", "None"] represents a 2-month hypothetical policy in which first month we lockdown and second month we do nothing     
        
        self.region = region
        self.policy_type = policy_type
        self.start_date = parse(start_date)
        if policy_type == "actual":
            if policy_length is None:
                raise ValueError("Policy length needs to be specified for actual policy")
            self.end_date = self.start_date + relativedelta(months = policy_length)
            self.num_months = policy_length
            # we do not have a policy vector under current policy
            self.policy_vec = None
        elif policy_type == "hypothetical":
            if policy_length is not None:
                raise ValueError("Policy length should not be specified for hypothetical policy")
            if policy_vec is None:
                raise ValueError("Policy vector needs to be specified for hypothetical policy")
            self.end_date = self.start_date + relativedelta(months = len(policy_vec))
            self.num_months = len(policy_vec)
            self.policy_vec = policy_vec
        # Given a fixed policy, we can calculate the number of deaths and hospitalizations incurred in such period using DELPHI. 
        # We call it here so that we dont have to repeatedly call DELPHI over and over again. 
        self.num_deaths, self.hospitalization_days, self.icu_days, self.ventilated_days = self._get_deaths_and_hospitalizations()        
        
        
    def _get_deaths_and_hospitalizations(self):
        # this function gets the number of deaths and hospitalizations that would occur under such policy, using DELPHI
        # the return value is a tuple of numbers
        # TODO: This is current a dummy function
        return 60000, 1000000, 200000, 100000
        
        


