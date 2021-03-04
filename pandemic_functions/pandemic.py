
class Pandemic:
    
    def __init__(self, policy, region):
    # This is the simulation of the pandemic under a certain policy
    # Given a fixed policy, we can calculate the number of deaths and hospitalizations incurred in such period using DELPHI. 
    # We call it here so that we dont have to repeatedly call DELPHI over and over again. 
        self.policy = policy
        self.region = region
        self.num_deaths, self.hospitalization_days, self.icu_days, self.ventilated_days = self._get_deaths_and_hospitalizations()        
        
        
    def _get_deaths_and_hospitalizations(self):
        # this function gets the number of deaths and hospitalizations that would occur under such policy, using DELPHI
        # the return value is a tuple of numbers
        # TODO: This is current a dummy function
        return 60000, 1000000, 200000, 100000
        
        


