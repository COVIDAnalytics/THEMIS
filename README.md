# THEMIS: Cost Model for Analyzing Non-Pharmaceutical Interventions

The repository contains code for a framework that attempts to analyze the different sources of costs incurred by the various non-pharmaceutical interventions that have been implemented in the COVID-19 pandemic. 

## 1. Repository Structure

Here we briefly describe how the repository is structured for better navigation of the reader. \\
\\
`THEMIS/` \\
1. `cost_functions/` - contains functions and data defining different cost components
    * `economic_cost/` - contains functions and data for different components of economic cost
        - `economic_data/` - contains the data for economic costs
            - `gdp/` - contains the GDP related data
            - `unemployment/` - contains the unemployment related data
            - `economic_params` - contains the parameters for economic costs
        - `long_term_economic_costs.py` (currently empty)
        - `short_term_economic_costs.py`
        - `short_term_gdp_costs.py`
        - `short_term_unemployment_costs.py`
    * `health_cost/` - contains functions and data for different components of health cost
        - `health_data/` - contains the data for health costs
            - `health_params` - contains the parameters for health costs
        - `death_costs.py`
        - `hospitalization_costs.py`
        - `mental_health_costs.py`
2. `pandemic_functions/` - contains functions and data for simulating the pandemic
    * `delphi_functions/` - contains functions for running the DELPHI model
        - `DELPHI_model_policy_scenarios.py` - the DELPHI model
    * `pandemic_data/` - contains historical data for the cases and deaths as well as DELPHI parameters
    * `pandemic.py` - encapsulates the pandemic scenario for a region
    * `pandemic_params.py`
3. `policy_functions/` - functions to define different policies
    * `policy.py`
4. `utils/` - miscellaneous tools required in the project
    * `cost_utils.py`
    * `visualization_utils.py`
5. `notebooks/` - notebooks and excel sheets that demonstrate the calculation of some of the parameters and plots
    * `monthly_dominant_policy.ipynb` - gamma values and GDP and Employment impact for different policies in different regions
    * `visualize_results_final.ipynb` - code to produce the plots
    * ...
6. `simulation_results/` - contains the results of the simulations based on which the plots are created
7. `pandemic_cost.py` - encapsulates all costs associated with the pandemic
8. `main.py` - the main script that can be run to compute the results
9. `LICENSE` - license for this project
10. `README.md` - brief documentation

## 2. Running the Model

The code was written with **Python 3.8.3**. The model is executed through the main file which takes three arguments
1. `--region` or `-r` : Region code for the region the simulation will be run. Should be one of "DE", "US-NY", "US-FL", "ES", "BR", "SG"
2. `--startdate` or `-sd` : (optional, default = `2020-03-01`) start date for the simulation as a string in YYYY-MM-DD format
3. `--length` or `-l` : (optional, default = 3) number of months the simulation will be run for starting from the start date
\\
Example:
```
> python main.py -r "DE" -sd "2020-03-15" -l 3
```

## 3. Authors

Michael L. Li, Saksham Soni & Baptiste Rossi
        