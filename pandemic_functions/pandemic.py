import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from datetime import datetime, timedelta
from policy_functions.policy import Policy
from pandemic_functions.delphi_functions.DELPHI_model_policy_scenarios import run_delphi_policy_scenario, get_region_gammas, get_region_gammas_v2, read_policy_data_us_only, read_oxford_country_policy_data
from pandemic_functions.pandemic_params import region_symbol_country_dict, p_v
from analyze_gamma_rank import build_gamma_matrix, rank1_imputation, POLICY_NAMES


class Pandemic_Factory:
    """A wrapper class to help with the loading of different parameters and computing the Pandemic object"""
    def __init__(self):
        self.d_read_data_total_cases = {}
        self.d_region_policy_gammas = {}
        self.d_region_policy_gamma_samples = {}
        self._rank1_initialized = False
        path_to_predictions_combined = "pandemic_functions/pandemic_data/Global_DELPHI_predictions_combined.csv"
        if os.path.exists(path_to_predictions_combined):
            self.delphi_prediction = pd.read_csv(path_to_predictions_combined, keep_default_na=False)
        else:
            raise FileNotFoundError(f"Can not find file - "+ path_to_predictions_combined + " for actual polcy outcome")

    def _initialize_rank1(self, start_date="2020-03-15", end_date="2020-06-15",
                          n_bootstrap=20, seed=42):
        """
        Build the global gamma matrix, run rank-1 ALS for point estimates,
        and run bootstrap ALS for confidence interval samples.
        Populates d_region_policy_gammas and d_region_policy_gamma_samples
        for every region that appears in the matrix.
        """
        gamma_matrix, obs_mask, region_ids, policy_names = build_gamma_matrix(
            start_date=start_date, end_date=end_date,
        )
        completed, k_R = rank1_imputation(gamma_matrix, obs_mask)

        n_policies = gamma_matrix.shape[1]
        g = np.zeros(n_policies)
        for j in range(n_policies):
            unobs = np.where(~obs_mask[:, j])[0]
            r = unobs[np.argmax(k_R[unobs])]
            g[j] = completed[r, j] / k_R[r] if k_R[r] > 0 else 0.0
        fitted = np.outer(k_R, g)
        residuals = gamma_matrix[obs_mask] - fitted[obs_mask]
        sigma = np.std(residuals) if len(residuals) > 1 else 0.0

        rng = np.random.default_rng(seed)
        bootstrap_completed = []
        for _ in range(n_bootstrap):
            perturbed = gamma_matrix.copy()
            noise = rng.normal(0, sigma, size=obs_mask.sum())
            perturbed[obs_mask] += noise
            np.clip(perturbed, 0, None, out=perturbed)
            comp_b, _ = rank1_imputation(perturbed, obs_mask)
            bootstrap_completed.append(comp_b)

        code_to_key = {}
        for code, (country, province) in region_symbol_country_dict.items():
            key = f"{country}__{province}".replace(" ", "_")
            code_to_key[code] = key

        for code, key in code_to_key.items():
            if key not in region_ids:
                continue
            idx = region_ids.index(key)
            gammas = {p: completed[idx, j] for j, p in enumerate(policy_names)}
            self.d_region_policy_gammas[code] = gammas

            samples = []
            for comp_b in bootstrap_completed:
                sample_dict = {p: comp_b[idx, j] for j, p in enumerate(policy_names)}
                samples.append(sample_dict)
            self.d_region_policy_gamma_samples[code] = samples

        self._rank1_initialized = True

    def compute_delphi(self, policy, region, **kwargs):
        """
        Loads the required data and simulates the pandemic using DELPHI for the given policy.
        Returns an object of type `Pandemic`.
        """
        if not self._rank1_initialized:
            self._initialize_rank1()

        country, province = region_symbol_country_dict[region]
        country_sub = country.replace(' ', '_')
        province_sub = province.replace(' ', '_')

        if region in self.d_region_policy_gammas:
            dict_region_policy_gamma = self.d_region_policy_gammas[region]
        else:
            raise KeyError(f"Region '{region}' not found in rank-1 gamma estimates. "
                           f"Available: {list(self.d_region_policy_gammas.keys())}")

        if region not in self.d_read_data_total_cases:
            csv_path = f"pandemic_functions/pandemic_data/Cases_{country_sub}_{province_sub}.csv"
            if os.path.exists(csv_path):
                self.d_read_data_total_cases[region] = pd.read_csv(csv_path)
            else:
                raise FileNotFoundError(f"Can not find file - {csv_path}")
        totalcases = self.d_read_data_total_cases[region]

        gamma_samples = self.d_region_policy_gamma_samples.get(region, [])
        return Pandemic(policy, region, self.delphi_prediction, totalcases,
                        dict_region_policy_gamma, gamma_samples=gamma_samples, **kwargs)


class Pandemic:
    """
    Class to encapsulate the pandemic scenario. When initialized with the optional parameter `sample_gammas=True` 
    and if the policy is not actual, it will sample gammas from a distribution to compute lower and upper bounds on predictions.
    The optional parameter `n_sample:int` is used to specify the number of times gamma is sampled (default = 20).
    """
    def __init__(self, policy, region, delphi_prediction, totalcases, dict_region_policy_gamma,
                 gamma_samples=None, **kwargs):
        self.policy = policy
        self.region = region
        self.dict_region_policy_gamma = dict(sorted(dict_region_policy_gamma.items(), key=lambda x: x[0]))
        self._gamma_samples = gamma_samples or []
        country, province = region_symbol_country_dict[region]

        if country == 'US':
            policy_data = read_policy_data_us_only(state=province, start_date=self.policy.start_date, end_date=self.policy.end_date)
        else:
            policy_data = read_oxford_country_policy_data(country=country, start_date=self.policy.start_date, end_date=self.policy.end_date)
        n_measures = policy_data.iloc[:, 3:].shape[1]
        dict_region_policy_counts = {
            policy_data.columns[3 + i]: policy_data[
                policy_data.iloc[:, 3 + i] == 1
            ]
            .iloc[:, 3 + i]
            .sum()
            for i in range(n_measures)
        }
        self.dict_region_policy_counts = dict(sorted(dict_region_policy_counts.items(), key=lambda x: x[0]))
        output = self._get_deaths_and_hospitalizations(delphi_prediction, totalcases, dict_region_policy_gamma, **kwargs)
        self.num_cases, self.num_cases_lb, self.num_cases_ub, self.num_deaths, self.num_deaths_lb, \
            self.num_deaths_ub, self.hospitalization_days, self.hospitalization_days_lb, self.hospitalization_days_ub, \
            self.icu_days, self.icu_days_lb, self.icu_days_ub, self.ventilated_days, self.ventilated_days_lb, \
            self.ventilated_days_ub = output   
        
        
    def _get_deaths_and_hospitalizations(self, delphi_prediction, totalcases, dict_region_policy_gamma, 
                                        sample_gammas:bool=False, n_sample:int=20):
        # this function gets the number of deaths and hospitalizations that would occur under such policy, using DELPHI
        # the return value is a tuple of numbers
        country, province = region_symbol_country_dict[self.region]
        if self.policy.policy_type == "actual":
            totalcases.date = pd.to_datetime(totalcases.date)
            start_date = pd.to_datetime(self.policy.start_date)
            end_date = start_date + pd.DateOffset(months=self.policy.num_months)
            cases_in_interval = totalcases.query("date >= @start_date and date <= @end_date")
            delphi_prediction.Day = pd.to_datetime(delphi_prediction.Day)
            preds_in_interval = delphi_prediction.query("Day >= @start_date and Day <= @end_date and Country == @country and Province == @province")

            num_deaths = cases_in_interval.iloc[-1]["death_cnt"] - cases_in_interval.iloc[0]["death_cnt"]
            num_cases = cases_in_interval.iloc[-1]["case_cnt"] - cases_in_interval.iloc[0]["case_cnt"]
            hospitalization_days = preds_in_interval["Active Hospitalized"].sum()
            ventilated_days = preds_in_interval["Active Ventilated"].sum()
            num_cases_lb = num_cases_ub = num_deaths_lb = num_deaths_ub = np.nan
            hospitalization_days_lb = hospitalization_days_ub = icu_days_lb = icu_days_ub = ventilated_days_lb = ventilated_days_ub = np.nan

            if self.region == "DE":
                hosp_global = pd.read_csv("pandemic_functions/pandemic_data/global_hospitalizations.csv")
                hosp_global.date = pd.to_datetime(hosp_global.date)
                hosp_germany = hosp_global[hosp_global.country_id == "DE"].copy()
                hosp_germany.date = pd.to_datetime(hosp_germany.date)
                hosp_germany = hosp_germany.query("date >= @start_date and date <= @end_date")
                icu_days = np.nansum(hosp_germany.icu_beds_used)
                icu_days = icu_days - ventilated_days
                hospitalization_days = hospitalization_days - icu_days
            else:
                icu_days = ventilated_days*(0.15/0.85) # (1/0.85 - 1)*ventilated_days
                hospitalization_days = hospitalization_days - icu_days
        else:
            num_cases, num_cases_lb, num_cases_ub, num_deaths, num_deaths_lb, num_deaths_ub, \
                    hospitalization_days, ventilated_days = run_delphi_policy_scenario(self.policy, self.region, totalcases, dict_region_policy_gamma)
            hospitalization_days_lb, ventilated_days_lb = hospitalization_days, ventilated_days
            hospitalization_days_ub, ventilated_days_ub = hospitalization_days, ventilated_days
            if sample_gammas and self._gamma_samples:
                for dict_gammas in self._gamma_samples:
                    _, nclb, ncub, _, ndlb, ndub, nhd, nvd = run_delphi_policy_scenario(self.policy, self.region, totalcases, dict_gammas)
                    num_cases_lb = min(num_cases_lb, nclb)
                    num_cases_ub = max(num_cases_ub, ncub)
                    num_deaths_lb = min(num_deaths_lb, ndlb)
                    num_deaths_ub = max(num_deaths_ub, ndub)
                    hospitalization_days_lb = min(hospitalization_days_lb, nhd)
                    hospitalization_days_ub = max(hospitalization_days_ub, nhd)
                    ventilated_days_lb = min(ventilated_days_lb, nvd)
                    ventilated_days_ub = max(ventilated_days_ub, nvd)
            # ventilated_days = hospitalization_days*p_v 
            icu_vec = np.array([ventilated_days, ventilated_days_lb, ventilated_days_ub])*(0.15/0.85)
            icu_days, icu_days_lb, icu_days_ub = tuple(icu_vec)
            hospitalization_days, hospitalization_days_lb, hospitalization_days_ub = \
                tuple(np.array([hospitalization_days, hospitalization_days_lb, hospitalization_days_ub]) - icu_vec)
        
        return num_cases, num_cases_lb, num_cases_ub, num_deaths, num_deaths_lb, num_deaths_ub, \
            hospitalization_days, hospitalization_days_lb, hospitalization_days_ub, \
            icu_days, icu_days_lb, icu_days_ub, ventilated_days, ventilated_days_lb, ventilated_days_ub
        
        


