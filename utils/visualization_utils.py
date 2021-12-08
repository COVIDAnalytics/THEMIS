import plotly.express as px
import matplotlib.pyplot as plt
import pandas as pd

def shorten_policy_string(pname):
    policies = pname.split("-")
    DICT_POLICY_CODE = {
        'No_Measure': "1",
        'Restrict_Mass_Gatherings': "2",
        'Authorize_Schools_but_Restrict_Mass_Gatherings_and_Others': "3",
        # 'Mass_Gatherings_Authorized_But_Others_Restricted': "3",
        'Restrict_Mass_Gatherings_and_Schools': "4",
        'Restrict_Mass_Gatherings_and_Schools_and_Others': "5",
        'Lockdown': "6"
    }
    short_name = '-'.join([DICT_POLICY_CODE[pol] for pol in policies])
    return short_name

def region_policy_scatter_plot(results:pd.DataFrame, region_name:str, start_date:str = "3/15/2020", currency_symbol:str = "\N{EURO SIGN}", y_val:str = 'num_deaths'):
    y_val_name = {'num_deaths': 'Number of Deaths', 'life_costs': 'Humanitarian Costs'}
    fig = px.scatter(results.query("start_date == @start_date"), 
                    x='st_economic_costs', y=y_val, color='is_actual', 
                    log_x=False, log_y=True, hover_name="short_policy_name", hover_data=["num_deaths", "num_cases", "mh_costs"],
                    title=f"Policy Simulations for {region_name} starting {start_date}",
                    labels={
                        'st_economic_costs': 'Economic Costs', y_val: y_val_name[y_val], "is_actual": "Scenario Type",
                    },
                    template="simple_white")
    fig.update_yaxes(
        showgrid=True
    )
    fig.update_xaxes(
        tickprefix=currency_symbol, showgrid=True
    )
    if y_val == 'life_costs':
        fig.update_yaxes(
            tickprefix=currency_symbol, showgrid=True
        )
    return fig

def best_policy_cost_breakdown_plot(results:pd.DataFrame, region_name:str, n:int = 20, start_date:str = "3/15/2020", currency_symbol:str = "\N{EURO SIGN}" ):
    fig = px.bar(results.query("start_date == @start_date").\
                        sort_values(by='total_cost', ascending=True).\
                        iloc[:n].\
                        melt(id_vars=['short_policy_name'], value_vars=['st_economic_costs', 'd_costs', 'h_costs', 'mh_costs'],
                            var_name='cost_type', value_name='cost'
                        ), 
                    x='short_policy_name', y='cost', color='cost_type', 
                    log_x=False, log_y=False, title=f"Minimum Total Cost Policies for {region_name} starting {start_date}",
                    labels={'cost_type':'Cost Type', 'short_policy_name':'Policy'},
                    template="simple_white")

    fig.update_yaxes(
        tickprefix=currency_symbol, showgrid=True
    )
    newnames = {'st_economic_costs': 'Economic Costs', 'd_costs': 'Loss of Life Costs', 'h_costs': 'Hospitalization Costs',
                'mh_costs': 'Mental Health Costs'}
    fig.for_each_trace(lambda t: t.update(name = newnames[t.name],
                                        legendgroup = newnames[t.name],
                                        hovertemplate = t.hovertemplate.replace(t.name, newnames[t.name]))
                    )
    return fig