import plotly.express as px
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math
from cost_functions.economic_cost.economic_data.economic_params import TOTAL_GDP

def shorten_policy_string(pname):
    policies = pname.split("-")
    DICT_POLICY_CODE = {
        'No_Measure': "1",
        'Restrict_Mass_Gatherings': "2",
        'Mass_Gatherings_Authorized_But_Others_Restricted': "3",
        'Restrict_Mass_Gatherings_and_Schools': "4",
        'Authorize_Schools_but_Restrict_Mass_Gatherings_and_Others': "5",
        'Restrict_Mass_Gatherings_and_Schools_and_Others': "6",
        'Lockdown': "7"
    }
    short_name = '-'.join([DICT_POLICY_CODE[pol] for pol in policies])
    return short_name

# Currency symbol mapping
currency_symbols = {
    'US': '$', 'ES': '€', 'GB': '£', 'CN': '¥', 'JP': '¥', 'IN': '₹', 'DE': '€', 'FR': '€',
    # Add more country codes and symbols as needed
}
name_codes = {
    'US-NY': 'New York, United States', 'ES': 'Spain', 'DE': 'Germany', 'BR': 'Brazil'
    # Add more country codes and symbols as needed
}

cost_colors = {
    'st_economic_costs': '#636EFA',  # blue
    'd_costs': '#EF553B',  # red
    'h_costs': '#00CC96',  # green
    'mh_costs': '#AB63FA'  # purple
}

def get_currency_symbol(country_code):
    # Extract the ISO country code (first two characters)
    iso_code = country_code.split('-')[0]
    return currency_symbols.get(iso_code, '$')  # Default to '$' if not found


def region_policy_scatter_plot_panel(results: pd.DataFrame, start_date: str = "3/15/2020", 
                                     y_val: str = 'num_deaths', width: int = 600, height: int = 600):
    regions = results['country'].unique()
    num_regions = len(regions)
    num_rows = math.ceil(num_regions / 2)

    most_severe = '7-7-7'

    fig = make_subplots(rows=num_rows, cols=2, shared_xaxes=False, shared_yaxes=False, 
                        subplot_titles=[name_codes[x] for x in regions], vertical_spacing=0.1, horizontal_spacing=0.1)
    
    for i, region in enumerate(regions):
        df = results.query("start_date == @start_date and country == @region").copy()
        gdp = TOTAL_GDP[region]

        if y_val == 'life_costs':
            df['life_costs'] = df.d_costs + df.h_costs + df.mh_costs
            df['life_costs_lb'] = df.d_costs_lb + df.h_costs + df.mh_costs_lb
            df['life_costs_ub'] = df.d_costs_ub + df.h_costs + df.mh_costs_ub

        cost_cols = ['st_economic_costs', 'st_economic_costs_lb', 'st_economic_costs_ub',
                     y_val, f'{y_val}_lb', f'{y_val}_ub']
        for col in cost_cols:
            if col in df.columns:
                df[col] = df[col] / gdp * 100

        df['st_economic_costs_lerr'] = df['st_economic_costs'] - df['st_economic_costs_lb']
        df['st_economic_costs_uerr'] = df['st_economic_costs_ub'] - df['st_economic_costs']
        df[f'{y_val}_lerr'] = df[y_val] - df[f'{y_val}_lb']
        df[f'{y_val}_uerr'] = df[f'{y_val}_ub'] - df[y_val]

        df['avg_strength'] = df['short_policy_name'].apply(lambda x: sum(map(int, x.split('-'))) / 3 if x != 'actual' else None)

        y_val_name = 'Number of Deaths' if y_val == 'num_deaths' else \
            'Humanitarian Costs (% of GDP)' if y_val == 'life_costs' else y_val
        
        scatter = px.scatter(df, x='st_economic_costs', y=y_val, color='avg_strength',
                             error_x='st_economic_costs_uerr', error_x_minus='st_economic_costs_lerr',
                             error_y=f'{y_val}_uerr', error_y_minus=f'{y_val}_lerr', log_x=False, log_y=True, 
                             hover_name="short_policy_name", hover_data=["avg_strength"],
                             labels={'st_economic_costs': 'Economic Costs (% of GDP)', y_val: y_val_name, "avg_strength": "Average Strength"},
                             template="plotly_white", color_continuous_scale='Reds')

        for trace in scatter.data:
            trace.error_x.update(color='rgba(150,150,150,0.3)', thickness=1)
            trace.error_y.update(color='rgba(150,150,150,0.3)', thickness=1)
            fig.add_trace(trace, row=(i//2) + 1, col=(i % 2) + 1)
        
        highlight_policy = df[df['short_policy_name'] == most_severe]
        if not highlight_policy.empty:
            fig.add_trace(go.Scatter(
                x=highlight_policy['st_economic_costs'],
                y=highlight_policy[y_val],
                mode='markers+text',
                marker=dict(color='red', size=12, symbol='diamond'),
                showlegend=False,
                text=[most_severe],
                textposition="top center",
                hoverinfo='skip'
            ), row=(i//2) + 1, col=(i % 2) + 1)

        actual_policy = df[df['short_policy_name'] == 'actual']
        if not actual_policy.empty:
            fig.add_trace(go.Scatter(
                x=actual_policy['st_economic_costs'],
                y=actual_policy[y_val],
                mode='markers+text',
                marker=dict(color='blue', size=12, symbol='star'),
                showlegend=False,
                text=["Actual"],
                textposition="top center",
                hoverinfo='skip'
            ), row=(i//2) + 1, col=(i % 2) + 1)

        fig.update_xaxes(title_text='Economic Costs (% of GDP)', ticksuffix='%', row=(i//2) + 1, col=(i % 2) + 1)
        fig.update_yaxes(title_text=y_val_name, type='log', row=(i//2) + 1, col=(i % 2) + 1)
        if y_val == 'life_costs':
            fig.update_yaxes(ticksuffix='%', type='log', row=(i//2) + 1, col=(i % 2) + 1)

    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode='markers',
        marker=dict(color='red', size=12, symbol='diamond'),
        name="Most Severe Restrictions"
    ))
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode='markers',
        marker=dict(color='blue', size=12, symbol='star'),
        name="Actual Policy"
    ))

    fig.update_layout(
        legend=dict(title='Reference Policies', orientation='h', y=1.05, yanchor='bottom', x=0.5, xanchor='center', traceorder='normal'),
        coloraxis_colorbar=dict(title='Average<br>Strength'),
        margin=dict(l=40, r=40, t=80, b=40),
        plot_bgcolor='rgba(0,0,0,0)',
        width=width * 2,
        height=height * num_rows
    )

    return fig


def region_policy_scatter_plot(results: pd.DataFrame, region_name: str, start_date: str = "3/15/2020", 
                               currency_symbol: str = "\N{EURO SIGN}", y_val: str = 'num_deaths'):
    df = results.query("start_date == @start_date")
    if y_val == 'life_costs':
        df['life_costs'] = df.d_costs + df.h_costs + df.mh_costs
        df['life_costs_lb'] = df.d_costs_lb + df.h_costs + df.mh_costs_lb
        df['life_costs_ub'] = df.d_costs_ub + df.h_costs + df.mh_costs_ub
    df['st_economic_costs_lerr'] = df['st_economic_costs'] - df['st_economic_costs_lb']
    df['st_economic_costs_uerr'] = df['st_economic_costs_ub'] - df['st_economic_costs']
    df[f'{y_val}_lerr'] = df[y_val] - df[f'{y_val}_lb']
    df[f'{y_val}_uerr'] = df[f'{y_val}_ub'] - df[y_val]

    # Calculate average strength and add it as a column
    df['avg_strength'] = df['short_policy_name'].apply(lambda x: sum(map(int, x.split('-'))) / 3 if x != 'actual' else None)

    y_val_name = 'Number of Deaths' if y_val == 'num_deaths' else \
        'Humanitarian Costs' if y_val == 'life_costs' else y_val
    fig = px.scatter(df, x='st_economic_costs', y=y_val, color='avg_strength',
                    error_x='st_economic_costs_uerr', error_x_minus='st_economic_costs_lerr',
                    error_y=f'{y_val}_uerr', error_y_minus=f'{y_val}_lerr', log_x=False, log_y=True, 
                    hover_name="short_policy_name", hover_data=["num_deaths", "num_cases", "mh_costs", "h_costs", "avg_strength"],
                    title=f"Policy Simulations for {region_name} starting {start_date}",
                    labels={
                        'st_economic_costs': 'Economic Costs', y_val: y_val_name, "avg_strength": "Average Strength",
                    },
                    template="plotly_white",
                    color_continuous_scale=px.colors.sequential.Plasma)

    # Customize the markers
    fig.update_traces(marker=dict(size=10, line=dict(width=0)),
                      selector=dict(mode='markers'))

    fig.update_yaxes(
        showgrid=True, title_text=y_val_name
    )
    fig.update_xaxes(
        tickprefix=currency_symbol, showgrid=True, title_text='Economic Costs'
    )
    if y_val == 'life_costs':
        fig.update_yaxes(
            tickprefix=currency_symbol, showgrid=True
        )
    
    # Highlight the policy "6-6-6"
    highlight_policy = df[df['short_policy_name'] == '6-6-6']
    if not highlight_policy.empty:
        fig.add_trace(go.Scatter(
            x=highlight_policy['st_economic_costs'],
            y=highlight_policy[y_val],
            mode='markers+text',
            marker=dict(color='red', size=12, symbol='diamond'),
            name="Most Severe Restriction",
            text=["6-6-6"],
            textposition="top center",
            hoverinfo='skip'
        ))

    # Highlight the "actual" policy
    actual_policy = df[df['short_policy_name'] == 'actual']
    if not actual_policy.empty:
        fig.add_trace(go.Scatter(
            x=actual_policy['st_economic_costs'],
            y=actual_policy[y_val],
            mode='markers+text',
            marker=dict(color='green', size=12, symbol='star'),
            name="Actual Policy",
            text=["Actual"],
            textposition="top center",
            hoverinfo='skip'
        ))

    # Improve layout aesthetics
    fig.update_layout(
        title=dict(text=f"Policy Simulations for {region_name} starting {start_date}", x=0.5),
        title_font=dict(size=20, family='Arial, bold'),
        legend=dict(title='', orientation='h', y=1.02, yanchor='bottom', x=0.5, xanchor='center', traceorder='reversed'),
        coloraxis_colorbar=dict(title='Average Strength'),
        margin=dict(l=40, r=40, t=80, b=40),
        plot_bgcolor='rgba(0,0,0,0)'
    )

    return fig

def best_policy_cost_breakdown_plot(results: pd.DataFrame, region_name: str, n: int = 20, start_date: str = "3/15/2020", currency_symbol: str = "\N{EURO SIGN}"):
    filtered_results = results.query("start_date == @start_date").sort_values(by='total_cost', ascending=True).iloc[:n]
    melted_results = filtered_results.melt(id_vars=['short_policy_name'], value_vars=['st_economic_costs', 'd_costs', 'h_costs', 'mh_costs'],
                                           var_name='cost_type', value_name='cost')

    fig = px.bar(melted_results, 
                 x='short_policy_name', y='cost', color='cost_type',
                 log_x=False, log_y=False, title=f"Minimum Total Cost Policies for {region_name} starting {start_date}",
                 labels={'cost_type':'Cost Type', 'short_policy_name':'Policy'},
                 template="simple_white")

    fig.update_yaxes(tickprefix=currency_symbol, showgrid=True)
    
    newnames = {'st_economic_costs': 'Economic Costs', 'd_costs': 'Loss of Life Costs', 'h_costs': 'Hospitalization Costs',
                'mh_costs': 'Mental Health Costs'}
    fig.for_each_trace(lambda t: t.update(name=newnames[t.name],
                                          legendgroup=newnames[t.name],
                                          hovertemplate=t.hovertemplate.replace(t.name, newnames[t.name]))
                      )

    # Highlight the policy "6-6-6"
    highlight_policy = filtered_results[filtered_results['short_policy_name'] == '6-6-6']
    if not highlight_policy.empty:
        for cost_type in ['st_economic_costs', 'd_costs', 'h_costs', 'mh_costs']:
            fig.add_trace(go.Scatter(
                x=[highlight_policy['short_policy_name'].values[0]],
                y=[highlight_policy[cost_type].values[0]],
                mode='markers',
                marker=dict(color='red', size=12, symbol='circle'),
                name=f"Highlight {newnames[cost_type]}",
                hoverinfo='skip'
            ))

    return fig


def best_policy_cost_breakdown_plot(results: pd.DataFrame, n: int = 20, start_date: str = "3/15/2020", currency_symbol: str = "\N{EURO SIGN}"):
    filtered_results = results.query("start_date == @start_date").sort_values(by='total_cost', ascending=True).iloc[:n]
    melted_results = filtered_results.melt(id_vars=['short_policy_name'], value_vars=['st_economic_costs', 'd_costs', 'h_costs', 'mh_costs'],
                                           var_name='cost_type', value_name='cost')

    fig = px.bar(melted_results, 
                 x='short_policy_name', y='cost', color='cost_type',
                 log_x=False, log_y=False, title=f"Minimum Total Cost Policies for {region_name} starting {start_date}",
                 labels={'cost_type':'Cost Type', 'short_policy_name':'Policy'},
                 template="simple_white")

    fig.update_yaxes(tickprefix=currency_symbol, showgrid=True)
    
    newnames = {'st_economic_costs': 'Economic Costs', 'd_costs': 'Loss of Life Costs', 'h_costs': 'Hospitalization Costs',
                'mh_costs': 'Mental Health Costs'}
    fig.for_each_trace(lambda t: t.update(name=newnames[t.name],
                                          legendgroup=newnames[t.name],
                                          hovertemplate=t.hovertemplate.replace(t.name, newnames[t.name]))
                      )

    # Highlight the policy "6-6-6"
    highlight_policy = filtered_results[filtered_results['short_policy_name'] == '6-6-6']
    if not highlight_policy.empty:
        for cost_type in ['st_economic_costs', 'd_costs', 'h_costs', 'mh_costs']:
            fig.add_trace(go.Scatter(
                x=[highlight_policy['short_policy_name'].values[0]],
                y=[highlight_policy[cost_type].values[0]],
                mode='markers',
                marker=dict(color='red', size=12, symbol='circle'),
                name=f"Highlight {newnames[cost_type]}",
                hoverinfo='skip'
            ))

    return fig



def best_policy_cost_breakdown_plot_panel(results: pd.DataFrame, n: int = 20, start_date: str = "3/15/2020", width: int = 600, height: int = 600):
    regions = results['country'].unique()
    num_regions = len(regions)
    num_rows = math.ceil(num_regions / 2)
    
    fig = make_subplots(rows=num_rows, cols=2, shared_xaxes=False, shared_yaxes=False, 
                        subplot_titles=[name_codes[x] for x in regions], vertical_spacing=0.1, horizontal_spacing=0.1)
    
    newnames = {'st_economic_costs': 'Economic Costs', 'd_costs': 'Loss of Life Costs', 'h_costs': 'Hospitalization Costs',
                'mh_costs': 'Mental Health Costs'}
    
    for i, region in enumerate(regions):
        gdp = TOTAL_GDP[region]
        filtered_results = results.query("start_date == @start_date and country == @region").copy()
        for col in ['st_economic_costs', 'd_costs', 'h_costs', 'mh_costs', 'total_cost']:
            if col in filtered_results.columns:
                filtered_results[col] = filtered_results[col] / gdp * 100
        filtered_results = filtered_results.sort_values(by='total_cost', ascending=True).iloc[:n]

        melted_results = filtered_results.melt(id_vars=['short_policy_name'], value_vars=['st_economic_costs', 'd_costs', 'h_costs', 'mh_costs'],
                                               var_name='cost_type', value_name='cost')

        for cost_type in ['st_economic_costs', 'd_costs', 'h_costs', 'mh_costs']:
            fig.add_trace(go.Bar(
                x=melted_results[melted_results['cost_type'] == cost_type]['short_policy_name'],
                y=melted_results[melted_results['cost_type'] == cost_type]['cost'],
                name=newnames[cost_type],
                marker_color=cost_colors[cost_type],
                legendgroup=newnames[cost_type],
                showlegend=(i == 0),
                hovertemplate=newnames[cost_type] + ': %{y:.1f}%<extra></extra>'
            ), row=(i//2) + 1, col=(i % 2) + 1)

        fig.update_yaxes(title_text='Cost (% of GDP)', ticksuffix='%', row=(i//2) + 1, col=(i % 2) + 1)
        fig.update_xaxes(title_text='Policy', row=(i//2) + 1, col=(i % 2) + 1)

    fig.update_layout(
        barmode='stack',
        title=dict(text='', x=0.5),
        title_font=dict(size=20, family='Arial, bold'),
        legend=dict(title='Cost Type', orientation='h', y=1.05, yanchor='bottom', x=0.5, xanchor='center', traceorder='normal'),
        margin=dict(l=40, r=40, t=80, b=40),
        plot_bgcolor='rgba(0,0,0,0)',
        width=width * 2,
        height=height * num_rows
    )

    return fig
