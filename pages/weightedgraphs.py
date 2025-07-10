from flask import Flask
import dash
from dash import Dash, html, dcc, Input, Output, callback
import pandas as pd
import plotly.express as px
import numpy as np #numerical edits
import plotly.express as px #Dynamic graphs mostly used for raw data
import plotly.graph_objects as go

#from datetime import datetime, timedelta, date #Time manipulation


dash.register_page(__name__, path="/weighted-graphs")

layout = html.Div([
    html.H2("Weighted Darkness Graphs"),
    dcc.Graph(id='weighted-graph')  # you can plug in your real graph here
])


data0TP = pd.read_csv('filtereddataTP0.csv')
data1TP = pd.read_csv('filtereddataTP1.csv')
data2TP = pd.read_csv('filtereddataTP2.csv')
data3TP = pd.read_csv('filtereddataTP3.csv')
data4TP = pd.read_csv('filtereddataTP4.csv')
#data5TP = pd.read_csv('TP-DataCompiled - TP2020.csv')
data1Kish = pd.read_csv('filtereddataKish1.csv')
data2Kish = pd.read_csv('filtereddataKish2.csv')
data3Kish = pd.read_csv('filtereddataKish3.csv')
data4Kish = pd.read_csv('filtereddataKish4.csv')
#data5Kish = pd.read_csv('filtereddataKish1.csv')
# Sample datasets
dfKish = pd.DataFrame({
    "Time": [2021, 2022, 2023, 2024],
    "Darkness": [20.0112328767, 19.82477778, 19.83640909, 19.574852941],
})

dfTP = pd.DataFrame({
    "Time": [2020, 2021, 2022, 2023],
    "Darkness": [20.17, 20.12, 19.64, 19.91]
})

datasets = {
    "Kishwauketoe, Williams Bay, WI": dfKish,
    "Ted Peters, North Linn, WI": dfTP
}
detailed_datasets = {
    'Kishwauketoe, Williams Bay, WI': {
        2021: data1Kish,
        2022: data2Kish,
        2023: data3Kish,
        2024: data4Kish,
    },
    'Ted Peters, North Linn, WI': {
        2020: data0TP,
        2021: data1TP,
        2022: data2TP,
        2023: data3TP,
        2024: data4TP,
    }
}
layout = html.Div([
    html.Div([
        html.Label("Select Dataset"),
        dcc.Dropdown(
            id='dataset-selector2',
            options=[{'label': name, 'value': name} for name in datasets.keys()],
            value='Kishwauketoe, Williams Bay, WI'
        ),
    ], style={'width': '48%', 'display': 'inline-block', 'backgroundColor': "#e6eff1"}),


    html.Div([
        dcc.Graph(id='darkness-graph2'),
        dcc.Graph(id='bar-graph2')
    ]),

    html.Div(
        id='prediction-label2',
        style={'padding': '10px', 'backgroundColor': "#e6eff1"}),
    
    html.Div([
        dcc.Input(id='year-input2', type='number', value=2025, step=1),
        html.Div(id='prediction-output2', style={'marginTop': '10px', 'fontWeight': 'bold'})
    ], style={'padding': '20px', 'backgroundColor': "#e6eff1"})
], style={'fontFamily': 'Atkinson Hyperlegible, sans-serif'})

@callback(
    Output('darkness-graph2', 'figure'),
    Input('dataset-selector2', 'value'),
    #Input('year-selector', 'value'),
)

def update_graph(dataset_name):
    yearly_data = detailed_datasets.get(dataset_name, {})
    mean_darkness_per_year = {}
    sem_per_year = {}

    for year, df in yearly_data.items():
        if isinstance(df, pd.DataFrame) and 'median_darkness' in df.columns:

            if len(df['median_darkness']) > 0:
                mean_darkness = np.mean(df['median_darkness'])
                sem = np.std(df['median_darkness'], ddof=1) / np.sqrt(len(df['median_darkness']))
                mean_darkness_per_year[year] = mean_darkness
                sem_per_year[year] = sem
 
            else:
                print(f"No valid data for Darkness in {year}")
        else:
            print(f"Skipping year {year}: missing DataFrame or 'Darkness' column.")
            print(f"Year {year}: columns = {df.columns}")

    if not mean_darkness_per_year:
        return go.Figure()  # empty graph if no data

    # Build arrays
    years = sorted(mean_darkness_per_year.keys())
    x_data = np.array([int(year) for year in years])
    y_data = np.array([mean_darkness_per_year[year] for year in years])
    y_err = np.array([sem_per_year[year] for year in years])

    # Weighted linear regression
    w1 = 1 / y_err**2
    x_bar1 = np.sum(w1 * x_data) / np.sum(w1)
    y_bar1 = np.sum(w1 * y_data) / np.sum(w1)

    numerator = np.sum(w1 * (x_data - x_bar1) * (y_data - y_bar1))
    denominator = np.sum(w1 * (x_data - x_bar1)**2)
    slope1 = numerator / denominator
    slope_se = np.sqrt(1 / denominator)
    intercept1 = y_bar1 - slope1*x_bar1
    y_fit = slope1 * x_data + intercept1

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_data,
        y=y_data,
        mode='markers',
        name='Data',
        error_y=dict(
            type='data',
            array=y_err,
            visible=True,
            color='#002DB3',
            thickness=3,
            width=6
        ),
        marker=dict(size=10, color='#002DB3')
    ))

    # Best-fit line
    fig.add_trace(go.Scatter(
        x=x_data,
        y=y_fit,
        mode='lines',
        name=f'Weighted Fit (slope = {slope1:.4f} +/- {slope_se:.4f})',
        line=dict(color="#2b7751", width=2)
    ))


    fig.update_layout(hovermode='x unified',
        plot_bgcolor='white',
        paper_bgcolor="#e6eff1",
        font=dict(
            family='Atkinson Hyperlegible, sans-serif',
            size=14,
            color='black'
        ),
        title_font=dict(
            family='Atkinson Hyperlegible, sans-serif',
            size=20,
            color='#2c3e50'
        ),
        xaxis=dict(
            title='Year',
            showgrid=True,
            gridcolor='#5F3B53',
            linecolor='black',
            ticks='outside'
        ),
        yaxis=dict(
            title='Darkness',
            showgrid=True,
            gridcolor="#5F3B53",
            linecolor='black',
            ticks='outside'
        ),
        autosize=True,
        )
    fig.update_traces(line=dict(width=3))
    return fig

dataset_dict = {
    "Total Data": pd.DataFrame({
        "datasets": ["Kishwauketoe", "Ted Peters"],
        "slope1": [0.1198, 0.1014],
        "slope_se": [0.0441, 0.0350]
    }),

}
@callback(
    Output('bar-graph2', 'figure'),
    Input('dataset-selector2', 'value')
)

def bar_graph(dataset_name):
    df = dataset_dict["Total Data"]
    if df is None:
        # Return empty figure or placeholder if dataset not found
        return px.bar(title="No data available")

    fig = px.bar(
        df,
        x='datasets',
        y='slope1',
        error_y='slope_se',
        title='Slope Comparison',
        color_discrete_sequence=["#C7B1FA"]
    )
    
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor="#c9d8cf",
        font=dict(
            family='Atkinson Hyperlegible, sans-serif',
            size=14,
            color='black'
        ),
        title_font=dict(
            family='Atkinson Hyperlegible, sans-serif',
            size=20,
            color='#2c3e50'
        ),
        xaxis=dict(
            title='Year',
            showgrid=True,
            gridcolor='#5F3B53',
            linecolor='black',
            ticks='outside'
        ),
        yaxis=dict(
            title='Darkness',
            showgrid=True,
            gridcolor="#5F3B53",
            linecolor='black',
            ticks='outside',
        )
    )
    return fig


@callback(
    Output('prediction-label2', 'children'),
    Input('dataset-selector2', 'value')
)
def update_prediction_label(dataset_name):
    return f"Enter a Year to Predict Darkness from {dataset_name}:"

@callback(
    Output('prediction-output2', 'children'),
    Input('year-input2', 'value'),
    Input('dataset-selector2', 'value')
)


def predict_darkness(year, dataset_name):
    if year is None:
        return "Please enter a year."

    # Sort by date/time or x_col
    df = datasets[dataset_name]

    degreeYear = 1
    x_data = df['Time']
    y_data = df['Darkness']

    coefficients = np.polyfit(x_data, y_data, 1)
    slope = coefficients[0]
    intercept = coefficients[1]

    p = np.poly1d(coefficients)

    predicted_darkness = p(year)
    return f"Predicted Darkness for {year}: {predicted_darkness:.2f} mag/arcsec² using weighted data."

