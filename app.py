import dash
from dash import Dash, html, dcc, Input, Output, callback
import pandas as pd
import plotly.express as px
import numpy as np #numerical edits
import plotly.express as px #Dynamic graphs mostly used for raw data

#from datetime import datetime, timedelta, date #Time manipulation



external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css',
                        'https://fonts.googleapis.com/css2?family=Atkinson+Hyperlegible&display=swap']

app = Dash(__name__, external_stylesheets=external_stylesheets)
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
    "Darkness": [19.93, 19.97, 19.81, 19.47],
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

app.layout = html.Div([
    html.Div([
        html.Label("Select Dataset"),
        dcc.Dropdown(
            id='dataset-selector',
            options=[{'label': name, 'value': name} for name in datasets.keys()],
            value='Kishwauketoe, Williams Bay, WI'
        ),
    ], style={'width': '48%', 'display': 'inline-block', 'backgroundColor': "#e6eff1"}),


    html.Div([
        dcc.Graph(id='darkness-graph'),
        dcc.Graph(id='detailed-graph')
    ]),

    html.Div(
        id='prediction-label',
        style={'padding': '10px', 'backgroundColor': "#e6eff1"}),
    
    html.Div([
        dcc.Input(id='year-input', type='number', value=2025, step=1),
        html.Div(id='prediction-output', style={'marginTop': '10px', 'fontWeight': 'bold'})
    ], style={'padding': '20px', 'backgroundColor': "#e6eff1"})
], style={'fontFamily': 'Atkinson Hyperlegible, sans-serif'})

@callback(
    Output('darkness-graph', 'figure'),
    Input('dataset-selector', 'value'),
    #Input('year-selector', 'value'),
)
def update_graph(dataset_name):
    df = datasets[dataset_name]
    fig = px.line(df, x='Time', y='Darkness', markers=True, title=f'{dataset_name} Darkness Over Time', color_discrete_sequence=["#002DB3"])
    fig.update_traces(marker=dict(size=10))  # Default is usually 6
    
    degreeYear = 1
    x_data = df['Time']
    y_data = df['Darkness']

    coefficients = np.polyfit(x_data, y_data, 1)
    slope = coefficients[0]
    intercept = coefficients[1]

    p = np.poly1d(coefficients)
    fig.add_scatter(x=x_data, y=p(x_data), mode='lines', name=f'Best Fit Line with slope = {slope}', line=dict(color="#2b7751", width=2, dash='solid'))


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
        ))
    fig.update_traces(line=dict(width=3))
    return fig
@callback(
    Output('detailed-graph', 'figure'),
    Input('darkness-graph', 'hoverData'),
    Input('dataset-selector', 'value')
)

def update_detailed_graph(hoverData, dataset_name):
    if hoverData is None:
        fig = px.scatter(title="Hover Over a Datapoint Above to See Yearly Data!")
        fig.update_layout(
            plot_bgcolor='pink',
            paper_bgcolor="#beb7be",
            font=dict(
                family='Atkinson Hyperlegible, sans-serif',
                size=14,
                color='black'),
            title_font=dict(
                family='Atkinson Hyperlegible, sans-serif',
                size=20,
                color='#2c3e50')
        )
        return fig
    
    year = hoverData['points'][0]['x']
    dataset_for_year = detailed_datasets.get(dataset_name, {}).get(year, None)
    if dataset_for_year is None:
        return px.scatter(title=f"No data for {dataset_name} in {year}")
    fig = px.scatter(dataset_for_year, x=dataset_for_year.columns[0], y=dataset_for_year.columns[1], title=f'{dataset_name} {year} Data', color_discrete_sequence=["#190038"])
    fig.update_traces(mode='markers')
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
            range=[5, 30]
        ))
    

    # Base scatter plot
    #fig = px.scatter(dataset_for_year, x=x_col, y=y_col, title=f'{dataset_name} {year} Data', color_discrete_sequence=["#190038"])
    #fig.update_traces(mode='markers')

    # Add best-fit line

    cleaned_df2 = dataset_for_year.dropna(subset=['night_of', 'median_darkness'])

    # Sort by date/time or x_col
    sorted_df2 = cleaned_df2.sort_values(by='night_of')
    degree2 = 0
    xs2 = pd.to_datetime(sorted_df2['night_of']).map(pd.Timestamp.toordinal)
    ys2 = sorted_df2['median_darkness']

    coeffss2 = np.polyfit(xs2, ys2, degree2)
    polys2 = np.poly1d(coeffss2)

    x_denses2 = np.linspace(xs2.min(), xs2.max(), 1000)
    y_poly_fits2 = polys2(x_denses2)
    dates_denses2 = [pd.Timestamp.fromordinal(int(d)) for d in x_denses2]

    fig.add_scatter(x=dates_denses2, y=y_poly_fits2, mode='lines', name=f'Best Fit Line {polys2}', line=dict(color="#2b7751", width=2, dash='solid'))

    # Layout styling
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor="#c9d8cf",
        font=dict(
            family='Atkinson Hyperlegible, sans-serif',
            size=18,
            color='black'
        ),
        title_font=dict(
            family='Atkinson Hyperlegible, sans-serif',
            size=20,
            color='#2c3e50'
        ),
        xaxis=dict(
            title="Night of Measurement",
            showgrid=True,
            gridcolor='#5F3B53',
            linecolor='black',                
            ticks='outside'
        ),
        yaxis=dict(
            title="Median Darkness",
            showgrid=True,
            gridcolor="#5F3B53",
            linecolor='black',
            ticks='outside',
            range=[5, 30]
        ),
        hovermode='x unified'
    )
    fig.update_traces(line=dict(width=3))
    return fig

@callback(
    Output('prediction-label', 'children'),
    Input('dataset-selector', 'value')
)
def update_prediction_label(dataset_name):
    return f"Enter a Year to Predict Darkness from {dataset_name}:"

@callback(
    Output('prediction-output', 'children'),
    Input('year-input', 'value'),
    Input('dataset-selector', 'value')
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
    return f"Predicted Darkness for {year}: {predicted_darkness:.2f} mag/arcsec²."
    
if __name__ == '__main__':
    app.run(debug=True)
