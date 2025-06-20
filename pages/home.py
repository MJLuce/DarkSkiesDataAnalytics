import dash
from dash import html


dash.register_page(__name__, path="/")

layout = html.Div([
    html.H1("Welcome to the Darkness Dashboard"),
    html.P("This is the homepage.")
], style={'fontFamily': 'Atkinson Hyperlegible, sans-serif'})