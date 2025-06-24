import dash
from dash import html, dcc


dash.register_page(__name__, path="/")

layout = html.Div([
    html.H1("Welcome to Dark Skies Data Analytics!", style={'textAlign': 'center', 'padding': '1px'}),  

    html.Div([
        dcc.Markdown('''
        Have you ever wondered why you see less stars in the middle of a city than in a rural region?  Artificial lighting, 
        such as glowing skyscrapers or city lamps, scatters off of dust and substances in the atmosphere. This is called
        light pollution. 

        ''', style={'textAlign': 'center', 'padding': '1px'})
    ]),
    html.Footer("© 2025 Geneva Lakes Astrophysics and STEAM", style={'textAlign': 'center', 'padding': '20px'})

], style={'fontFamily': 'Atkinson Hyperlegible, sans-serif'})