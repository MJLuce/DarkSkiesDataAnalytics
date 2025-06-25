import dash
from dash import html, dcc


dash.register_page(__name__, path="/")

layout = html.Div([
    html.H1(
        "Welcome to Dark Skies Data Analytics!", 
        style={'textAlign': 'center', 
                'padding': '60px', 
                'fontSize':'60px', 
                'marginBottom': '50px',
                'color': "#f9f9f9",
                'background': 'radial-gradient(circle, #29035a, #29035a, #f9f9f9)'
                }),  
    html.Div([
        dcc.Markdown('''
        Have you ever wondered why you see less stars in the middle of a city than in a rural region?  Artificially sourced light, 
        such as from glowing skyscrapers or city lamps, scatters off of dust and substances in the atmosphere. This is called
        light pollution. 

        ''', style={'width':'60%', 'margin': '0 auto', 'textAlign': 'center', 'padding': '1px', 'fontSize':'24px', 'marginBottom': '50px'})
    ]),
    html.Div([
        dcc.Markdown('''
        At GLAS Education, we use local Sky Quality Meters to collect data on light pollution. The
        data is measured in units of "magnitudes per arcsecond squared." To summarize this jargon, the units are distorted,
        so higher values indicate darker skies. The per arcsecond squared  just means that the data is measured
        with respect to a specific size region in the sky. 
        To understand the scale of the data, the values range from 18 to 22, 18 representing 
        urban levels of light pollution and 22 is a near perfect dark sky.

        ''', style={'width':'60%', 'margin': '0 auto', 'textAlign': 'center', 'padding': '1px', 'fontSize':'24px', 'marginBottom': '50px'}),
        html.Img(
            src="/assets/DarkSkyInfographic.jpg",
            style={
                'marginTop': '1px',
                'width': '50%',
                'borderRadius': '10px',
                'display': 'flex',
                'margin': '0 auto',
                'marginBottom': '50px'  # centers the image horizontally
            }
        ),
        dcc.Markdown('''
        Our data is measuring the darkness around Geneva Lake in southeastern Wisconsin. Geneva Lake
        is a tourist destination surrounded by three decently sized towns: Fontana, Williams Bay, and (most urban) Lake Geneva).
        Geneva Lake is equidistant, also, from Chicago, Milwaukee, and Madison. As a result, we have
        mild suburban approaching bright suburban levels of light pollution.

        ''', style={'width':'60%', 'margin': '0 auto', 'textAlign': 'center', 'padding': '1px', 'fontSize':'24px', 'marginBottom': '50px'})
    
    ]),

    html.Footer("© 2025 Geneva Lakes Astrophysics and STEAM", style={'textAlign': 'center', 'padding': '20px'})

], style={'fontFamily': 'Atkinson Hyperlegible, sans-serif'})