from dash import Dash, html, dcc, page_container, page_registry

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css',
                        'https://fonts.googleapis.com/css2?family=Atkinson+Hyperlegible&display=swap']
app = Dash(__name__, use_pages=True, suppress_callback_exceptions=True, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    html.Div([
        dcc.Link("Home", href="/", className="nav-link"),
        dcc.Link("Unweighted Graphs", href="/unweighted-graphs", className="nav-link"),
        dcc.Link("Weighted Graphs", href="/weighted-graphs", className="nav-link"),
  
    ], className="nav-bar"),
    html.Hr(),
    page_container  # This loads the current page
])

server = app.server

if __name__ == "__main__":
    app.run(debug=True)