import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf
import pandas_ta as ta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from datetime import datetime, timedelta
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import dash
from dash import html, dcc
from dash.dependencies import Output, Input


df = pd.read_csv("data/marketing_campaign.csv")

df["Date"] = pd.to_datetime(df["Date"])
df["Acquisition_Cost"] = df["Acquisition_Cost"].str.replace('$', '', regex=False).str.replace(',', '', regex=False)
df["Acquisition_Cost"] = pd.to_numeric(df["Acquisition_Cost"])
df["Conversions"] = round(df["Clicks"] * (df["Conversion_Rate"] / 100))

max_conversion_b = html.B(children=[], id="max")
min_conversion_b = html.B(children=[], id="min")

ROI = html.B(children=[], id="ROI")
CVR = html.B(children=[], id="CVR")
CPC = html.B(children=[], id="CPC")

vars = [
    {"label":"Tipo de campaña","value":"Campaign_Type"},
    {"label":"Compañía","value":"Company"},
    {"label":"Canal usado","value":"Channel_Used"},
    {"label":"Locación","value":"Location"}
]

app = dash.Dash(__name__)
server = app.server

app.layout =  html.Div(id="body", className="e6_body", children=[
    html.H1("Análisis de dimensiones personalizadas", id="H1", className="e6_title"),
    dcc.Dropdown(id="dropdown_vars", style={"width":"160px"}, className="e6_dropdown_1",
                        options=vars,
                        value="Campaign_Type",
                        multi=False,
                        clearable=False),
    html.Div(id="graph_div_1", className="e6_graph_div_1", children=[
        html.Div(id="KPI_div_1", className="e6_KPI_div_1", children=[
            html.P(max_conversion_b, className="e6_KPI_1", style={"margin-right":"25px","width":"300px"}),
            html.P(min_conversion_b, className="e6_KPI_1", style={"margin-left":"25px","width":"300px"})
        ]),
        dcc.Graph(id="conversions_analysis", figure={}, className="e6_graph_1", style={"width":"74%"})
    ]),
    html.A(href="https://github.com/genagithub/proyecto-6/blob/main/forecating_de_conversiones_para_iniciativas_de_marketing.ipynb", children=[html.H2("Pronóstico de conversiones (iniciativas de marketing)", id="H2", className="e6_title")]),
    html.Div(id="forecast_div", className="e6_forecast_div", children=[
        html.Div(id="KPI_div_2", className="e6_KPI_div_2", children=[
            html.Div(className="e6_KPI_2", children=[html.P("ROI", className="e6_KPI_title"), html.P([ROI,"%"], className="e6_KPI_p")]),
            html.Div(className="e6_KPI_2", children=[html.P("CVR", className="e6_KPI_title"), html.P([CVR,"%"], className="e6_KPI_p")]),
            html.Div(className="e6_KPI_2", children=[html.P("CPC", className="e6_KPI_title"), html.P(["$",CPC], className="e6_KPI_p")])
        ]),
        html.Div(id="graph_div_2", className="e6_graph_div_2", children=[
        html.Div(id="dropdown_div", className="e6_dropdown_div", children=[
            dcc.Dropdown(id="dropdown_var1", className="e6_dropdown_2",
                        options=df["Campaign_Type"].unique(),
                        value=df["Campaign_Type"].unique()[0],
                        multi=False,
                        clearable=False),
            dcc.Dropdown(id="dropdown_var2", className="e6_dropdown_2",
                        options=df["Company"].unique(),
                        value=df["Company"].unique()[0],
                        multi=False,
                        clearable=False),
            dcc.Dropdown(id="dropdown_var3", className="e6_dropdown_2",
                        options=df["Channel_Used"].unique(),
                        value=df["Channel_Used"].unique()[0],
                        multi=False,
                        clearable=False),
            dcc.Dropdown(id="dropdown_var4", className="e6_dropdown_2",
                        options=df["Location"].unique(),
                        value=df["Location"].unique()[0],
                        multi=False,
                        clearable=False)
        ]),
        dcc.Graph(id="forecasting", figure={}, className="e6_graph_2") 
        ])   
    ])   
])


@app.callback(
    [Output(component_id="max",component_property="children"),
    Output(component_id="min",component_property="children"),
    Output(component_id="conversions_analysis",component_property="figure"),
    Output(component_id="dropdown_var1",component_property="style"),
    Output(component_id="dropdown_var2",component_property="style"),
    Output(component_id="dropdown_var3",component_property="style"),
    Output(component_id="dropdown_var4",component_property="style"),
    Output(component_id="forecasting",component_property="figure"),
    Output(component_id="ROI",component_property="children"),
    Output(component_id="CVR",component_property="children"),
    Output(component_id="CPC",component_property="children")],
    [Input(component_id="dropdown_vars",component_property="value"),
    Input(component_id="dropdown_var1",component_property="value"),
    Input(component_id="dropdown_var2",component_property="value"),
    Input(component_id="dropdown_var3",component_property="value"),
    Input(component_id="dropdown_var4",component_property="value")]
)

def update_graph(slct_var, slct_campaign, slct_company, slct_channel, slct_location):
    
    campaign_style = {"position":"absolute","top":"0","left":"0"}
    company_style = {"position":"absolute","top":"0","left":"0"}
    channel_style = {"position":"absolute","top":"0","left":"0"}
    location_style = {"position":"absolute","top":"0","left":"0"}
    
    if slct_var == "Campaign_Type":        
        campaign_style["zIndex"] = 5   
        df_segment = df[df[slct_var] == slct_campaign].copy()
    elif slct_var == "Company":
        company_style["zIndex"] = 5
        df_segment = df[df[slct_var] == slct_company].copy()
    elif slct_var == "Channel_Used":        
        channel_style["zIndex"] = 5
        df_segment = df[df[slct_var] == slct_channel].copy()
    else:        
        location_style["zIndex"] = 5
        df_segment = df[df[slct_var] == slct_location].copy()
        
    df_grouped = df.groupby(slct_var)["Conversions"].sum().reset_index()
    idx_min = df_grouped["Conversions"].idxmin()
    idx_max = df_grouped["Conversions"].idxmax()
        
    var_min = df_grouped.loc[idx_min, slct_var]
    var_max = df_grouped.loc[idx_max, slct_var]
        
    value_min = df_grouped.loc[idx_min, "Conversions"]
    value_max = df_grouped.loc[idx_max, "Conversions"]
    
    min_conversion = f"Mín: {var_min} ({int(value_min)})"
    max_conversion = f"Máx: {var_max} ({int(value_max)})"

    df_grouped = df.groupby(["Month", slct_var])["Conversions"].sum().reset_index()
    
    slct_label = next((opt["label"] for opt in vars if opt["value"] == slct_var), "No encontrado")
    line_chart = px.line(df_grouped, x="Month", y="Conversions", color=slct_var, markers=True, title=f"Éxito anual de Conversiones por {slct_label}")
    line_chart.update_layout(yaxis_title="sumatoria por mes")
    
    df_ts = df_segment.groupby("Date").agg({
        "Conversions": "sum",
        "Acquisition_Cost": "sum",
        "Clicks": "sum",
        "ROI": "mean",           
        "Conversion_Rate": "mean"
    }).reset_index().sort_values("Date")

    df_ts["Conv_Lag1"] = df_ts["Conversions"].shift(1)
    df_ts["ROI_Lag1"] = df_ts["ROI"].shift(1)
    df_ts["day_of_week"] = df_ts["Date"].dt.dayofweek
    
    df_model = df_ts.dropna()

    features = ["Conv_Lag1", "ROI_Lag1", "day_of_week"]
    targets = ["Conversions", "ROI", "Conversion_Rate"]
    
    X = df_model[features]
    y = df_model[targets]
    
    rf = RandomForestRegressor(n_estimators=50, random_state=42)
    rf.fit(X, y)

    future_dates = pd.date_range(df_ts["Date"].max() + pd.Timedelta(days=1), periods=14)
    
    last_conv = df_ts["Conversions"].iloc[-1]
    last_roi = df_ts["ROI"].iloc[-1]
    mean_cost = df_ts["Acquisition_Cost"].mean()
    mean_clicks = df_ts["Clicks"].mean()
    
    dates, conversions, ROIs, CVRs, CPCs = [], [], [], [], []
    curr_conv, curr_roi = last_conv, last_roi

    for date in future_dates:
        X_input = pd.DataFrame([[curr_conv, curr_roi, date.dayofweek]], columns=features)
        res = rf.predict(X_input)[0] 
        cpc_pred = mean_cost / mean_clicks
        
        dates.append(date)
        conversions.append(res[0])
        ROIs.append(res[1])
        CVRs.append(res[2])
        CPCs.append(cpc_pred)
    
        curr_conv, curr_roi = res[0], res[1]
        
    df_forecast = pd.DataFrame({
        "Date": dates,
        "Conversions": conversions,
        "ROI": ROIs,
        "CVR": CVRs,
        "CPC": CPCs
    })
        
    forecasting = go.Figure()
    forecasting.add_trace(go.Scatter(x=df_ts["Date"], y=df_ts["Conversions"], mode="lines", fill="tozeroy", fillcolor="rgba(0, 0, 255, 0.2)", name="Conversiones Históricas"))
    forecasting.add_trace(go.Scatter(x=df_forecast["Date"], y=df_forecast["Conversions"], mode="lines", fill="tozeroy", fillcolor="rgba(255, 165, 0, 0.2)", name="Pronóstico de 14 días"))
    forecasting.update_layout(title_text=" ", yaxis_title=" ")
    
    ROI = round(df_forecast["ROI"].mean(), 2)
    CVR = round(df_forecast["CVR"].mean(), 2)
    CPC = round(df_forecast["CPC"].mean(), 2)

    return max_conversion, min_conversion, line_chart, campaign_style, company_style, channel_style, location_style, forecasting, ROI, CVR, CPC

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050)) 
    app.run_server(host='0.0.0.0', port=port)
