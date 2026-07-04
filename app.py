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

factor_top_b = html.B(children=[], id="factor")
value_top_b = html.B(children=[], id="value")

ROI = html.B(children=[], id="ROI")
CVR = html.B(children=[], id="CVR")
CPC = html.B(children=[], id="CPC")

vars = [
    {"label":"Tipo de campaña","value":"Campaign_Type"},
    {"label":"Compañía","value":"Company"},
    {"label":"Canal usado","value":"Channel_Used"},
    {"label":"Locación","value":"Location"}
]

random_forest_forecast = RandomForestRegressor(n_estimators=100, 
                                               max_depth=6,        
                                               min_samples_leaf=5,   
                                               random_state=42)

app = dash.Dash(__name__)
server = app.server

app.layout =  html.Div(id="body", className="e6_body", children=[
    html.H1("Análisis por dimensiones personalizadas", id="H1", className="e6_title"),
    html.Div(id="dropdown_div_1", className="e6_dropdown_div_1", children=[
        dcc.Dropdown(id="dropdown_vars", className="e6_dropdown_1",
                    options=vars,
                    value="Campaign_Type",
                    multi=False,
                    clearable=False)
    ]),
    html.Div(id="graph_div_1", className="e6_graph_div_1", children=[
        html.Div(id="KPI_div_1", className="e6_KPI_div_1", children=[
            html.P(factor_top_b, className="e6_KPI_1", style={"margin-right":"25px","width":"310px"}),
            html.P(value_top_b, className="e6_KPI_1", style={"margin-left":"25px","width":"310px"})
        ]),
        dcc.Graph(id="conversions_analysis", figure={}, className="e6_graph_1", style={"width":"74%"})
    ]),
    html.A(href="https://github.com/genagithub/proyecto-6/blob/main/forecating_por_dimensiones_personalizadas.ipynb", children=[html.H2("Proyección de conversiones (modelos promocionales)", id="H2", className="e6_title")]),
    html.Div(id="forecast_div", className="e6_forecast_div", children=[
        html.Div(id="KPI_div_2", className="e6_KPI_div_2", children=[
            html.Div(className="e6_KPI_2", children=[html.P("ROI", className="e6_KPI_title"), html.P([ROI,"%"], className="e6_KPI_p")]),
            html.Div(className="e6_KPI_2", children=[html.P("CVR", className="e6_KPI_title"), html.P([CVR,"%"], className="e6_KPI_p")]),
            html.Div(className="e6_KPI_2", children=[html.P("CPC", className="e6_KPI_title"), html.P(["$",CPC], className="e6_KPI_p")])
        ]),
        html.Div(id="graph_div_2", className="e6_graph_div_2", children=[
        html.Div(id="dropdown_div_2", className="e6_dropdown_div_2", children=[
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
    [Output(component_id="factor",component_property="children"),
    Output(component_id="value",component_property="children"),
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


def update_forecast(slct_var, slct_campaign, slct_company, slct_channel, slct_location):
    
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

    all_categorical_vars = ["Campaign_Type", "Company", "Channel_Used", "Location"]
    categorical_features = [v for v in all_categorical_vars if v != slct_var]

    df_ts = df_segment.groupby(["Date"] + categorical_features).agg({
        "Conversions": "sum",
        "Acquisition_Cost": "sum",
        "Clicks": "sum",
        "Impressions": "sum",
        "ROI": "mean",
        "Conversion_Rate": "mean"
    }).reset_index().sort_values("Date")

    df_ts["Date"] = pd.to_datetime(df_ts["Date"])

    df_ts["CTR"] = (df_ts["Clicks"] / df_ts["Impressions"]).replace([np.inf, -np.inf], 0).fillna(0)
    df_ts["CPC"] = (df_ts["Acquisition_Cost"] / df_ts["Clicks"]).replace([np.inf, -np.inf], 0).fillna(0)
    df_ts["CPM"] = ((df_ts["Acquisition_Cost"] / df_ts["Impressions"]) * 1000).replace([np.inf, -np.inf], 0).fillna(0)

    df_ts["Conv_Lag1"] = df_ts["Conversions"].shift(1)
    df_ts["ROI_Lag1"] = df_ts["ROI"].shift(1)
    df_ts["Clicks_Lag1"] = df_ts["Clicks"].shift(1)
    df_ts["day_of_week"] = df_ts["Date"].dt.dayofweek

    df_model = df_ts.dropna()

    if df_model.empty:
        raise PreventUpdate

    numerical_features = ["Conv_Lag1", "ROI_Lag1", "Clicks_Lag1", "day_of_week", "CTR", "CPC", "CPM"]
    targets = ["Conversions", "ROI", "Conversion_Rate"]

    X_raw = df_model[numerical_features + categorical_features]
    y = df_model[targets]
    X = pd.get_dummies(X_raw, columns=categorical_features)
    
    if hasattr(random_forest_forecast, "feature_names_in_"):
        X = X.reindex(columns=random_forest_forecast.feature_names_in_, fill_value=0)

    random_forest_forecast.fit(X, y)

    final_features = list(X.columns)
    importance = random_forest_forecast.feature_importances_

    df_imp = pd.DataFrame({
        "factor": final_features,
        "importance": importance
    }).sort_values(by="importance", ascending=True)

    idx_max_imp = df_imp["importance"].idxmax()
    factor_top = df_imp.loc[idx_max_imp, "factor"]
    value_top = round(df_imp.loc[idx_max_imp, "importance"] * 100, 1)

    factor_top_text = f"Impulsor Principal: {factor_top}"
    value_top_text = f"Influencia en el Éxito: {value_top}%"
    
    slct_label = next((opt["label"] for opt in vars if opt["value"] == slct_var), "Variable")

    bar_chart = px.bar(
        df_imp.tail(3), 
        x="importance", 
        y="factor", 
        orientation="h",
        title=f"Impacto de los Factores en el éxito por {slct_label}",
        color="importance",
        color_continuous_scale="Blues"
    )
    
    bar_chart.update_layout(
        height=350,
        autosize=False,  
        showlegend=False,
        margin=dict(l=20, r=20, t=40, b=20),
        template="plotly_white"
    )

    future_dates = pd.date_range(df_model["Date"].max() + pd.Timedelta(days=1), periods=14)
    last_row = df_model.iloc[-1]

    mean_cost = df_ts["Acquisition_Cost"].mean()
    mean_clicks = df_ts["Clicks"].mean()
    last_cpc = mean_cost / mean_clicks if mean_clicks > 0 else 0

    dates = [last_row["Date"]]
    conversions = [last_row["Conversions"]]
    ROIs = [last_row["ROI"]]
    CVRs = [last_row["Conversion_Rate"]]
    CPCs = [last_cpc]
    
    curr_conv = last_row["Conversions"]
    curr_roi = last_row["ROI"]
    curr_clicks_lag = mean_clicks
    curr_ctr = last_row["CTR"]
    curr_cpc = last_row["CPC"]
    curr_cpm = last_row["CPM"]
    
    dummy_context_row = X.iloc[[-1]].copy()

    for date in future_dates:
        dummy_context_row["Conv_Lag1"] = curr_conv
        dummy_context_row["ROI_Lag1"] = curr_roi
        dummy_context_row["Clicks_Lag1"] = curr_clicks_lag
        dummy_context_row["day_of_week"] = date.dayofweek
        dummy_context_row["CTR"] = curr_ctr
        dummy_context_row["CPC"] = curr_cpc
        dummy_context_row["CPM"] = curr_cpm
        
        res_raw = random_forest_forecast.predict(dummy_context_row)
        
        pred_conv = res_raw[0][0]
        pred_roi = res_raw[0][1]
        pred_cvr = res_raw[0][2]
        cpc_pred = mean_cost / mean_clicks if mean_clicks > 0 else 0
        
        dates.append(date)
        conversions.append(pred_conv)
        ROIs.append(pred_roi)
        CVRs.append(pred_cvr)
        CPCs.append(cpc_pred)
    
        curr_conv, curr_roi = pred_conv, pred_roi
        
    df_forecast = pd.DataFrame({
        "Date": dates,
        "Conversions": conversions,
        "ROI": ROIs,
        "CVR": CVRs,
        "CPC": CPCs
    })

    df_ts_recent = df_ts[df_ts["Date"] >= "2021-09-01"]
    
    forecasting = go.Figure()
    forecasting.add_trace(go.Scatter(x=df_ts_recent["Date"], y=df_ts_recent["Conversions"], mode="lines", fill="tozeroy", fillcolor="rgba(0, 0, 255, 0.2)", name="Conversiones Históricas"))
    forecasting.add_trace(go.Scatter(x=df_forecast["Date"], y=df_forecast["Conversions"], mode="lines", fill="tozeroy", fillcolor="rgba(255, 165, 0, 0.2)", name="Pronóstico de 14 días"))
    forecasting.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=40, b=20))
    
    ROI = round(df_forecast["ROI"].mean(), 2)
    CVR = round(df_forecast["CVR"].mean(), 2)
    CPC = round(df_forecast["CPC"].mean(), 2)

    return factor_top_text, value_top_text, bar_chart, campaign_style, company_style, channel_style, location_style, forecasting, ROI, CVR, CPC


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050)) 
    app.run_server(host='0.0.0.0', port=port)
