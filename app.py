import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.dates as mdates
import streamlit as st
import os
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import HuberRegressor, LinearRegression
from sklearn.metrics import r2_score
import update_summaries
from scipy.stats import linregress
import sqlite3
# import glob
from windrose import WindroseAxes
from datetime import datetime, timedelta
from PIL import Image
import math
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

plot_size = (18, 4)
update_summaries.main()

selceted_site = ""
selceted_site_bold = ""


def get_db(sqlite_file):
    db_path = sqlite_file
    conn = sqlite3.connect(db_path)
    query = "SELECT * FROM summary"
    merged_df = pd.read_sql(query, conn)
    # Close the connection
    conn.close()
    merged_df["DateTime"] = pd.to_datetime(merged_df["DateTime"])
    merged_df.set_index("DateTime", inplace=True)
    return merged_df


st.set_page_config(
    page_title="Daily EC data monitor",
    layout="wide",
    page_icon="🗼",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .section1 {
        background-color: #f0f0f0;  /* Change to your desired color */
        padding: 1em;
        border-radius: 25px;
        text-align: left;
        font-size: 0.75em;
    }
    .section2 {
        background-color: #f0f0f0;  /* Change to your desired color */
        padding: 20px;
        border-radius: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

sites = update_summaries.sites

with st.sidebar:
    st.title("Navigation")
    selceted_site = st.radio("Select EC site:", sites)
    selceted_site_bold = "$\mathbf{" + selceted_site + "}$"
    days_limit = st.number_input("Enter days to show:", step=7, min_value=1, value=14)
    st.markdown("---")
    st.header("Sensor separation calculator:")
    separation = st.number_input(
        "Sonic-Gas analyzer distance (cm):",
        min_value=0.0,
        max_value=25.0,
        value=20.0,
        step=1.0,
    )
    angle_from_north = st.number_input(
        "Sonic to Gas analyzer angle (0-360):",
        min_value=0.0,
        max_value=359.9,
        value=275.0,
        step=1.0,
    )
    sensor_east = separation * math.sin(math.radians(angle_from_north))
    sensor_west = separation * math.cos(math.radians(angle_from_north))
    st.markdown(f"Eastward (X) : **{round(sensor_east, 2)} cm**.")
    st.markdown(f"Northward (Y) : **{round(sensor_west,2)} cm**.")
    st.markdown("---")

ticks = np.clip(int(days_limit / 10), 1, 15)

script_path = os.getcwd()
head_col1, head_col2 = st.columns([8,1])
with head_col1:
    st.markdown(
    """
    <div class='section1' style='text-decoration:underline;'>
    <h1 > Upper CO River Basin Commision (UCRBC) Project</h1>
    <h3> Monitoring EC data from all sites.</h3>
    </div>
    """,
    unsafe_allow_html=True,
    )
with head_col2:
    image_self = Image.open(r"Data/SatelliteImage/self.jpg")
    st.image(image=image_self, caption="Site created by Ashish Masih.", width=100)
    email_link = f"###### [**@ Contact:**](mailto:amasih2@unl.edu?subject=Subject&body=Message)"
    st.markdown(email_link, unsafe_allow_html=True)
    # st.markdown(email_link, unsafe_allow_html=True)


st.markdown(f"## {selceted_site_bold} EC tower data:-")
st.markdown("- This is raw and uncorrected data and should be used only to review sensor's functionality.")
date_range_placeholder = st.empty()
date_range_placeholder.text("")

st.text("")
st.text("")


def clean_column(df, col_name, window=13, threshold=2.5):
    if col_name == "P_RAIN_1_1_1":
        return df
    try:
        rolling_median = df[col_name].rolling(window=window, center=True).median()
        rolling_std = df[col_name].rolling(window=window, center=True).std()
        outliers = np.abs(df[col_name] - rolling_median) > (threshold * rolling_std)
        # Create a copy of the original data
        df_cleaned = df.copy()    
        # Replace outliers with NaN
        df_cleaned.loc[outliers, col_name] = np.nan
        return df_cleaned[col_name]
    
    except:
        return df[col_name]

def plot_et(df):
    """Display bar plot of daily values (from hourly summed values.)"""
    ET_Daily = df["ET"].resample("D").sum()
    ET_Daily = ET_Daily / 2
    fig, ax = plt.subplots(figsize=plot_size)
    bar_width = np.clip((0.0023 * ET_Daily.shape[0] + 0.0204), 0.025, 0.2)
    ax.bar(
        ET_Daily.index,
        ET_Daily.values,
        label="Cumulative daily ET (mm)",
        width=bar_width,
    )

    ax.xaxis.set_major_locator(mdates.DayLocator(interval=ticks))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d-%y"))
    plt.xticks(rotation=45, ha="right")
    ax.set_xlim(date_range)
    ax.set_ylim(0, 12)
    ax.set_title(selceted_site_bold + f": Daily cumulative ET (mm).")
    ax.legend(loc="lower left")
    st.pyplot(fig)
    plt.close()


def plot_temperatures(df):
    col_name1 = "sonic_temperature"
    col_name2 = "TA_1_1_1"
    col_name3 = "TC_1_1_1"
    col_name4 = "air_temperature"

    fig, ax = plt.subplots(figsize=(18,6))
    ax.plot(
        df.index,
        clean_column(df, col_name1) - 273.15,
        linewidth=0.5,
        label="Sonic air temperature(C)", marker='o', markersize=0.5
    )
    ax.plot(
        df.index,
        clean_column(df, col_name2) - 273.15,
        linewidth=1.5,
        label="Vaisala Air temperature(C)",
        linestyle='dashed', marker='o', markersize=0.5
    )
    ax.plot(
        df.index,
        clean_column(df, col_name4) - 273.15,
        linewidth=1.5,
        label="LI-7500 temperature(C)",
        linestyle='dotted', marker='o', markersize=0.5
    )
    try:
        ax.plot(
            df.index,
            clean_column(df, col_name3) - 273.15,
            linewidth=0.5,
            label="IRT Canopy temperature(C)", marker='o', markersize=0.5
        )
        ax.set_title(
            selceted_site_bold
            + ": Sonic, Vaisala, LI-7500 & IRT Canopy temperatures."
        )
    except:
        ax.set_title(selceted_site_bold + ": Sonic, Vaisala & LI-7500 (C)")

    locator = mdates.AutoDateLocator(minticks=5, maxticks=12)
    formatter = mdates.AutoDateFormatter(locator)
    formatter.scaled[1/(24*60)] = '%y/%m/%d %H:%M'
    formatter.scaled[1/24] = '%y/%m/%d %H:%M'
    formatter.scaled[1] = '%y/%m/%d'
    ax.xaxis.set_major_formatter(formatter)
    locator = ax.xaxis.set_major_locator(locator)
    ax.grid(True)
    plt.xticks(rotation=45, ha="right")
    ax.set_xlim(date_range)
    ax.set_ylim(-15, 40)
    ax.legend(loc="lower left")
    st.pyplot(fig)
    plt.close()


def plot_SWC(merged_df):
    col_name1 = "SWC_1_1_1"
    col_name2 = "SWC_2_1_1"
    col_name3 = "SWC_3_1_1"
    col_name4 = "SWC_4_1_1"
    col_name5 = "SWC_5_1_1"
    col_name6 = "SWC_6_1_1"

    fig, ax = plt.subplots(figsize=plot_size)
    ax.plot(
        merged_df.index,
        clean_column(merged_df, col_name1),
        linewidth=1,
        label="SWC_1_1_1 m3/m3", marker='o', markersize=0.5
    )
    ax.plot(
        merged_df.index,
        clean_column(merged_df, col_name2),
        linewidth=1,
        label="SWC_2_1_1 m3/m3", marker='o', markersize=0.5
    )
    ax.plot(
        merged_df.index,
        clean_column(merged_df, col_name3),
        linewidth=1,
        label="SWC_3_1_1 m3/m3", marker='o', markersize=0.5
    )

    ax.plot(
        merged_df.index,
        clean_column(merged_df, col_name4),
        linewidth=1,
        label="SWC_4_1_1 m3/m3", marker='o', markersize=0.5
    )
    ax.plot(
        merged_df.index,
        clean_column(merged_df, col_name5),
        linewidth=1,
        label="SWC_5_1_1 m3/m3", marker='o', markersize=0.5
    )
    ax.plot(
        merged_df.index,
        clean_column(merged_df, col_name6),
        linewidth=1,
        label="SWC_6_1_1 m3/m3", marker='o', markersize=0.5
    )

    ax.xaxis.set_major_locator(mdates.DayLocator(interval=ticks))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d-%y"))
    plt.xticks(rotation=45, ha="right")
    ax.set_xlim(date_range)
    ax.set_ylim(0, 0.75)
    ax.set_title(selceted_site_bold + ": Soil water content Hydra Probes (all)")
    ax.legend(loc="lower left")
    ax.invert_yaxis()
    ax.grid(True)
    st.pyplot(fig)
    plt.close()


def plot_RH_regression(merged_df):
    col_name1 = "RH"
    col_name2 = "RH_1_1_1"
    components = pd.DataFrame()
    components["RH_1_1_1"] = merged_df[col_name1].values
    components["RH"] = merged_df[col_name2].values
    components = components.dropna()
    X = components["RH_1_1_1"]
    X = X.to_numpy().reshape(-1, 1)

    y = components["RH"].to_numpy().ravel()

    # Perform Huber Regression
    huber = HuberRegressor()
    huber.fit(X, y)
    huber_slope = huber.coef_[0]
    huber_intercept = huber.intercept_
    huber_predicted_y = huber.predict(X)
    huber_r2 = r2_score(y, huber_predicted_y)
    huber_regression = "Huber Regression:"
    huber_regression += f"\n    Intercept: {huber_intercept:.4f}"
    huber_regression += f"\n    Slope: {huber_slope:.4f}"
    huber_regression += f"\n    R2: {huber_r2:.4f}"

    # Perform Linear Regression
    linear = LinearRegression()
    linear.fit(X, y)
    linear_slope = linear.coef_[0]
    linear_intercept = linear.intercept_
    linear_predicted_y = linear.predict(X)
    linear_r2 = r2_score(y, linear_predicted_y)

    linear_regression = "Linear Regression:"
    linear_regression += f"\n    Intercept: {linear_intercept:.4f}"
    linear_regression += f"\n    Slope: {linear_slope:.4f}"
    linear_regression += f"\n    R2: {float(linear_r2):.4f}"

    # Plot the results
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(X, y, s=1, label="Data points")
    ax.plot(X, huber_predicted_y, color="brown", linewidth=2, label="Huber Regression")
    ax.plot(X, linear_predicted_y, color="blue", linewidth=2, label="Linear Regression")
    ax.plot((0,100), (0,100), color="gray", linewidth=2, label="1:1 Line")
    ax.set_xlabel("Relative humidity from Vaisala (%)")
    ax.set_ylabel("Relative humidity from LI-7500 (%)")
    ax.set_title(
        selceted_site_bold
        + ": "
        + "RH comparisoin Vaisala (X-axis) Vs Gas analyzer RH (Y-axis)"
    )
    ax.text(0.05, 0.72, f"{linear_regression}", color="blue", transform=ax.transAxes)
    ax.text(0.05, 0.58, f"{huber_regression}", color="brown", transform=ax.transAxes)
    ax.text(
        0.12,
        0.05,
        "Regression equation is calculated using 'Huber-Regresion'\n"+
        "to ignore larger outliers (e.g. explore Cora).",
        transform=ax.transAxes
    )
    ax.set_ylim(-5, 115)
    ax.set_xlim(-5, 115)
    ax.legend()
    ax.grid(True)
    return fig


def plot_RH(df):
    col_name1 = "RH_1_1_1"
    col_name2 = "RH"
    fig, ax = plt.subplots(figsize=plot_size)
    ax.plot(
        df.index,
        clean_column(df, col_name1),
        linewidth=1,
        label="RH (vaisala)", marker='o', markersize=0.5
    )
    ax.plot(
        df.index,
        clean_column(df, col_name2),
        linewidth=1,
        label="RH (gas analyzer)", marker='o', markersize=0.5
    )

    ax.xaxis.set_major_locator(mdates.DayLocator(interval=ticks))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d-%y"))
    plt.xticks(rotation=45, ha="right")
    ax.set_xlim(date_range)
    ax.set_title(
        selceted_site_bold + ": Relative humidity from Vaisala (2m) and gas analyzer (3m-4m)."
    )
    ax.legend(loc="lower left")
    ax.set_ylim(0,105)
    ax.grid(True)
    st.pyplot(fig)
    plt.close()


def plot_co2signal(df):
    col_name = "co2_signal_strength_7500_mean"
    fig, ax = plt.subplots(figsize=plot_size)
    ax.plot(df.index, clean_column(df, col_name), linewidth=1, marker='o', markersize=0.5)

    ax.xaxis.set_major_locator(mdates.DayLocator(interval=ticks))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d-%y"))
    plt.xticks(rotation=45, ha="right")
    plt.grid()
    ax.set_xlim(date_range)
    ax.set_ylim(60, 105)
    ax.grid(True)
    ax.set_title(selceted_site_bold + ": " + col_name + " (%)")
    st.pyplot(fig)
    plt.close()

def plot_bowen_ratio(df):
    col_name = "bowen_ratio"
    df = df[(df["daytime"] > 0)]
    fig, ax = plt.subplots(figsize=plot_size)
    ax.plot(df.index, np.zeros(df.shape[0]), linewidth=2, color="gray")
    ax.plot(df.index, clean_column(df, col_name,window=5), linewidth=1, marker='o', markersize=0.5)

    ax.xaxis.set_major_locator(mdates.DayLocator(interval=ticks))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d-%y"))
    plt.xticks(rotation=45, ha="right")
    ax.grid(True)
    ax.set_xlim(date_range)
    ax.set_ylim(-2, 3)
    ax.set_title(selceted_site_bold + ": " + col_name + " (only daytime values) un-spiked")
    st.pyplot(fig)
    plt.close()

def plot_SHF(df):
    col_name1 = "SHF_1_1_1"
    col_name2 = "SHF_2_1_1"
    col_name3 = "SHF_3_1_1"
    fig, ax = plt.subplots(figsize=plot_size)
    ax.plot(
        df.index,
        clean_column(df, col_name1),
        linewidth=1,
        label="SHF_1_1_1 (W/m2)", marker='o', markersize=0.5
    )
    ax.plot(
        df.index,
        clean_column(df, col_name2),
        linewidth=1,
        label="SHF_2_1_1 (W/m2)", marker='o', markersize=0.5
    )
    ax.plot(
        df.index,
        clean_column(df, col_name3),
        linewidth=1,
        label="SHF_3_1_1 (W/m2)", marker='o', markersize=0.5
    )

    ax.xaxis.set_major_locator(mdates.DayLocator(interval=ticks))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d-%y"))
    plt.xticks(rotation=45, ha="right")
    ax.set_xlim(date_range)
    ax.set_ylim(-75, 100)
    ax.set_title(selceted_site_bold + ": Soil Heat flux 1,2 & 3 (W/m2)")
    ax.legend(loc="lower left")
    ax.grid(True)
    st.pyplot(fig)
    plt.close()


def plot_closure(df):
    fig, ax = plt.subplots(figsize=(9, 9))
    try:
        components = pd.DataFrame()
        Yvals = clean_column(df, "RN_1_1_1") - clean_column(
            df, "SHF_3_1_1"
        )
        Xvals = clean_column(df, "LE") + clean_column(df, "H")
        components["RN_G"] = Yvals
        components["LE_H"] = Xvals
        components["daytime"] = clean_column(df, "daytime")
        components["u*"] = clean_column(df, "u*")
        eb_ratio_array = components["LE_H"].to_numpy() / components["RN_G"].to_numpy()
        eb_ratio_array = np.clip(eb_ratio_array, -0.5, 1.5)
        components["EB_Array"] = eb_ratio_array
        eb_ratio = round((components["LE_H"].sum() / components["RN_G"].sum()), 4)
        components = components[(components["daytime"] > 0)]
        components = components[(components["u*"] >= 0.15)]
        components = components.dropna()
        # maxval = max(components.RN_G.max(), components.LE_H.max())
        regression_results = linregress(components.RN_G, components.LE_H)
        slope = regression_results.slope
        intercept = regression_results.intercept
        rvalue = regression_results.rvalue
        x_fit = np.linspace(
            components.RN_G.min(), components.RN_G.max(), components.shape[0]
        )
        y_fit = slope * x_fit + intercept

        scatter = ax.scatter(
            components.RN_G.values,
            components.LE_H.values,
            c=components["EB_Array"],
            cmap="plasma",
            label="Energy balance closure",
            s=2,
        )
        plt.colorbar(scatter, ax=ax, label="EBR (unitless)")
        ax.plot(x_fit, y_fit, color="red", label="Regression line")
        # Create the equation string
        equation = f"Linear regression:\n    y = {slope:.2f}x + {intercept:.2f}\n    $R^2$={round(rvalue**2,4)}"
        # Annotate the plot with the equation
        ax.text(
            0.05,
            0.9,
            transform=ax.transAxes,
            s=equation,
            color="red",
            fontsize=12,
            ha="left",
        )
        eq_string = r"$EBR = \frac {\sum{(LE+H)}}{\sum{(RN-G)}} =$"
        ax.text(
            0.05,
            0.8,
            transform=ax.transAxes,
            s=f"Energy Balance Ratio:-\n    {eq_string} {eb_ratio}",
            color="brown",
            fontsize=12,
            ha="left",
        )
        zero_x = components.RN_G.values.reshape(-1, 1)
        zero_y = components.LE_H.values.reshape(-1, 1)
        zero_model = LinearRegression(fit_intercept=False)
        zero_model.fit(zero_x, zero_y)
        zero_slope = zero_model.coef_[0]
        zero_y_pred = zero_slope * zero_x
        zero_r_squared = zero_model.score(zero_x, zero_y)
        ax.plot(zero_x, zero_y_pred, color="blue", label="Regression line zero offset")
        zero_eq_string = f"Zero offset slope = {round(zero_slope[0],4)};\n     $R^2$ = {round(zero_r_squared,4)}"
        ax.text(
            0.05,
            0.7,
            transform=ax.transAxes,
            s=zero_eq_string,
            color="blue",
            fontsize=12,
            ha="left",
        )
        # ax.set_xlim(-50,maxval*1.1)
        # ax.set_ylim(-50, maxval*1.1)
        ax.set_xlim(-100, 900)
        ax.set_ylim(-100, 900)
    except Exception as ex:
        print(ex)
        pass
    ax.set_title(
        selceted_site_bold
        + ": "
        + "Energy balance closure (daytime hours; Friction velocity u*>=0.15 m/s)"
    )
    ax.set_xlabel("RN-G (W/m2)")
    ax.set_ylabel("LE + H (W/m2)")
    ax.legend(loc="upper right")
    return fig

def combo_temperature_shf_plot(df, date_range):
    col_name1 = "TS_1_1_1"
    col_name2 = "TS_2_1_1"
    col_name3 = "TS_3_1_1"

    col_name4 = "SHF_1_1_1"
    col_name5 = "SHF_2_1_1"
    col_name6 = "SHF_3_1_1"

    col_name7 = "TS_7_1_1"
    col_name8 = "TS_8_1_1"
    col_name9 = "TS_9_1_1"

    # col_name10 = "TC_1_1_1"

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Scatter(x=df.index, y=df[col_name1]-273.15, name='HydraProbe_1 temperature (C)'), secondary_y=False )
    fig.add_trace(go.Scatter(x=df.index, y=df[col_name2]-273.15, name='HydraProbe_2 temperature (C)'), secondary_y=False )
    fig.add_trace(go.Scatter(x=df.index, y=df[col_name3]-273.15, name='HydraProbe_3 temperature (C)'), secondary_y=False )  
    # fig.add_trace(go.Scatter(x=df.index, y=df[col_name10]-273.15, name='Canopy temperature (C)'), secondary_y=False )  

    fig.add_trace(go.Scatter(x=df.index, y=clean_column(df,col_name7,window=48)-273.15, name='SoilProbe_1 temperature (C)'), secondary_y=False)
    fig.add_trace(go.Scatter(x=df.index, y=clean_column(df,col_name8,window=48)-273.15, name='SoilProbe_2 temperature (C)'), secondary_y=False)    
    fig.add_trace(go.Scatter(x=df.index, y=clean_column(df,col_name9,window=48)-273.15, name='SoilProbe_3 temperature (C)'), secondary_y=False)

    fig.add_trace(go.Scatter(x=df.index, y=clean_column(df,col_name4), name='Soil Heat Flux 1 (W/m2)'), secondary_y=True)
    fig.add_trace(go.Scatter(x=df.index, y=clean_column(df,col_name5), name='Soil Heat Flux 2 (W/m2)'), secondary_y=True)    
    fig.add_trace(go.Scatter(x=df.index, y=clean_column(df,col_name6), name='Soil Heat Flux 3 (W/m2)'), secondary_y=True)

    fig.update_layout(
        title=f"{selceted_site}: Comparision of Hydra probe & Soil temperature probe Vs Soil Heat Flux diurnal pattern.",
        title_x=0.15,
        xaxis=dict(range = date_range ),
        xaxis_title='Date',
        legend_title='Columns',
        hovermode='x unified'
    )
    fig.update_yaxes(title_text="<b>Hydra probe & soil probe temperatures (C)</b>", secondary_y=False)
    fig.update_yaxes(title_text="<b>SHF (W/m2)</b>", secondary_y=True)
    fig.update_traces(mode='lines')

    st.plotly_chart(fig, use_container_width=True)



def plot_hydra_probe_temperatures(df):
    col_name4 = "TS_4_1_1"
    col_name5 = "TS_5_1_1"
    col_name6 = "TS_6_1_1"
    fig, ax = plt.subplots(figsize=plot_size)

    # fig, ax = plt.subplots(figsize=plot_shape)
    ax.plot(
        df.index,
        clean_column(df, col_name4) - 273.15,
        linewidth=1,
        label="TS_4_1_1 (C)", marker='o', markersize=0.5
    )
    ax.plot(
        df.index,
        clean_column(df, col_name5) - 273.15,
        linewidth=1,
        label="TS_5_1_1 (C)", marker='o', markersize=0.5
    )
    ax.plot(
        df.index,
        clean_column(df, col_name6) - 273.15,
        linewidth=1,
        label="TS_6_1_1 (C)", marker='o', markersize=0.5
    )

    ax.xaxis.set_major_locator(mdates.DayLocator(interval=ticks))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d-%y"))
    plt.xticks(rotation=45, ha="right")
    ax.set_xlim(date_range)
    ax.set_ylim(0, 25)
    ax.set_title(f"{selceted_site_bold}: Soil temperatures from hydra probes at 20, 40 & 60 cm depths.")
    ax.legend(loc="lower left")
    ax.grid(True)
    st.pyplot(fig)
    plt.close()


def plot_temperature_probe(df):
    col_name1 = "TS_7_1_1"
    col_name2 = "TS_8_1_1"
    col_name3 = "TS_9_1_1"
    fig, ax = plt.subplots(figsize=plot_size)
    ax.plot(
        df.index,
        clean_column(df, col_name1) - 273.15,
        linewidth=1,
        label="TS_7_1_1 (C)", marker='o', markersize=0.5
    )
    ax.plot(
        df.index,
        clean_column(df, col_name2) - 273.15,
        linewidth=1,
        label="TS_8_1_1 (C)", marker='o', markersize=0.5
    )
    ax.plot(
        df.index,
        clean_column(df, col_name3) - 273.15,
        linewidth=1,
        label="TS_9_1_1 (C)", marker='o', markersize=0.5
    )

    ax.xaxis.set_major_locator(mdates.DayLocator(interval=ticks))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d-%y"))
    plt.xticks(rotation=45, ha="right")
    ax.set_xlim(date_range)
    ax.set_ylim(0, 45)
    ax.set_title(selceted_site_bold + ": Soil temperature from (LI-7800-180) probes")
    ax.legend(loc="lower left")
    ax.grid(True)
    st.pyplot(fig)
    plt.close()


def plot_ppdf_swin(df):
    col_name1 = "PPFD_1_1_1"
    col_name2 = "SWIN_1_1_1"
    fig, ax = plt.subplots(figsize=plot_size)
    ax.plot(
        df.index,
        clean_column(df, col_name1) * 0.51,
        color="red",
        linewidth=1,
        label="PPFD (factor= 0.5100) (W/m2", marker='o', markersize=0.5
    )

    ax.xaxis.set_major_locator(mdates.DayLocator(interval=ticks))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d-%y"))
    ax.plot(
        df.index,
        clean_column(df, col_name2),
        color="blue",
        linewidth=1,
        label="SWIN (W/m2)", marker='o', markersize=0.5
    )
    ax.set_xlim(date_range)
    ax.set_ylim(-25, 1100)
    plt.xticks(rotation=45, ha="right")
    ax.set_title(selceted_site_bold + ": Quantum sensor & Shortwave_In")
    ax.legend(loc="lower left")
    ax.grid(True)
    st.pyplot(fig)
    plt.close()


def plot_solar_components(df):
    col_name1 = "SWIN_1_1_1"
    col_name2 = "SWOUT_1_1_1"
    col_name3 = "LWIN_1_1_1"
    col_name4 = "LWOUT_1_1_1"
    fig, ax = plt.subplots(figsize=plot_size)
    ax.plot(
        df.index,
        clean_column(df, col_name1),
        linewidth=1,
        label="SWIN (W/m2)", marker='o', markersize=0.5
    )
    ax.plot(
        df.index,
        clean_column(df, col_name2),
        linewidth=1,
        label="SWOUT (W/m2)", marker='o', markersize=0.5
    )
    ax.plot(
        df.index,
        clean_column(df, col_name3),
        linewidth=1,
        label="LWIN (W/m2)", marker='o', markersize=0.5
    )
    ax.plot(
        df.index,
        clean_column(df, col_name4),
        linewidth=1,
        label="LWOUT (W/m2)", marker='o', markersize=0.5
    )

    ax.xaxis.set_major_locator(mdates.DayLocator(interval=ticks))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d-%y"))
    plt.xticks(rotation=45, ha="right")
    ax.set_xlim(date_range)
    ax.set_ylim(-25, 1100)
    ax.set_title(
        selceted_site_bold
        + ": "
        + col_name1
        + "; "
        + col_name2
        + "; "
        + col_name3
        + "; "
        + col_name4
    )
    ax.legend(loc="lower left")
    ax.grid(True)
    st.pyplot(fig)
    plt.close()


def plot_precip(df):
    ET_Daily = df["P_RAIN_1_1_1"].resample("D").sum()
    ET_Daily = ET_Daily * 100
    col_name1 = "P_RAIN_1_1_1"
    fig, ax = plt.subplots(figsize=plot_size)
    bar_width = np.clip((0.0023 * ET_Daily.shape[0] + 0.0204), 0.025, 0.2)
    ax.bar(ET_Daily.index, ET_Daily.values, label="Liquid precip (cm)", width=bar_width)

    ax.xaxis.set_major_locator(mdates.DayLocator(interval=ticks))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d-%y"))
    plt.xticks(rotation=45, ha="right")
    ax.set_xlim(date_range)
    ax.set_ylim(0, 2.54 * 2)
    ax.set_title(selceted_site_bold + ": Liquid precipitation cumulative daily (cm).")
    ax.legend(loc="lower left")
    st.pyplot(fig)
    plt.close()


def energy_balance_ratio(df):
    try:
        df = df[(df["daytime"] > 0)]
        RN = df["RN_1_1_1"].to_numpy()
        LE = df["LE"].to_numpy()
        G = df["SHF_1_1_1"].to_numpy()
        H = df["H"].to_numpy()
        df.loc[:,'EBR'] = df.apply((LE + H)/(RN  - G), axis=1)
        EBR = clean_column(df, 'EBR',window=15,threshold=2)

        fig, ax = plt.subplots(figsize=plot_size)
        ax.plot(df.index, EBR, linewidth=1)
        ax.plot(df.index, np.ones(df.shape[0]), linewidth=1, color="gray")    
        # ax.plot(merged_df.index, np.zeros(merged_df.shape[0]), linewidth=2, color="gray")     
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=ticks))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d-%y"))
        plt.xticks(rotation=45, ha="right")
        ax.set_xlim(date_range)
        ax.set_title(selceted_site_bold + ": " + "Hourly Energy Balance Ratio (LE+H)/(RN-G) (De-spiked, only daytime values)")
        ax.set_ylim(-1, 2)
        st.pyplot(fig)
        plt.close()
    except:
        pass

def plot_co2_comparision(df):
    col_name1 = "co2_molar_density"
    col_name2 = "air_pressure"
    col_name3 = "air_temperature"
    airPressure = clean_column(df, col_name2)
    airTemperature = clean_column(df, col_name3)
    CO2NumberDensity = 425.55 * airPressure / (8.314 * airTemperature) * 1 / 1000
    fig, ax = plt.subplots(figsize=plot_size)
    ax.plot(
        df.index,
        clean_column(df, col_name1),
        color="red",
        linewidth=1,
        label="LI-7500 CO2 molar density mmol/m3", marker='o', markersize=0.5
    )

    ax.xaxis.set_major_locator(mdates.DayLocator(interval=ticks))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d-%y"))
    ax.plot(
        df.index,
        CO2NumberDensity,
        color="blue",
        linewidth=1,
        label="Estimated CO2 number density mmol/m3", marker='o', markersize=0.5
    )
    ax.set_xlim(date_range)
    plt.xticks(rotation=45, ha="right")
    ax.set_title(
        selceted_site_bold + ": LI-7500 CO2 molar density & CO2 number density"
    )
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
    plt.close()



def wind_rose(df):
    # Create a windrose plot
    df = df[(df["daytime"] > 0)]
    fig, ax = plt.subplots(subplot_kw={"projection": "windrose"})
    ax.bar(
        df["wind_dir"],
        df["wind_speed"],
        normed=True,
        opening=0.8,
        edgecolor="white",
        nsector=36,
        cmap=plt.get_cmap('gist_ncar'),
        bins=[0, 2, 5, 7, 10, 15, 19.99],
    )
    ax.set_title(f"Windrose: {selceted_site_bold}")
    ax.set_legend(
        title="Windrose (m/s)",
        loc="lower left",
        bbox_to_anchor=(-0.1, -0.1),
        bbox_transform=ax.transAxes,
    )
    col1, col2, col3 = st.columns(3)
    try:
        with col1:
            st.markdown(f"**Windrose: {selceted_site_bold} (daytime hours).**")
            st.pyplot(fig)
            plt.close()
    except:
        st.text("Variable not found")
        pass
    try:
        with col2:
            string1 = "###### Windrose"
            string2 = ": Nearest weatherstation."
            st.markdown(f"{string1}{string2}")
            relative_path = update_summaries.windroses[f"{selceted_site}"]
            wind_rose_path = f"{script_path}\{relative_path}"
            wind_rose_path = wind_rose_path.replace("\\", "/")
            image = Image.open(wind_rose_path)

            st.image(image, caption="")
    except:
        st.text("Variable not found")
        pass

    try:
        with col3:
            st.markdown(f"**Satellite image: {selceted_site_bold}.**")
            relative_path = update_summaries.satellite_images[selceted_site]
            satellite_img_path = f"{script_path}/{relative_path}"
            satellite_img_path = satellite_img_path.replace("\\", "/")
            # st.markdown(f"Satellite image path: {satellite_img_path}")
            image = Image.open(satellite_img_path)

            st.image(image, caption="")
    except:
        st.text("Variable not found")
        pass


merged_df = pd.DataFrame()
dbPath = f"{script_path}/Data/{str(selceted_site)}/summaries/{str(selceted_site)}.db"
# st.write(f"{dbPath}")
merged_df = get_db(dbPath)
start_date = merged_df.index.values[0]
data_max_date = merged_df.index.values[-1]
today = datetime.now()
today_end = today.replace(hour=23, minute=59, second=59) # + timedelta(days=1)
end_date = np.datetime64(today_end)
limit_date = end_date - pd.Timedelta(days=days_limit)

if data_max_date<limit_date:
    st.markdown("## Dates are out of range. Increase interval and check if tower was functional.")
    st.markdown(f"#### Data starts from: {np.datetime64(start_date, 'D')} & ends on: {np.datetime64(data_max_date, 'D')}")
    st.markdown(f"#### Wheras data requested from: {datetime.date(limit_date)} to: {np.datetime64(end_date, 'D')}")
    st.stop()
if limit_date < start_date:
    limit_date = start_date
    limit_date = limit_date.astype('datetime64[ns]').astype('datetime64[D]').astype(limit_date.dtype)
else:
    limit_date = limit_date.replace(hour=0, minute=0, second=0)


date_range = [limit_date, end_date]
merged_df = merged_df.sort_index(axis=1)
merged_df = merged_df[
    (merged_df.index >= date_range[0]) & (merged_df.index <= end_date)
]
date_start = datetime.date(merged_df.index[0])
date_end = datetime.date(merged_df.index[-1])  
difference_days = (date_end-date_start).days

date_range_placeholder.markdown(
    f"- **Displayed date range:** {datetime.date(merged_df.index[0])} to {datetime.date(merged_df.index[-1])}; (**{difference_days}+ days**)"
)

plt.rcParams["figure.facecolor"] = "lightcyan"
plt.rcParams["axes.facecolor"] = "lightcyan"

plot_RH(merged_df)

plot_co2signal(merged_df)
plot_bowen_ratio(merged_df)
plot_precip(merged_df)
energy_balance_ratio(merged_df)

col1, col2 = st.columns(2)
with col1:
    fig1 = plot_RH_regression(merged_df)
    st.pyplot(fig1)
    plt.close()
with col2:
    fig2 = plot_closure(merged_df)
    st.pyplot(fig2)
    plt.close()

wind_rose(merged_df)

plot_ppdf_swin(merged_df)

plot_co2_comparision(merged_df)

# outer area color
plt.rcParams["figure.facecolor"] = "whitesmoke"
# Setting the axes background color
plt.rcParams["axes.facecolor"] = "whitesmoke"

st.header("Additional sensors data:-")

plot_solar_components(merged_df)

plot_et(merged_df)

plot_SWC(merged_df)

plot_temperatures(merged_df)

plot_hydra_probe_temperatures(merged_df)

# plot_temperature_probe(merged_df)

# plot_SHF(merged_df)

combo_temperature_shf_plot(merged_df, date_range)

if st.button("**Display time-series data:**"):
    st.dataframe(merged_df)