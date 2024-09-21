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
import glob
from windrose import WindroseAxes
from datetime import datetime, timedelta
from PIL import Image
import math
plot_shape = (18, 4)
update_summaries.main()

selceted_site = ""

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
sites = update_summaries.sites

with st.sidebar:
    st.title("Navigation")
    selceted_site = st.radio("Select EC site:", sites)
    days_limit = st.number_input("Enter days to show:", step=7, min_value=1, value=21)   
    st.markdown("---")
    st.header("Sensor separation calculator:")
    separation = st.number_input("Sonic-Gas analyzer distance (cm):", min_value=0.0, max_value=25.0, value=20.0, step=1.0)
    angle_from_north = st.number_input("Enter Sonic to Gas analyzer angle (0-360):", min_value=0.0, max_value=359.9, value=275.0, step=1.0)
    sensor_east = separation * math.sin(math.radians(angle_from_north))
    sensor_west = separation * math.cos(math.radians(angle_from_north))
    st.markdown(f"Eastward separation : **{round(sensor_east, 2)} cm**.")
    st.markdown(f"Northward spearation: **{round(sensor_west,2)} cm**.")
    st.markdown("---")

ticks = np.clip(int(days_limit / 10), 1, 15)

script_path = os.getcwd()
# all_files = glob.glob(script_path + "**/**/*", recursive=True)
st.markdown(
    "<h1 style='text-align: center; text-decoration:underline;'> Upper CO River Basin Commision (UCRBC) Project</h2>",
    unsafe_allow_html=True,
)
# st.markdown("### Daily EC summary data from all online towers in CO, WY, NM & NE")
st.markdown(f"## **{selceted_site}**:-")
st.markdown(
    f"- This is raw and uncorrected data and should be used only to review sensor's functionality."
)
date_range_placeholder = st.empty()
date_range_placeholder.text("")

st.text("")
st.text("")


def clean_column(df, colName):
    if colName == "P_RAIN_1_1_1":
        return df
    try:
        Q1 = df[colName].quantile(0.1)
        Q3 = df[colName].quantile(0.9)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[colName] = df[colName].where(
            (df[colName] >= lower_bound) & (df[colName] <= upper_bound), np.nan
        )
        return df[colName]
    except:
        return df[colName]


def plot_ET(merged_df):
    ET_Daily = merged_df["ET"].resample("D").sum()
    ET_Daily = ET_Daily / 2
    fig, ax = plt.subplots(figsize=plot_shape)
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
    ax.set_ylim(0,12)
    ax.set_title(selceted_site + f": Daily cumulative ET (mm).")
    ax.legend(loc="lower left")
    st.pyplot(fig)
    plt.close()


def plot_temperatures(merged_df):
    colName1 = "sonic_temperature"
    colName2 = "TA_1_1_1"
    colName3 = "TC_1_1_1"
    colName4 = "TCNR4_C_1_1_1"

    fig, ax = plt.subplots(figsize=plot_shape)
    ax.plot(
        merged_df.index, clean_column(merged_df, colName1) - 273.15, linewidth=1, label="Sonic (C)"
    )
    ax.plot(merged_df.index, clean_column(merged_df, colName2) - 273.15, linewidth=1, label="Air (C)")
    try:
        ax.plot(merged_df.index, clean_column(merged_df, colName4), linewidth=1, label="TCNR4 (C)")
    except:
        pass
    try:
        ax.plot(
            merged_df.index,
            clean_column(merged_df, colName3) - 273.15,
            linewidth=1,
            label="Canopy (C)",
        )
        ax.set_title(
            selceted_site
            + ": Sonic, Air, Canopy temperature & NR body temperatures (C)"
        )
    except:
        ax.set_title(selceted_site + ": Sonic, Air & NR body temperatures (C)")

    ax.xaxis.set_major_locator(mdates.DayLocator(interval=ticks))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d-%y"))
    plt.xticks(rotation=45, ha="right")
    ax.set_xlim(date_range)
    ax.set_ylim(0,45)
    ax.legend(loc="lower left")
    st.pyplot(fig)
    plt.close()


def plot_horizontal_SWC(merged_df):
    colName1 = "SWC_1_1_1"
    colName2 = "SWC_2_1_1"
    colName3 = "SWC_3_1_1"
    colName4 = "SWC_4_1_1"
    colName5 = "SWC_5_1_1"
    colName6 = "SWC_6_1_1"
    
    fig, ax = plt.subplots(figsize=plot_shape)
    ax.plot(
        merged_df.index,
        clean_column(merged_df, colName1),
        linewidth=1,
        label="SWC_1_1_1 m3/m3",
    )
    ax.plot(
        merged_df.index,
        clean_column(merged_df, colName2),
        linewidth=1,
        label="SWC_2_1_1 m3/m3",
    )
    ax.plot(
        merged_df.index,
        clean_column(merged_df, colName3),
        linewidth=1,
        label="SWC_3_1_1 m3/m3",
    )

    ax.plot(
        merged_df.index,
        clean_column(merged_df, colName4),
        linewidth=1,
        label="SWC_4_1_1 m3/m3",
    )
    ax.plot(
        merged_df.index,
        clean_column(merged_df, colName5),
        linewidth=1,
        label="SWC_5_1_1 m3/m3",
    )
    ax.plot(
        merged_df.index,
        clean_column(merged_df, colName6),
        linewidth=1,
        label="SWC_6_1_1 m3/m3",
    )

    ax.xaxis.set_major_locator(mdates.DayLocator(interval=ticks))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d-%y"))
    plt.xticks(rotation=45, ha="right")
    ax.set_xlim(date_range)
    ax.set_ylim(0,1)
    ax.set_title(selceted_site + ": Soil water content Hydra Probes (all)")
    ax.legend(loc="lower left")
    ax.invert_yaxis()
    st.pyplot(fig)
    plt.close()


def plot_RH_regression(merged_df):
    colName1 = "RH"
    colName2 = "RH_1_1_1"
    components = pd.DataFrame()
    components["RH_1_1_1"] = merged_df[colName1].values
    components["RH"] = merged_df[colName2].values
    components = components.dropna()
    X = components[["RH_1_1_1"]]
    y = components[["RH"]].values

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Perform Huber Regression
    huber = HuberRegressor()
    huber.fit(X.values, y)
    huber_slope = huber.coef_[0]
    huber_intercept = huber.intercept_
    huber_yPredicted = huber.predict(X)
    huber_r2 = r2_score(y, huber_yPredicted)
    huber_regression = "Huber Regression:"
    huber_regression += f"\n    Intercept: {huber_intercept:.4f}"
    huber_regression += f"\n    Slope: {huber_slope:.4f}"
    huber_regression += f"\n    R2: {huber_r2:.4f}"

    # Perform Linear Regression
    linear = LinearRegression()
    linear.fit(X.values, y)
    linear_slope = linear.coef_[0]
    linear_intercept = linear.intercept_
    linear_yPredicted = linear.predict(X)
    linear_r2 = r2_score(y, linear_yPredicted)

    linear_regression = "Linear Regression:"
    linear_regression += f"\n    Intercept: {linear_intercept[0]:.4f}"
    linear_regression += f"\n    Slope: {linear_slope[0]:.4f}"
    linear_regression += f"\n    R2: {float(linear_r2):.4f}"

    # Plot the results
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(X, y, s=1, label="Data points")
    ax.plot(X, huber_yPredicted, color="brown", linewidth=2, label="Huber Regression")
    ax.plot(X, linear_yPredicted, color="blue", linewidth=2, label="Linear Regression")
    ax.set_xlabel("Relative humidity from Vaisala (%)")
    ax.set_ylabel("Relative humidity from LI-7500 (%)")
    ax.set_title(
        selceted_site
        + ": "
        + "RH comparisoin Vaisala (X-axis) Vs Gas analyzer RH (Y-axis)"
    )
    ax.text(0.05, 0.75, f"{linear_regression}", color="blue", transform=ax.transAxes)
    ax.text(0.05, 0.55, f"{huber_regression}", color="brown", transform=ax.transAxes)
    ax.text(
        0.05,
        0.05,
        f"Regression equation is calculated using 'Huber-Regresion' to ignore larger outliers,\n as it is not a systematic error (e.g. explore Cora).",
    )
    ax.set_ylim(-5, 105)
    ax.set_xlim(-5, 105)
    ax.legend()
    ax.grid(True)
    return fig


def plot_RH(merged_df):
    colName1 = "RH_1_1_1"
    colName2 = "RH"
    fig, ax = plt.subplots(figsize=plot_shape)
    ax.plot(
        merged_df.index,
        clean_column(merged_df, colName1),
        linewidth=1,
        label="RH (vaisala)",
    )
    ax.plot(
        merged_df.index,
        clean_column(merged_df, colName2),
        linewidth=1,
        label="RH (gas analyzer)",
    )

    ax.xaxis.set_major_locator(mdates.DayLocator(interval=ticks))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d-%y"))
    plt.xticks(rotation=45, ha="right")
    ax.set_xlim(date_range)
    ax.set_title(selceted_site + ": Relative humidity from Vaisala and gas analyzer.")
    ax.legend(loc="lower left")
    ax.grid(True)
    st.pyplot(fig)
    plt.close()


def plot_co2signal(merged_df):
    colName = "co2_signal_strength_7500_mean"
    fig, ax = plt.subplots(figsize=plot_shape)
    ax.plot(merged_df.index, clean_column(merged_df, colName), linewidth=1)

    ax.xaxis.set_major_locator(mdates.DayLocator(interval=ticks))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d-%y"))
    plt.xticks(rotation=45, ha="right")
    plt.grid()
    ax.set_xlim(date_range)
    ax.set_ylim(90, 100)
    ax.set_title(selceted_site + ": " + colName)
    st.pyplot(fig)
    plt.close()


def plot_SHF(merged_df):
    colName1 = "SHF_1_1_1"
    colName2 = "SHF_2_1_1"
    colName3 = "SHF_3_1_1"
    fig, ax = plt.subplots(figsize=plot_shape)
    ax.plot(
        merged_df.index,
        clean_column(merged_df, colName1),
        linewidth=1,
        label="SHF_1_1_1 (W/m2)",
    )
    ax.plot(
        merged_df.index,
        clean_column(merged_df, colName2),
        linewidth=1,
        label="SHF_2_1_1 (W/m2)",
    )
    ax.plot(
        merged_df.index,
        clean_column(merged_df, colName3),
        linewidth=1,
        label="SHF_3_1_1 (W/m2)",
    )

    ax.xaxis.set_major_locator(mdates.DayLocator(interval=ticks))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d-%y"))
    plt.xticks(rotation=45, ha="right")
    ax.set_xlim(date_range)
    ax.set_ylim(-50,100)
    ax.set_title(selceted_site + ": Soil Heat flux 1,2 & 3 (W/m2)")
    ax.legend(loc="lower left")
    # return fig
    st.pyplot(fig)
    plt.close()


def plot_closure(merged_df):
    fig, ax = plt.subplots(figsize=(9, 9))
    try:
        components = pd.DataFrame()
        Yvals = clean_column(merged_df, "RN_1_1_1") - clean_column(
            merged_df, "SHF_3_1_1"
        )
        Xvals = clean_column(merged_df, "LE") + clean_column(merged_df, "H")
        components["RN_G"] = Yvals
        components["LE_H"] = Xvals
        components["daytime"] = clean_column(merged_df, "daytime")

        eb_ratio_array = components["LE_H"].values / components["RN_G"].values
        eb_ratio_array = np.clip(eb_ratio_array, -0.5, 1.5)
        components["EB_Array"] = eb_ratio_array
        eb_ratio = round((components["LE_H"].sum() / components["RN_G"].sum()), 4)
        # components = components[
        #     (components.index.hour >= 7) & (components.index.hour < 19)
        # ]
        components = components[(components["daytime"]> 0)]
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
        ax.plot(zero_x, zero_y_pred, color='blue', label='Regression line zero offset')
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
    ax.set_title(selceted_site + ": " + "Energy balance closure (daytime hours)")
    ax.set_xlabel("RN-G (W/m2)")
    ax.set_ylabel("LE + H (W/m2)")
    ax.legend(loc="upper right")
    return fig

def plot_temperature_SHF(merged_df):
    colName1 = "TS_1_1_1"
    colName2 = "TS_2_1_1"
    colName3 = "TS_3_1_1"
    colName4 = "TS_4_1_1"
    colName5 = "TS_5_1_1"
    colName6 = "TS_6_1_1"
    fig, ax = plt.subplots(figsize=plot_shape)
    ax.plot(
        merged_df.index,
        clean_column(merged_df, colName1) - 273.15,
        linewidth=1,
        label="TS_1_1_1 (C)",
    )
    ax.plot(
        merged_df.index,
        clean_column(merged_df, colName2) - 273.15,
        linewidth=1,
        label="TS_2_1_1 (C)",
    )
    ax.plot(
        merged_df.index,
        clean_column(merged_df, colName3) - 273.15,
        linewidth=1,
        label="TS_3_1_1 (C)",
    )
    # fig, ax = plt.subplots(figsize=plot_shape)
    ax.plot(
        merged_df.index,
        clean_column(merged_df, colName4) - 273.15,
        linewidth=1,
        label="TS_4_1_1 (C)",
    )
    ax.plot(
        merged_df.index,
        clean_column(merged_df, colName5) - 273.15,
        linewidth=1,
        label="TS_5_1_1 (C)",
    )
    ax.plot(
        merged_df.index,
        clean_column(merged_df, colName6) - 273.15,
        linewidth=1,
        label="TS_6_1_1 (C)",
    )


    ax.xaxis.set_major_locator(mdates.DayLocator(interval=ticks))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d-%y"))
    plt.xticks(rotation=45, ha="right")
    ax.set_xlim(date_range)
    ax.set_ylim(0,45)
    ax.set_title(selceted_site + ": Soil temperature from all hydra probes.")
    ax.legend(loc="lower left")
    st.pyplot(fig)
    plt.close()


def plot_temperature_probe(merged_df):
    colName1 = "TS_7_1_1"
    colName2 = "TS_8_1_1"
    colName3 = "TS_9_1_1"
    fig, ax = plt.subplots(figsize=plot_shape)
    ax.plot(
        merged_df.index,
        clean_column(merged_df, colName1) - 273.15,
        linewidth=1,
        label="TS_7_1_1 (C)",
    )
    ax.plot(
        merged_df.index,
        clean_column(merged_df, colName2) - 273.15,
        linewidth=1,
        label="TS_8_1_1 (C)",
    )
    ax.plot(
        merged_df.index,
        clean_column(merged_df, colName3) - 273.15,
        linewidth=1,
        label="TS_9_1_1 (C)",
    )

    ax.xaxis.set_major_locator(mdates.DayLocator(interval=ticks))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d-%y"))
    plt.xticks(rotation=45, ha="right")
    ax.set_xlim(date_range)
    ax.set_ylim(0,45)
    ax.set_title(
        selceted_site + ": Soil temperature from (LI-7800-180) probes"
    )
    ax.legend(loc="lower left")
    st.pyplot(fig)
    plt.close()


def plot_ppdf_swin(merged_df):
    colName1 = "PPFD_1_1_1"
    colName2 = "SWIN_1_1_1"
    fig, ax = plt.subplots(figsize=plot_shape)
    ax.plot(
        merged_df.index,
        clean_column(merged_df, colName1) * 0.51,
        color="red",
        linewidth=1,
        label="PPFD (factor= 0.5100) (W/m2",
    )

    ax.xaxis.set_major_locator(mdates.DayLocator(interval=ticks))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d-%y"))
    ax.plot(
        merged_df.index,
        clean_column(merged_df, colName2),
        color="blue",
        linewidth=1,
        label="SWIN (W/m2)",
    )
    ax.set_xlim(date_range)
    ax.set_ylim(-25,1100)
    plt.xticks(rotation=45, ha="right")
    ax.set_title(selceted_site + ": Quantum sensor & Shortwave_In")
    ax.legend(loc="lower left")
    st.pyplot(fig)
    plt.close()


def plot_solar_components(merged_df):
    colName1 = "SWIN_1_1_1"
    colName2 = "SWOUT_1_1_1"
    colName3 = "LWIN_1_1_1"
    colName4 = "LWOUT_1_1_1"
    fig, ax = plt.subplots(figsize=plot_shape)
    ax.plot(
        merged_df.index,
        clean_column(merged_df, colName1),
        linewidth=1,
        label="SWIN (W/m2)",
    )
    ax.plot(
        merged_df.index,
        clean_column(merged_df, colName2),
        linewidth=1,
        label="SWOUT (W/m2)",
    )
    ax.plot(
        merged_df.index,
        clean_column(merged_df, colName3),
        linewidth=1,
        label="LWIN (W/m2)",
    )
    ax.plot(
        merged_df.index,
        clean_column(merged_df, colName4),
        linewidth=1,
        label="LWOUT (W/m2)",
    )

    ax.xaxis.set_major_locator(mdates.DayLocator(interval=ticks))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d-%y"))
    plt.xticks(rotation=45, ha="right")
    ax.set_xlim(date_range)
    ax.set_ylim(-25,1100)
    ax.set_title(
        selceted_site
        + ": "
        + colName1
        + "; "
        + colName2
        + "; "
        + colName3
        + "; "
        + colName4
    )
    ax.legend(loc="lower left")
    st.pyplot(fig)
    plt.close()


def plot_vaporpressure(merged_df):
    colName1 = "e"
    colName2 = "es"
    fig, ax = plt.subplots(figsize=plot_shape)
    ax.plot(
        merged_df.index,
        clean_column(merged_df, colName1),
        linewidth=1,
        label="vapor pressure (Pa)",
    )
    ax.plot(
        merged_df.index,
        clean_column(merged_df, colName2),
        linewidth=1,
        label="Saturation vapor pressure (Pa)",
    )

    ax.xaxis.set_major_locator(mdates.DayLocator(interval=ticks))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d-%y"))
    plt.xticks(rotation=45, ha="right")
    ax.set_title(selceted_site + ": " + colName1 + " & " + colName2)
    ax.set_xlim(date_range)
    ax.legend(loc="lower left")
    st.pyplot(fig)
    plt.close()


def plot_airpresure(merged_df):
    colName = "air_pressure"
    fig, ax = plt.subplots(figsize=plot_shape)
    ax.plot(merged_df.index, merged_df[colName], linewidth=1)

    ax.xaxis.set_major_locator(mdates.DayLocator(interval=ticks))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d-%y"))
    plt.xticks(rotation=45, ha="right")
    ax.set_xlim(date_range)
    ax.set_title(selceted_site + ": " + colName + " (Pa)")
    st.pyplot(fig)
    plt.close()


def plot_albedo(merged_df):
    try:
        colName = "ALB_1_1_1"
        fig, ax = plt.subplots(figsize=plot_shape)
        ax.plot(merged_df.index, clean_column(merged_df, colName), linewidth=1)

        ax.xaxis.set_major_locator(mdates.DayLocator(interval=ticks))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d-%y"))
        plt.xticks(rotation=45, ha="right")
        ax.set_xlim(date_range)
        ax.set_title(selceted_site + ": " + colName)
        st.pyplot(fig)
        plt.close()
    except:
        pass


def plot_precip(merged_df):
    ET_Daily = merged_df["P_RAIN_1_1_1"].resample("D").sum()
    ET_Daily = ET_Daily * 100
    colName1 = "P_RAIN_1_1_1"
    fig, ax = plt.subplots(figsize=plot_shape)
    bar_width = np.clip((0.0023 * ET_Daily.shape[0] + 0.0204), 0.025, 0.2)
    ax.bar(ET_Daily.index, ET_Daily.values, label="Liquid precip (cm)", width=bar_width)

    ax.xaxis.set_major_locator(mdates.DayLocator(interval=ticks))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d-%y"))
    plt.xticks(rotation=45, ha="right")
    ax.set_xlim(date_range)
    ax.set_ylim(0,2.54*2)
    ax.set_title(selceted_site + ": Liquid precipitation cumulative daily (cm).")
    ax.legend(loc="lower left")
    st.pyplot(fig)
    plt.close()


def plot_co2_comparision(merged_df):
    colName1 = "co2_molar_density"
    colName2 = "air_pressure"
    colName3 = "air_temperature"
    airPressure = clean_column(merged_df, colName2)
    airTemperature = clean_column(merged_df, colName3)
    CO2NumberDensity = 425.55 * airPressure / (8.314 * airTemperature) * 1 / 1000
    fig, ax = plt.subplots(figsize=plot_shape)
    ax.plot(
        merged_df.index,
        clean_column(merged_df, colName1),
        color="red",
        linewidth=1,
        label="LI-7500 CO2 molar density mmol/m3",
    )

    ax.xaxis.set_major_locator(mdates.DayLocator(interval=ticks))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d-%y"))
    ax.plot(
        merged_df.index,
        CO2NumberDensity,
        color="blue",
        linewidth=1,
        label="Estimated CO2 number density mmol/m3",
    )
    ax.set_xlim(date_range)
    plt.xticks(rotation=45, ha="right")
    ax.set_title(selceted_site + ": LI-7500 CO2 molar density & CO2 number density")
    ax.legend()
    st.pyplot(fig)
    plt.close()


def wind_rose(merged_df):
    # Create a windrose plot
    # Only daytime hours 7am to 7pm
    merged_df = merged_df[(merged_df.index.hour >= 7) & (merged_df.index.hour < 19)]    
    fig, ax = plt.subplots(subplot_kw={"projection": "windrose"})
    ax.bar(
        merged_df["wind_dir"],
        merged_df["wind_speed"],
        normed=True,
        opening=0.8,
        edgecolor="white",
        nsector=36,
        cmap=plt.cm.jet,
        bins=[0, 2, 5, 7, 10, 15, 19.99],
    )
    ax.set_title(f"Windrose: {selceted_site}")
    ax.set_legend(
        title="Windrose (m/s)",
        loc="lower left",
        bbox_to_anchor=(-0.1, -0.1),
        bbox_transform=ax.transAxes,
    )
    col1, col2, col3 = st.columns(3)
    try:
        with col1:
            st.markdown(f"**Windrose: {selceted_site} (7am - 7pm).**")
            st.pyplot(fig)
            plt.close()
    except:
        st.text("Variable not found")
        pass
    try:
        with col2:
            st.markdown(f"**Windrose: Nearest weatherstation.**")
            relative_path = update_summaries.windroses[f"{selceted_site}"]
            wind_rose_path = f"{script_path}\{relative_path}"
            wind_rose_path = wind_rose_path.replace("\\", "/")
            # st.markdown(f"Wind rose path: {wind_rose_path}")
            image = Image.open(wind_rose_path)

            st.image(image, caption=f"{selceted_site} Wind rose.")
    except:
        st.text("Variable not found")
        pass

    try:
        with col3:
            st.markdown(f"**Satellite image: {selceted_site}.**")
            relative_path = update_summaries.satellite_images[selceted_site]
            satellite_img_path = f"{script_path}/{relative_path}"
            satellite_img_path = satellite_img_path.replace("\\", "/")
            # st.markdown(f"Satellite image path: {satellite_img_path}")
            image = Image.open(satellite_img_path)

            st.image(image, caption=f"{selceted_site} satellite image.")
    except:
        st.text("Variable not found")
        pass


merged_df = pd.DataFrame()
dbPath = f"{script_path}/Data/{selceted_site}/summaries/{selceted_site}.db"
# st.write(f"{dbPath}")
merged_df = get_db(dbPath)
start_date = merged_df.index.values[0]
end_date = merged_df.index.values[-1]
limit_date = end_date - pd.Timedelta(days=days_limit)
if limit_date < start_date:
    limit_date = start_date
date_range = (limit_date, end_date)
merged_df = merged_df.sort_index(axis=1)
merged_df = merged_df[
    (merged_df.index >= date_range[0]) & (merged_df.index <= end_date)
]

date_range_placeholder.markdown(f"- **Displayed date range:** {datetime.date(merged_df.index[0])} to {datetime.date(merged_df.index[-1])}")
plt.rcParams["figure.facecolor"] = "lightcyan"
# Setting the axes background color
plt.rcParams["axes.facecolor"] = "lightcyan"

plot_RH(merged_df)

plot_co2signal(merged_df)

plot_precip(merged_df)

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

# plot_EB_components(merged_df)

plot_solar_components(merged_df)

plot_ET(merged_df)

plot_horizontal_SWC(merged_df)

plot_temperatures(merged_df)

plot_temperature_SHF(merged_df)

plot_temperature_probe(merged_df)

plot_SHF(merged_df)
