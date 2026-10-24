import sys
import math
from datetime import datetime, time
import matplotlib.dates as mdates
import matplotlib as mpl
import matplotlib.pyplot as plt
from PlotGenerator import PlotGenerator as PlotGeneratorClass
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import HistGradientBoostingClassifier
from Parameters import Parameters
from PIL import Image
import streamlit as st
from sklearn.ensemble import IsolationForest
import pytz
from db_reader import DB_Reader
import os


DAQM_columns = Parameters().DAQM_columns
sites = Parameters().sites
days_limit = 7
TICK = 3
selected_site = "NAPI"

time_zone = pytz.timezone("US/Mountain")

today = datetime.combine(datetime.today(), time(0,0,0))

today_end = today + pd.Timedelta(hours=24)

limit_date = today_end - pd.Timedelta(days=60)
date_range:tuple = (limit_date, today_end)

# Set global rcParams for date formatting and locator
mpl.rcParams['date.autoformatter.day'] = "%b-%d-%y"
# mpl.rcParams['date.daylocator.interval'] = 1  # Set interval to 1 day

PLOT_SIZE = (18, 4)


def nan_check(df, col_name):
    """Check if all column values are NaNs."""
    if col_name not in df.columns:
        return True

    if df[col_name].isnull().all():            
        return True
    else:
        return False


def remove_outliers(df, contamination=0.01):
    # Create a copy of the DataFrame
    df_clean = df.copy()
    # Initialize the IsolationForest model
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    index_reshaped = np.array(df.index).reshape(-1, 1)
    # Iterate through each column
    for column in df.columns:
        if df[column].isna().all():
            continue
        # Reshape the data for IsolationForest
        X = df[[column]]
        clf = HistGradientBoostingClassifier()
        clf.fit(index_reshaped, X) 
        # Fit the model and predict
        predictions = iso_forest.fit_predict(X)
        
        # Replace outliers with NaN
        df_clean.loc[predictions == -1, column] = np.nan
    
    return df_clean


def clean_column(df, col_name, window=24, threshold=2.5):

    try:
        if col_name not in df.columns:
            df[col_name] = np.nan
            return df[col_name]
        if col_name == "P_RAIN_1_1_1":
            return df[col_name]
        if col_name == "TS_7_1_1" or col_name == "TS_8_1_1" or col_name == "TS_9_1_1":
            df.loc[:,col_name] = df.loc[:,col_name] - 273.15
            df.loc[:,col_name] = df.loc[:,col_name] + 273.15
            return df[col_name]

        if df[col_name].isnull().all():

            df[col_name] = np.nan

            return df[col_name]


        rolling_median = df[col_name].rolling(window=window, center=True).median()

        rolling_std = df[col_name].rolling(window=window, center=True).std()

        outliers = np.abs(df[col_name] - rolling_median) > (threshold * rolling_std)

        # Create a copy of the original data

        df_cleaned = df.copy()

        # Replace outliers with NaN

        df_cleaned.loc[outliers, col_name] = np.nan

        return df_cleaned[col_name]


    except Exception as ex:
        print(ex)

        return df[col_name]


def plot_temperatures(df, date_range, selected_site_bold):

    col_name1 = "sonic_temperature"

    col_name2 = "TA_1_1_1"

    col_name3 = "TC_1_1_1"

    col_name4 = "air_temperature"


    fig, ax = plt.subplots(figsize=(18, 6))
    ax.plot(
        df.index,

        clean_column(df, col_name1) - 273.15,

        linewidth=0.5,

        label="Sonic air temperature(C)",

        marker="o",

        markersize=0.5,
    )
    ax.plot(
        df.index,

        clean_column(df, col_name2),# - 273.15,

        linewidth=1.5,

        label="Vaisala Air temperature(C)",

        linestyle="dashed",

        marker="o",

        markersize=0.5,
    )
    ax.plot(
        df.index,

        clean_column(df, col_name4) - 273.15,

        linewidth=1.5,

        label="LI-7500 temperature(C)",

        linestyle="dotted",

        marker="o",

        markersize=0.5,
    )

    try:
        ax.plot(
            df.index,

            clean_column(df, col_name3), #- 273.15,

            linewidth=0.5,

            label="IRT Canopy temperature(C)",

            marker="o",

            markersize=0.5,
        )
        ax.set_title(

            selected_site_bold + ": Sonic, Vaisala, LI-7500 & IRT Canopy temperatures."
        )

    except Exception as e:

        ax.set_title(selected_site_bold + ": Sonic, Vaisala & LI-7500 (C)")

        print(f"An error occurred: {e}")

    # ax.xaxis.set_major_locator(mdates.DayLocator(interval=tick))

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))

    ax.grid(True)

    plt.xticks(rotation=-45, ha="right")

    ax.set_xlim(date_range)

    # ax.set_ylim(-15, 40)

    ax.legend(loc="lower left")

    st.pyplot(fig)
    plt.close()


def plot_SWC(merged_df,date_range, selected_site_bold):

    col_name1 = "SWC_1_1_1"

    col_name2 = "SWC_2_1_1"

    col_name3 = "SWC_3_1_1"

    col_name4 = "SWC_4_1_1"

    col_name5 = "SWC_5_1_1"

    col_name6 = "SWC_6_1_1"


    fig, ax = plt.subplots(figsize=PLOT_SIZE)
    ax.plot(

        merged_df.index,

        clean_column(merged_df, col_name1),

        linewidth=1,

        label="SWC_1_1_1 m3/m3",

        marker="o",

        markersize=0.5,
    )
    ax.plot(

        merged_df.index,

        clean_column(merged_df, col_name2),

        linewidth=1,

        label="SWC_2_1_1 m3/m3",

        marker="o",

        markersize=0.5,
    )
    ax.plot(

        merged_df.index,

        clean_column(merged_df, col_name3),

        linewidth=1,

        label="SWC_3_1_1 m3/m3",

        marker="o",

        markersize=0.5,
    )

    ax.plot(

        merged_df.index,

        clean_column(merged_df, col_name4),

        linewidth=1,

        label="SWC_4_1_1 m3/m3",

        marker="o",

        markersize=0.5,
    )
    ax.plot(

        merged_df.index,

        clean_column(merged_df, col_name5),

        linewidth=1,

        label="SWC_5_1_1 m3/m3",

        marker="o",

        markersize=0.5,
    )
    ax.plot(

        merged_df.index,

        clean_column(merged_df, col_name6),

        linewidth=1,

        label="SWC_6_1_1 m3/m3",

        marker="o",

        markersize=0.5,
    )

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))

    plt.xticks(rotation=-45, ha="right")

    ax.set_xlim(date_range)

    ax.set_ylim(0, 1)

    ax.set_title(selected_site_bold + ": Soil water content Hydra Probes (all)")

    ax.legend(loc="lower left")

    ax.grid(True)

    st.pyplot(fig)
    plt.close()


def plot_RH(df, selected_site_bold):

    col_name1 = "RH_1_1_1"

    col_name2 = "RH"

    fig, ax = plt.subplots(figsize=PLOT_SIZE)
    ax.plot(
        df.index,

        clean_column(df, col_name1),

        linewidth=1,

        label="RH (vaisala)",

        marker="o",

        markersize=0.5,
    )
    ax.plot(
        df.index,

        clean_column(df, col_name2),

        linewidth=1,

        label="RH (gas analyzer)",

        marker="o",

        markersize=0.5,
    )

    # ax.xaxis.set_major_locator(mdates.DayLocator(interval=tick))

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))

    plt.xticks(rotation=-45, ha="right")

    ax.set_xlim(df.index[0], df.index[-1])
    ax.set_title(

        selected_site_bold

        + ": Relative humidity from Vaisala (2 meters) and gas analyzer (3-4 meters)."
    )

    ax.legend(loc="lower left")

    ax.set_ylim(0, 105)

    ax.grid(True)

    st.pyplot(fig)
    plt.close()


def plot_co2signal(df:pd.DataFrame, selected_site_bold):

    col_name = "co2_signal_strength_7500_mean"

    if nan_check(df, col_name):
        return

    fig, ax = plt.subplots(figsize=PLOT_SIZE)
    ax.plot(
        df.index, clean_column(df, col_name), linewidth=1, marker="o", markersize=0.5
    )

    # ax.xaxis.set_major_locator(mdates.DayLocator(interval=tick))

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))

    plt.xticks(rotation=-45, ha="right")

    plt.grid()

    ax.set_xlim(df.index[0], df.index[-1])

    ax.grid(True)

    ax.set_title(selected_site_bold + ": " + col_name + " (%)")

    st.pyplot(fig)
    plt.close()


def combo_temperature_shf_plot(df, date_range, selected_site_bold):


    col_name1 = "TS_1_1_1"

    col_name2 = "TS_2_1_1"

    col_name3 = "TS_3_1_1"

    col_name4 = "SHF_1_1_1"

    col_name5 = "SHF_2_1_1"

    col_name6 = "SHF_3_1_1"


    col_name7 = "TS_7_1_1"

    col_name8 = "TS_8_1_1"

    col_name9 = "TS_9_1_1"


    fig = make_subplots(specs=[[{"secondary_y": True}]])


    fig.add_trace(

        go.Scatter(

            x=df.index, y=df[col_name1], name="HydraProbe_1 temperature (C)"
        ),

        secondary_y=False,
    )

    fig.add_trace(

        go.Scatter(

            x=df.index, y=df[col_name2], name="HydraProbe_2 temperature (C)"
        ),

        secondary_y=False,
    )

    fig.add_trace(

        go.Scatter(

            x=df.index, y=df[col_name3] , name="HydraProbe_3 temperature (C)"
        ),

        secondary_y=False,
    )


    fig.add_trace(

        go.Scatter(
            x=df.index,

            y=clean_column(df, col_name7, window=48),

            name="SoilProbe_1 temperature (C)",
        ),

        secondary_y=False,
    )

    fig.add_trace(

        go.Scatter(
            x=df.index,

            y=clean_column(df, col_name8, window=48),

            name="SoilProbe_2 temperature (C)",
        ),

        secondary_y=False,
    )

    fig.add_trace(

        go.Scatter(
            x=df.index,

            y=clean_column(df, col_name9, window=48),

            name="SoilProbe_3 temperature (C)",
        ),

        secondary_y=False,
    )


    fig.add_trace(

        go.Scatter(

            x=df.index, y=clean_column(df, col_name4), name="Soil Heat Flux 1 (W/m2)"
        ),

        secondary_y=True,
    )

    fig.add_trace(

        go.Scatter(

            x=df.index, y=clean_column(df, col_name5), name="Soil Heat Flux 2 (W/m2)"
        ),

        secondary_y=True,
    )

    fig.add_trace(

        go.Scatter(

            x=df.index, y=clean_column(df, col_name6), name="Soil Heat Flux 3 (W/m2)"
        ),

        secondary_y=True,
    )


    fig.update_layout(

        title=f"{selected_site}: Comparision of Hydra probe & Soil temperature probe Vs Soil Heat Flux diurnal pattern.",

        title_x=0.15,

        xaxis=dict(range=date_range),

        xaxis_title="Date",

        legend_title="",

        legend=dict(orientation="h", yanchor="bottom", y=-0.45, xanchor="right", x=1),

        hovermode="x unified",
    )

    fig.update_yaxes(

        title_text="<b>Hydra probe & soil probe temperatures (C)</b>", secondary_y=False
    )

    fig.update_yaxes(title_text="<b>SHF (W/m2)</b>", secondary_y=True)

    fig.update_traces(mode="lines")


    st.plotly_chart(fig, width='stretch')


def plot_hydra_probe_temperatures(df, selected_site_bold):

    col_name4 = "TS_4_1_1"

    col_name5 = "TS_5_1_1"

    col_name6 = "TS_6_1_1"

        # Calculate means
    mean_ts4 = df[col_name4].mean()
    mean_ts5 = df[col_name5].mean()
    mean_ts6 = df[col_name6].mean()

    # col_name10 = "TC_1_1_1"
    if mean_ts4 > -200 or mean_ts5 > -200 or mean_ts6 > -200:
        df.loc[:,col_name4]= df[col_name4] + 273.15
        df.loc[:,col_name5]= df[col_name5] + 273.15
        df.loc[:,col_name6]= df[col_name6] + 273.15


    fig, ax = plt.subplots(figsize=PLOT_SIZE)

    ax.plot(
        df.index,

        clean_column(df, col_name4) - 273.15,

        linewidth=1,

        label="TS_4_1_1 (C)",

        marker="o",

        markersize=0.5,
    )
    ax.plot(
        df.index,

        clean_column(df, col_name5) - 273.15,

        linewidth=1,

        label="TS_5_1_1 (C)",

        marker="o",

        markersize=0.5,
    )
    ax.plot(
        df.index,

        clean_column(df, col_name6) - 273.15,

        linewidth=1,

        label="TS_6_1_1 (C)",

        marker="o",

        markersize=0.5,
    )


    # ax.xaxis.set_major_locator(mdates.DayLocator(interval=tick))

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))

    plt.xticks(rotation=-45, ha="right")

    ax.set_xlim(df.index[0], df.index[-1])

    # ax.set_ylim(0, 30)
    ax.set_title(

        f"{selected_site_bold}: Soil temperatures from hydra probes at 20, 40 & 60 cm depths."
    )

    ax.legend(loc="lower left")

    ax.grid(True)

    st.pyplot(fig)
    plt.close()


def plot_temperature_probe(df, selected_site_bold):

    col_name1 = "TS_7_1_1"

    col_name2 = "TS_8_1_1"

    col_name3 = "TS_9_1_1"

    fig, ax = plt.subplots(figsize=PLOT_SIZE)
    ax.plot(
        df.index,

        clean_column(df, col_name1, window=5) - 273.15,

        linewidth=1,

        label="TS_7_1_1 (C)",

        marker="o",

        markersize=0.5,
    )
    ax.plot(
        df.index,

        clean_column(df, col_name2, window=5) - 273.15,

        linewidth=1,

        label="TS_8_1_1 (C)",

        marker="o",

        markersize=0.5,
    )
    ax.plot(
        df.index,

        clean_column(df, col_name3, window=5) - 273.15,

        linewidth=1,

        label="TS_9_1_1 (C)",

        marker="o",

        markersize=0.5,
    )


    # ax.xaxis.set_major_locator(mdates.DayLocator(interval=tick))

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))

    plt.xticks(rotation=-45, ha="right")

    ax.set_xlim(df.index[0], df.index[-1])

    # ax.set_ylim(0, 45)

    ax.set_title(selected_site_bold + ": Soil temperature from (LI-7800-180) probes")

    ax.legend(loc="lower left")

    ax.grid(True)

    st.pyplot(fig)
    plt.close()


def plot_ppdf_swin(df, selected_site_bold):

    col_name1 = "PPFD_1_1_1"

    col_name2 = "SWIN_1_1_1"
    ppdf = clean_column(df, col_name1)*0.51
    # print(ppdf.mean())
    ppfdMean = round(ppdf.mean(),2)

    swin = clean_column(df, col_name2)
    swinMean = round(swin.mean(),2)
    ppfd_delta = round((ppfdMean/swinMean)*100,1)
    if ppfd_delta > 105 or ppfd_delta < 95:
        color = "red"
    else:
        color = "black"

    fig, ax = plt.subplots(figsize=PLOT_SIZE)
    ax.plot(
        df.index,

        ppdf,
        color="red",

        linewidth=0.5,

        label=f"PPFD (factor= 0.5100) (W/m2); Mean: {ppfdMean}",

        marker="o",

        markersize=0.5,
    )


    # ax.xaxis.set_major_locator(mdates.DayLocator(interval=tick))

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.plot(
        df.index,

        swin,

        color="blue",

        linewidth=0.5,

        label=f"SWIN (W/m2); Mean: {swinMean}",

        marker="o",

        markersize=0.5,
    )

    ax.set_xlim(df.index[0], df.index[-1])

    # ax.set_ylim(-25, 1100)

    plt.xticks(rotation=-45, ha="right")

    ax.set_title(selected_site_bold + f": Quantum sensor & Shortwave_In; (PPFD is {round((ppfdMean/swinMean)*100,1)}% of SWIN)", color=color)

    ax.legend(loc="lower left")

    ax.grid(True)

    st.pyplot(fig)
    plt.close()


def plot_solar_components(df, selected_site_bold):

    col_name1 = "SWIN_1_1_1"

    col_name2 = "SWOUT_1_1_1"

    col_name3 = "LWIN_1_1_1"

    col_name4 = "LWOUT_1_1_1"

    fig, ax = plt.subplots(figsize=PLOT_SIZE)
    ax.plot(
        df.index,

        clean_column(df, col_name1),

        linewidth=1,

        label="SWIN (W/m2)",

        marker="o",

        markersize=0.5,
    )
    ax.plot(
        df.index,

        clean_column(df, col_name2),

        linewidth=1,

        label="SWOUT (W/m2)",

        marker="o",

        markersize=0.5,
    )
    ax.plot(
        df.index,

        clean_column(df, col_name3),

        linewidth=1,

        label="LWIN (W/m2)",

        marker="o",

        markersize=0.5,
    )
    ax.plot(
        df.index,

        clean_column(df, col_name4),

        linewidth=1,

        label="LWOUT (W/m2)",

        marker="o",

        markersize=0.5,
    )


    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))

    plt.xticks(rotation=-45, ha="right")

    ax.set_xlim(df.index[0], df.index[-1])

    ax.set_ylim(-25, 1100)
    ax.set_title(

        selected_site_bold

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

    if fig:
        st.pyplot(fig)
    plt.close()


def plot_precip(df, selected_site_bold):

    rain_daily = df["P_RAIN_1_1_1"].resample("D").sum()

    fig, ax = plt.subplots(figsize=PLOT_SIZE)

    bar_width = np.clip((0.0023 * rain_daily.shape[0] + 0.0204), 0.025, 0.2)

    ax.bar(rain_daily.index, rain_daily.values, label="Liquid precip (mm)", width=bar_width)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))

    plt.xticks(rotation=-45, ha="right")

    ax.set_xlim(df.index[0], df.index[-1])

    ax.set_ylim(0, 20)

    ax.set_title(selected_site_bold + ": Liquid precipitation cumulative daily (mm).")

    ax.legend(loc="lower left")

    st.pyplot(fig)
    plt.close()


def energy_balance_ratio(df, date_range, selected_site_bold):
    required_columns = ["RN_1_1_1", "daytime", "LE", "SHF_1_1_1", "H"]
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        print(f"Missing columns: {missing_columns}")
        return
    try:
        condition = df['daytime'] ==0
        df.loc[condition,'daytime'] = np.nan
        df = df.dropna(subset = ['daytime'])

        RN = df["RN_1_1_1"].to_numpy()

        LE = df["LE"].to_numpy()

        G = df["SHF_1_1_1"].to_numpy()

        H = df["H"].to_numpy()
        closure = ((LE + H) / (RN - G))

        df.loc[:,"EBR"] = closure.to_numpy()

        df.loc[:,'EBR'] = clean_column(df, "EBR", window=15, threshold=2)
        df_median = df.resample('D').median()

        fig, ax = plt.subplots(figsize=PLOT_SIZE)

        ax.plot(df_median.index, df_median['EBR'], linewidth=1)

        ax.plot(df_median.index, np.ones(df_median.shape[0]), linewidth=1, color="gray")

        ax.plot(df_median.index, np.zeros(df_median.shape[0])+0.4, linewidth=2, color="red", label='Check values below this line')

        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))

        plt.xticks(rotation=-45, ha="right")

        ax.set_xlim(date_range)
        ax.set_title(

            selected_site_bold

            + ": "

            + "Hourly Energy Balance Ratio (LE+H)/(RN-G) (De-spiked, only daytime; daily median value.)"
        )

        ax.set_ylim(0, 0.9)
        plt.legend()
        st.pyplot(fig)
        plt.close()

    except Exception as e:
        print(f"An error occurred: {e}")
        pass


def main():

    global selected_site

    if os.path.exists("last_compiled.txt"):
        with open("last_compiled.txt", "r") as f:
            compilation_time = f.read().strip()
    else:
        compilation_time = "Unknown"

    st.set_page_config(

        page_title="Daily EC data monitor",
        layout="wide",
        page_icon="🗼",
        initial_sidebar_state="expanded",
    )
           # background-color: #f0f0f0;  /* Change to your desired color */
    st.markdown(
        """

        <style>

        .section1 {

            color: #000000

            padding: 1em;

            border-radius: 25px;

            border-thickness:2px;

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

        unsafe_allow_html=True,
    )

    with st.sidebar:

        st.title("Navigation")

        global selected_site
        selected_site = st.radio("Select EC site:", sites)

        selected_site_bold = f"$\\mathbf{{{selected_site}}}$"

        st.markdown("---")

        st.header("Sensor separation calculator:")

        separation = st.number_input(

            "Sonic-Gas analyzer distance (cm):",

            min_value=0.0,

            max_value=25.0,

            value=21.59,

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

        st.markdown(f"Northward (Y) : **{round(sensor_west,2)} cm**.")

        st.markdown(f"Eastward (X) : **{round(sensor_east, 2)} cm**.")

        st.markdown("---")

    head_col1, head_col2 = st.columns([8, 1])

    with head_col1:

        st.markdown(
            """

        <div class='section1' style='padding:0 1.5em;'>    

        <h2> Upper CO River Basin Commision (UCRBC) Project</h2>

        <span style='font-size:1.5em;'> EC data monitoring from all sites.</span>
        <p style='font-size:1.5em;'> <a href='https://accurate-una-ashish-personal-a1bba9d0.koyeb.app/'> Alternative site.</a></p>
        </div>
        """,

            unsafe_allow_html=True,
        )

    with head_col2:

        image_self = Image.open(r"Data/SatelliteImage/self.jpg")

        st.image(image=image_self, caption="Site created by Ashish Masih.", width=75)

        email_link = (

            "###### [**@ Contact:**](mailto:amasih2@unl.edu?subject=Subject&body=Message)"
        )

        st.markdown(email_link, unsafe_allow_html=True)


    st.markdown(f"## {selected_site_bold} EC tower data:-")

    st.markdown("- This is raw and uncorrected data and should be used only to review sensor's functionality.")

    date_range_placeholder = st.empty()

    date_range_placeholder.text("")

    st.text("")
    st.text("")
    
    merged_df = pd.DataFrame()

    db_reader = DB_Reader()
    merged_df = db_reader.get_db(str(selected_site))
    lastIndex = merged_df.shape[0]-1

    if merged_df.empty:
        st.markdown(f"## No data found in the database of {selected_site} site.")
        st.stop()

    custom_range = st.slider(
        "Select date range",
        min_value=0,
        max_value=lastIndex,
        step=144,
        value=(0,lastIndex)
    )

    st.divider()
    st.write("#### Chart min/max date range:", merged_df.index[custom_range[0]], merged_df.index[custom_range[1]])
    merged_df = merged_df.iloc[custom_range[0]: custom_range[1]+1]

    start_date = merged_df.index[0]

    data_max_date = merged_df.index[-1]
    date_range = (start_date, data_max_date)

    last_valid_index = merged_df.dropna(how='all').index[-1]
    first_valid_index = merged_df.dropna(how='all').index[0]

    date_range_placeholder.markdown(
        f"- #### **Start time:** {first_valid_index} (MST); **End time:** {last_valid_index} (MST); Site updated at: {compilation_time} (CDT).")

    plt.rcParams["figure.facecolor"] = "lightcyan"

    plt.rcParams["axes.facecolor"] = "lightcyan"
    
    plot_gen = PlotGeneratorClass(merged_df, PLOT_SIZE, selected_site)
    plot_gen.plot_t_et_vpd()
    plot_gen.plot_evap_fraction()

    col1, col2 = st.columns((8, 8.9))

    with col1:   
        plot_gen.plot_fetch()

    with col2:

        fig2 = plot_gen.plot_closure()

        if fig2 is not None:
            st.pyplot(fig2)
            plt.close()
    
    st.divider()

    plot_gen.wind_rose()

    plot_precip(merged_df, selected_site_bold)
    plot_co2signal(merged_df, selected_site_bold)
    plot_RH(merged_df, selected_site_bold)
    plot_gen.plot_rads_ratios()
    plot_ppdf_swin(merged_df, selected_site_bold)
    plot_temperatures(merged_df, date_range, selected_site_bold)
    plot_hydra_probe_temperatures(merged_df, selected_site_bold)
    plot_SWC(merged_df, date_range, selected_site_bold)
    combo_temperature_shf_plot(merged_df, date_range, selected_site_bold)
    plot_solar_components(merged_df, selected_site_bold)
    st.divider()

    energy_balance_ratio(merged_df, date_range, selected_site_bold)

    st.image(image=f"./Data/Offset/{selected_site}_offset.png", caption="Timing difference (s) between GHG and Sonic", width='stretch')
    st.text("Timing difference between GHG and SF3. If the system is working normally, the values should be 0 or 1. But when SF3 doesn't communicate with the installed GPS reciever to get the satellite time, the values are -1. If no time stamp is found at all (i.e. corrupt file, which rarely happens) the values are -2.")

    plot_gen.sf3_json_stats()


if __name__ == "__main__":
    # all_site_data = DB_Reader()
    main()
