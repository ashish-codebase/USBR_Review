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

selceted_site = ""

@st.cache_data
def get_parquet(parquet_file):
    return pd.read_parquet(parquet_file)


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


plot_shape = (18, 3)
update_summaries.main()
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
    days_limit = st.number_input(
        "Enter days to show:", step=7, min_value=1, value=21
    )

ticks = np.clip(int(days_limit/10),1,15)

script_path = os.getcwd()
all_files = glob.glob(script_path + "**/**/*", recursive=True)
st.markdown(
    "<h1 style='text-align: center; text-decoration:underline;'> Upper CO River Basin Commision (UCRBC) Project</h2>",
    unsafe_allow_html=True,
)
st.markdown("### Daily EC summary data from all online towers in CO, WY, NM & NE")

st.text("")
st.text("")


def clean_column(df, colName):
    if colName == "P_RAIN_1_1_1":
        return df
    Q1 = df[colName].quantile(0.01)
    Q3 = df[colName].quantile(0.99)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[colName] = df[colName].where(
        (df[colName] >= lower_bound) & (df[colName] <= upper_bound), np.nan
    )
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
        merged_df.index, merged_df[colName1] - 273.15, linewidth=1, label="Sonic (C)"
    )
    ax.plot(merged_df.index, merged_df[colName2] - 273.15, linewidth=1, label="Air (C)")
    try:
        ax.plot(merged_df.index, merged_df[colName4], linewidth=1, label="TCNR4 (C)")
    except:
        pass
    try:
        ax.plot(
            merged_df.index,
            merged_df[colName3] - 273.15,
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
    ax.legend(loc="lower left")
    st.pyplot(fig)
    plt.close()


def plot_horizontal_SWC(merged_df):
    colName1 = "SWC_1_1_1"
    colName2 = "SWC_2_1_1"
    colName3 = "SWC_3_1_1"
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

    ax.xaxis.set_major_locator(mdates.DayLocator(interval=ticks))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d-%y"))

    plt.xticks(rotation=45, ha="right")
    ax.set_xlim(date_range)
    ax.set_title(selceted_site + ": Soil water content for soil heat flux ~5 cm.")
    ax.legend(loc="lower left")
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
    huber.fit(X, y)
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
    linear.fit(X, y)
    linear_slope = linear.coef_[0]
    linear_intercept = linear.intercept_
    linear_yPredicted = linear.predict(X)
    linear_r2 = r2_score(y, linear_yPredicted)

    linear_regression = "Linear Regression:"
    linear_regression += f"\n    Intercept: {float(linear_intercept):.4f}"
    linear_regression += f"\n    Slope: {float(linear_slope):.4f}"
    linear_regression += f"\n    R2: {float(linear_r2):.4f}"

    # Generate predictions
    X_plot = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    X_plot_scaled = scaler.transform(X_plot)
    y_pred_huber = huber.predict(X_plot_scaled)
    y_pred_linear = linear.predict(X_plot_scaled)

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
    st.pyplot(fig)
    plt.close()


def plot_co2signal(merged_df):
    colName = "co2_signal_strength_7500_mean"
    fig, ax = plt.subplots(figsize=plot_shape)
    ax.plot(merged_df.index, clean_column(merged_df, colName), linewidth=1)

    ax.xaxis.set_major_locator(mdates.DayLocator(interval=ticks))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d-%y"))
    plt.xticks(rotation=45, ha="right")
    ax.set_xlim(date_range)
    ax.set_ylim(0, 110)
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
    ax.set_title(selceted_site + ": Soil Heat flux 1,2 & 3 (W/m2)")
    ax.legend(loc="lower left")
    st.pyplot(fig)
    plt.close()


def plot_closure(merged_df):
    fig, ax = plt.subplots(figsize=(9, 9))
    try:
        components = pd.DataFrame()
        Yvals = clean_column(merged_df,"RN_1_1_1") - clean_column(merged_df,"SHF_3_1_1")
        Xvals = clean_column(merged_df,"LE") + clean_column(merged_df,"H")
        components["RN_G"] = Yvals
        components["LE_H"] = Xvals
        eb_ratio_array = (components["LE_H"].values/components["RN_G"].values)
        eb_ratio_array = np.clip(eb_ratio_array,-0.5, 1.5)
        components["EB_Array"] = eb_ratio_array
        eb_ratio = round((components["LE_H"].sum()/components["RN_G"].sum()),4)
        components = components[
            (components.index.hour >= 7) & (components.index.hour < 19)
        ]
        components = components.dropna()
        maxval = max(components.RN_G.max(), components.LE_H.max())
        slope, intercept, r_value, p_value, std_err = linregress(
            components.RN_G, components.LE_H
        )
        x_fit = np.linspace(
            components.RN_G.min(), components.RN_G.max(), components.shape[0]
        )
        y_fit = slope * x_fit + intercept

        scatter = ax.scatter(
            components.RN_G.values,
            components.LE_H.values,
            c=components["EB_Array"],
            cmap='plasma',
            label="Energy balance closure",
            s=2,
        )
        plt.colorbar(scatter, ax=ax, label='EBC fraction')
        ax.plot(x_fit, y_fit, color="red", label="Regression line")
        # Create the equation string
        equation = f"y = {slope:.2f}x + {intercept:.2f}\n$R^2$={round(r_value,4)}"
        # Annotate the plot with the equation
        ax.text(
            0.1,
            0.9,
            transform=ax.transAxes,
            s=equation,
            color="red",
            fontsize=12,
            ha="left",
        )
        eq_string = r"$EBR = \frac {\sum{LE+H}}{\sum{RN-G}} =$"
        ax.text(
            0.1,
            0.7,
            transform=ax.transAxes,
            s=f"Energy Balance Ratio:-\n    {eq_string} {eb_ratio}",
            color="brown",
            fontsize=12,
            ha="left",
        )
        # ax.set_xlim(-50,maxval*1.1)
        # ax.set_ylim(-50, maxval*1.1)
        ax.set_xlim(-50, 800)
        ax.set_ylim(-50, 800)
    except:
        pass
    ax.set_title(selceted_site + ": " + "Energy balance closure (7am - 7pm)")
    ax.set_xlabel("RN-G (W/m2)")
    ax.set_ylabel("LE + H (W/m2)")
    ax.legend(loc="upper right")

    return fig


def plot_soil_profile_temperature(merged_df):
    colName1 = "TS_4_1_1"
    colName2 = "TS_5_1_1"
    colName3 = "TS_6_1_1"
    fig, ax = plt.subplots(figsize=plot_shape)
    ax.plot(
        merged_df.index,
        clean_column(merged_df, colName1) - 273.15,
        linewidth=1,
        label="TS_4_1_1 (C)",
    )
    ax.plot(
        merged_df.index,
        clean_column(merged_df, colName2) - 273.15,
        linewidth=1,
        label="TS_5_1_1 (C)",
    )
    ax.plot(
        merged_df.index,
        clean_column(merged_df, colName3) - 273.15,
        linewidth=1,
        label="TS_6_1_1 (C)",
    )


    ax.xaxis.set_major_locator(mdates.DayLocator(interval=ticks))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d-%y"))
    plt.xticks(rotation=45, ha="right")
    ax.set_xlim(date_range)
    ax.set_title(
        selceted_site
        + ": Soil temperature from hydra probe; Soil profile measurement 20, 40 & 60 cm"
    )
    ax.legend(loc="lower left")
    st.pyplot(fig)
    plt.close()


def plot_temperature_SHF(merged_df):
    colName1 = "TS_1_1_1"
    colName2 = "TS_2_1_1"
    colName3 = "TS_3_1_1"
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


    ax.xaxis.set_major_locator(mdates.DayLocator(interval=ticks))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d-%y"))
    plt.xticks(rotation=45, ha="right")
    ax.set_xlim(date_range)
    ax.set_title(selceted_site + ": Soil temperature from hydra probe; SHF measurement")
    ax.legend(loc="lower left")
    st.pyplot(fig)
    plt.close()


def plot_profile_SWC(merged_df):
    colName1 = "SWC_4_1_1"
    colName2 = "SWC_5_1_1"
    colName3 = "SWC_6_1_1"
    fig, ax = plt.subplots(figsize=plot_shape)
    ax.plot(
        merged_df.index,
        clean_column(merged_df, colName1),
        linewidth=1,
        label="SWC_4_1_1 m3/m3",
    )
    ax.plot(
        merged_df.index,
        clean_column(merged_df, colName2),
        linewidth=1,
        label="SWC_5_1_1 m3/m3",
    )
    ax.plot(
        merged_df.index,
        clean_column(merged_df, colName3),
        linewidth=1,
        label="SWC_6_1_1 m3/m3",
    )


    ax.xaxis.set_major_locator(mdates.DayLocator(interval=ticks))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d-%y"))
    plt.xticks(rotation=45, ha="right")
    ax.set_xlim(date_range)
    ax.set_title(
        selceted_site + ": Soil water content for soil moisture profile 20, 40 & 60 cm"
    )
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
    ax.set_title(
        selceted_site + ": Soil temperature from temperature probes; SHF measurement"
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


def plot_EB_components(merged_df):
    colName1 = "RN_1_1_1"
    colName2 = "SHF_1_1_1"
    colName3 = "H"
    colName4 = "LE"
    fig, ax = plt.subplots(figsize=plot_shape)
    ax.plot(
        merged_df.index,
        clean_column(merged_df, colName1),
        linewidth=1,
        label="RN (W/m2)",
    )
    ax.plot(
        merged_df.index,
        clean_column(merged_df, colName2),
        linewidth=1,
        label="G (W/m2)",
    )
    ax.plot(
        merged_df.index,
        clean_column(merged_df, colName3),
        linewidth=1,
        label="H (W/m2)",
    )
    ax.plot(
        merged_df.index,
        clean_column(merged_df, colName4),
        linewidth=1,
        label="LE (W/m2)",
    )

    ax.xaxis.set_major_locator(mdates.DayLocator(interval=ticks))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d-%y"))
    plt.xticks(rotation=45, ha="right")
    ax.set_xlim(date_range)
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
    ax.set_title(selceted_site + ": Liquid precipitation cumulative daily (cm).")
    ax.legend(loc="lower left")
    st.pyplot(fig)
    plt.close()


def plot_co2_comparision(merged_df):
    # url = ("https://seer.sct.embrapa.br/index.php/agrometeoros/article/view/26527/14623")
    # st.markdown(f"[Conversion of PPFD umol/m2/s to W/m2 for 400 - 700 nm wavelength (eq. 38). ]({url})")
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
    ax.set_title(selceted_site + ": Quantum sensor & CO2 number density")
    ax.legend()
    st.pyplot(fig)
    plt.close()


def wind_rose(merged_df):
    # Create a windrose plot
    fig, ax = plt.subplots(subplot_kw={'projection': 'windrose'})
    ax.bar(merged_df['wind_dir'], merged_df['wind_speed'], normed=True, opening=0.8, edgecolor='white', nsector=36,cmap=plt.cm.jet, bins=[0,2,5,7,10,15,19.99])
    ax.set_legend()
    ax.set_title(f'Windrose: {selceted_site}')
    col1, col2 = st.columns(2)
    try:
        with col1:
            st.pyplot(fig)
            plt.close()
    except:
        st.text("Variable not found")
        pass
    try:
        with col2:
            st.image(update_summaries.windroses[f"{selceted_site}"], caption=selceted_site)
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

# plot_airpresure(merged_df)

# outer area color
plt.rcParams["figure.facecolor"] = "whitesmoke"
# Setting the axes background color
plt.rcParams["axes.facecolor"] = "whitesmoke"


# plot_albedo(merged_df)

# plot_vaporpressure(merged_df)

plot_EB_components(merged_df)

plot_solar_components(merged_df)

plot_ET(merged_df)

plot_temperatures(merged_df)

plot_horizontal_SWC(merged_df)

plot_SHF(merged_df)

plot_profile_SWC(merged_df)


plot_temperature_SHF(merged_df)

plot_temperature_probe(merged_df)

plot_soil_profile_temperature(merged_df)
