import logging
from typing import Tuple, Optional
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
import pandas as pd
import streamlit as st
from scipy.stats import linregress
from sklearn.linear_model import HuberRegressor, LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from Parameters import Parameters
from PIL import Image
from windrose import WindroseAxes


class PlotGenerator:
    """
    A class for generating various environmental data plots using Streamlit and Matplotlib.

    This class provides methods to create different types of plots for environmental monitoring data,
    including evapotranspiration, vapor pressure deficit, energy balance closure, wind roses, and more.

    Attributes:
        df (pd.DataFrame): The input DataFrame containing environmental data.
        plot_size (Tuple[int, int]): The size of the plots as (width, height).
        selected_site (str): The name of the site being analyzed.
        selected_site_bold (str): The site name formatted in bold LaTeX for plot titles.
        ticks (int): Number of ticks on the x-axis for date formatting.
    """

    def __init__(self, df: pd.DataFrame, plot_size: Tuple[int, int], site_name: str,
                 clean_window: int = 24, clean_threshold: float = 2.5) -> None:
        """
        Initialize the PlotGenerator with data and configuration.

        Args:
            df (pd.DataFrame): The DataFrame containing environmental data with datetime index.
            plot_size (Tuple[int, int]): The dimensions for the plots (width, height).
            site_name (str): The name of the monitoring site.
            clean_window (int): Window size for outlier detection in data cleaning. Default is 24.
            clean_threshold (float): Threshold multiplier for outlier detection. Default is 2.5.

        Raises:
            ValueError: If df is not a pandas DataFrame or if plot_size is invalid.
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("df must be a pandas DataFrame")
        if not isinstance(plot_size, tuple) or len(plot_size) != 2 or not all(isinstance(x, int) and x > 0 for x in plot_size):
            raise ValueError("plot_size must be a tuple of two positive integers")
        if not isinstance(site_name, str) or not site_name.strip():
            raise ValueError("site_name must be a non-empty string")

        self.df = df
        self.plot_size = plot_size
        self.selected_site = site_name.strip()
        self.selected_site_bold = f"$\\mathbf{{{self.selected_site}}}$"
        self.ticks = 3  # Default number of ticks for date formatting
        self.clean_window = clean_window
        self.clean_threshold = clean_threshold
        self.logger = logging.getLogger(__name__)

    def t_string(self)->str:
        """\
        Returns timestamp string for error logging.
        """
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    def is_all_nans(self, col_name: str) -> bool:
        """
        Check if a column contains only NaN values.

        Args:
            col_name (str): The name of the column to check.

        Returns:
            bool: True if the column doesn't exist or contains only NaN values, False otherwise.
        """
        if col_name not in self.df.columns:
            return True
        return self.df[col_name].isnull().all()


    def clean_column(self, col_name: str) -> pd.Series:
        """
        Clean a column by removing outliers using rolling statistics.

        Args:
            col_name (str): Name of the column to clean.

        Returns:
            pd.Series: The cleaned column data.
        """
        df = self.df.copy()
        try:
            if col_name not in df.columns:
                df[col_name] = np.nan
                return df[col_name]
            if col_name == "P_RAIN_1_1_1":
                return df[col_name]
            if col_name == "TS_7_1_1" or col_name == "TS_8_1_1" or col_name == "TS_9_1_1":
                df[col_name] =df[col_name] - 273.15
                df[col_name] = np.where((df[col_name] >= -15) & (df[col_name] <= 75), df[col_name], np.nan)
                df[col_name] =df[col_name] + 273.15
                return df[col_name]

            if df[col_name].isnull().all():

                df[col_name] = np.nan

                return df[col_name]


            rolling_median = df[col_name].rolling(window=self.clean_window, center=True).median()
    
            rolling_std = df[col_name].rolling(window=self.clean_window, center=True).std()
    
            outliers = np.abs(df[col_name] - rolling_median) > (self.clean_threshold * rolling_std)

            # Create a copy of the original data

            df_cleaned = df.copy()

            # Replace outliers with NaN

            df_cleaned.loc[outliers, col_name] = np.nan

            return df_cleaned[col_name]


        except Exception as ex:
            self.logger.error(f"{self.t_string()}: Error cleaning column {col_name}: {ex}")
            return df[col_name]


    def plot_t_et_vpd(self) -> None:
        """
        Plot daily evapotranspiration (ET), maximum temperature, and maximum vapor pressure deficit (VPD).

        This method creates a bar plot showing:
        - Daily ET (summed from half hourly values, divided by 2)
        - Daily maximum VPD (in kPa)
        - Daily maximum air temperature (in Celsius)

        The plot is displayed using Streamlit.

        Returns:
            None
        """

        if self.is_all_nans("ET"):
            return

        df_copy = self.df.copy()
        df_copy.loc[df_copy["ET"]<0,"ET"] = np.nan
        ET_Daily = df_copy["ET"].resample("D").sum()
        ET_Daily = ET_Daily / 2
        VPD = df_copy['VPD'].resample('D').max()*0.001
        TMAX = df_copy['air_temperature'].resample('D').max() - 273.15
        fig, ax1 = plt.subplots(figsize=self.plot_size)
        plt.subplots_adjust(right=0.75) 
        ax2 = ax1.twinx()
        ax3 = ax1.twinx()

        bar_width = 0.1

        ax1.bar(ET_Daily.index, ET_Daily.to_numpy(), color='blue', width=bar_width)        
        ax2.bar(VPD.index+pd.Timedelta(hours=4), VPD.to_numpy(), color='red', width=bar_width)
        ax3.bar(TMAX.index-pd.Timedelta(hours=4), TMAX.to_numpy(), color='green', width=bar_width)

        # ax1.xaxis.set_major_locator(mdates.DayLocator(interval=self.ticks))

        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))


        ax1.set_xlim(df_copy.index.min(), df_copy.index.max())
        ax1.set_ylim(bottom=0, top=None)
        ax2.set_ylim(bottom=0, top=None)
        ax3.set_ylim(bottom=0, top=None)
        ax1.tick_params(axis='x', labelrotation=-45)
        ax1.tick_params(axis='y',colors='blue')
        ax2.tick_params(axis='y', colors='red', direction='out', length=8, width=1)
        ax3.tick_params(axis='y', colors='green', direction='out', length=8, width=1)
        ax1.set_title(self.selected_site_bold + r": Daily ET, T$_{max}$ & VPD$_{max}$.")
        ax1.set_ylabel("Daily ET (mm)", color='blue')
        ax2.set_ylabel("Daily max. VPD (Kpa)",loc='center', labelpad=10, color='red')
        ax3.set_ylabel("Daily max. T air (C)", loc='center', labelpad=10, color='green')
        ax3.spines['right'].set_position(('outward', 50))
        st.pyplot(fig)
        plt.close()

    
    def plot_closure(self):
        df_copy = self.df.copy()
        required_columns = ["RN_1_1_1", "SHF_3_1_1", "LE","H","daytime","u*","RN_G","LE_H"]
        missing_columns = [col for col in required_columns if col not in df_copy.columns]

        if missing_columns:
            print(f"Missing columns: {missing_columns}")
            return

        fig, ax = plt.subplots(figsize=(9, 8))
        ax.plot(

            (-100, 900),

            (-100, 900),

            linewidth=2,

            color="gray",

            linestyle="dashdot",

            label="1:1 line",
        )

        try:

            components = pd.DataFrame()
            components.index = df_copy.index


            components["RN_G"] = self.clean_column( "RN_1_1_1") - self.clean_column( "SHF_1_1_1")

            components["LE_H"] = self.clean_column("LE") + self.clean_column("H")

            components["daytime"] = self.clean_column("daytime")

            components["u*"] = self.clean_column( "u*")

            eb_ratio_array = components["LE_H"].to_numpy() / components["RN_G"].to_numpy()

            eb_ratio_array = np.clip(eb_ratio_array, -0.5, 1.5)

            components["EB_Ratio"] = eb_ratio_array
            components = components[components['EB_Ratio']>0]


            eb_ratio = round((components["LE_H"].sum() / components["RN_G"].sum()), 4)

            components = components[(components["daytime"] > 0)]
            
            components = components[(components["LE_H"] > 0)]

            components = components[(components["RN_G"] > 0)]

            components = components[(components["u*"] >= 0.15)]

            components = components.dropna()

            slope, intercept, rvalue, _,_ = linregress(components.RN_G, components.LE_H)
            rvalue = np.asarray(rvalue)

            x_fit = np.linspace(

                np.min(components.RN_G), np.max(components.RN_G), components.shape[0]
            )

            y_fit = slope * x_fit + intercept

            scatter = ax.scatter(

                x = components['RN_G'].to_numpy(),

                y = components['LE_H'].to_numpy(),

                c=components.index.dayofyear,
                cmap="brg",

                label="Energy balance closure",

                s=2,
            )

            plt.colorbar(scatter, ax=ax, label="Day of Year")

            ax.plot(x_fit, y_fit, color="red", label="Regression line")

            # Create the equation string
            rsq_str = f"{rvalue**2:.4f}"
            equation = f"Linear regression:\n    y = {slope:.2f}x + {intercept:.2f}\n    $R^2$={rsq_str}"

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

            zero_x = components.RN_G.to_numpy()[:, np.newaxis]

            zero_y = components.LE_H.to_numpy()[:, np.newaxis]

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

            ax.set_xlim(-20, 900)
            ax.set_ylim(-20, 900)
        except Exception as ex:
            self.logger.error(f"{self.t_string()}: Error in energy balance closure plotting: {ex}")
            pass
        ax.set_title(

            self.selected_site_bold

            + ": "

            + "Energy balance closure (daytime hours; Friction velocity u*>=0.15 m/s)"
        )

        ax.set_xlabel("RN-G (W/m2)")

        ax.set_ylabel("LE + H (W/m2)")

        ax.legend(loc="upper right")

        return fig


    def wind_rose(self):
        required_columns = ["wind_dir", "daytime", "wind_speed"]
        missing_columns = [col for col in required_columns if col not in self.df.columns]

        if missing_columns:
            self.logger.warning(f"{self.t_string()}: Missing columns for wind rose plotting: {missing_columns}")
            return

        if self.df["wind_dir"].isnull().all():
            return
        if self.df["daytime"].isnull().all():
            return
        if self.df["wind_speed"].isnull().all():
            return

        df = self.df[(self.df["daytime"] > 0)]

        fig, ax = plt.subplots(subplot_kw={"projection": "windrose"})

        ax.bar(

            df["wind_dir"],

            df["wind_speed"],

            normed=True,

            opening=0.8,

            edgecolor="white",

            nsector=36,

            cmap=plt.get_cmap("gist_ncar"),

            bins=[0, 2, 5, 7, 10, 15, 19.99],
        )

        ax.set_title(f"Windrose: {self.selected_site_bold}")

        ax.legend(

            title="Windrose (m/s)",

            loc="lower left",

            bbox_to_anchor=(-0.1, -0.1),

            bbox_transform=ax.transAxes,
        )

        col1, col2, col3 = st.columns(3)

        try:

            with col1:

                st.markdown(f"**Windrose: {self.selected_site_bold} (daytime hours).**")

                st.pyplot(fig)
                plt.close()

        except Exception as e:
            self.logger.error(f"{self.t_string()}: Error displaying wind rose for site {self.selected_site}: {e}")
            st.text("Variable not found")
            pass

        try:

            with col2:

                string1 = "###### Windrose"

                string2 = ": Nearest weatherstation."

                st.markdown(f"{string1}{string2}")

                relative_path = Parameters.windroses[f"{self.selected_site}"]

                wind_rose_path = f"./{relative_path}"

                wind_rose_path = wind_rose_path.replace("\\", "/")

                image = Image.open(wind_rose_path)


                st.image(image, caption="")

        except Exception as e:
            self.logger.error(f"{self.t_string()}: Error displaying nearest weather station wind rose: {e}")
            st.text("Variable not found")
            pass


        try:

            with col3:

                st.markdown(f"**Satellite image: {self.selected_site_bold}.**")

                if self.selected_site:
                    relative_path = Parameters.satellite_images[self.selected_site]
                else:
                    relative_path = None

                satellite_img_path = f"./{relative_path}"
                satellite_img_path = satellite_img_path.replace("\\", "/")

                image = Image.open(satellite_img_path)


                st.image(image, caption="")

        except Exception as e:
            self.logger.error(f"{self.t_string()}: Error displaying satellite image: {e}")
            st.text("Variable not found")
            pass

    def plot_evap_fraction(self):
        df_copy = self.df.copy()
        LE = self.clean_column("LE")
        H = self.clean_column("H")
        EF = LE / (LE + H)
        df_copy['evap_fraction'] = EF
        df_copy = df_copy.resample('3h').median()
        df_copy.loc[df_copy["daytime"] == 0, 'evap_fraction']= np.nan
        df_copy.loc[df_copy["SWIN_1_1_1"] <= 25, 'evap_fraction']= np.nan
        col_name = "evap_fraction"

        fig, ax = plt.subplots(figsize=self.plot_size)
        df_copy['0.5_line'] = 0.5
        if df_copy['evap_fraction'].min() < 0.5:
            ax.plot(df_copy.index, df_copy['0.5_line'], linewidth=1.5, color="red", label="Investigate: EF values too low.")

        ax.plot(df_copy.index, df_copy['evap_fraction'], linewidth=0.5, color="blue", label=col_name, marker='o', markersize=2)
        # ax.xaxis.set_major_locator(mdates.DayLocator(interval=self.ticks))

        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))

        plt.xticks(rotation=-45, ha="right")

        ax.grid(True)
        ax.legend()
        ax.set_xlim(self.df.index.min(), self.df.index.max())

        ax.set_ylim(-0.5, 2)
        ax.set_title(

            self.selected_site_bold + ": " + "Evaporative Fraction" + " (Daytime values; SWIN >50 W/m$^2$; de-spiked) (EF = LE / (LE + H))"
        )

        st.pyplot(fig)
        plt.close()

    def plot_ebr(self):
        df_copy = self.df[["LE", "H", "RN_1_1_1", "SHF_1_1_1"]].copy()
        df_copy = df_copy.resample('D').sum()
        LE_H = df_copy["LE"] + df_copy["H"]
        RN_G = df_copy["RN_1_1_1"] - df_copy["SHF_1_1_1"]

        df_copy['EBR'] = LE_H/RN_G
        df_copy = df_copy.dropna()
        ebr_min = 0.25
        fig, ax = plt.subplots(figsize=self.plot_size)
        df_copy['min_line'] = ebr_min
        ax.plot(df_copy.index, df_copy['min_line'], linewidth=1.5, color="red", label="Investigate: EBR values too low.")

        ax.plot(df_copy.index, df_copy['EBR'], linewidth=0.5, color="blue", label="EBR: Daily", marker='o', markersize=2)
        # ax.xaxis.set_major_locator(mdates.DayLocator(interval=self.ticks))

        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))

        plt.xticks(rotation=-45, ha="right")

        ax.grid(True)
        ax.legend()
        ax.set_xlim(df_copy.index.min(), df_copy.index.max())

        ax.set_ylim(-1, 1.5)
        ax.set_title(

            self.selected_site_bold + ": " + "EBR" + " (Daytime values; SWIN >50 W/m2; de-spiked)"
        )

        st.pyplot(fig)
        plt.close()


    def plot_fetch(self):
        try:
            df = self.df.copy()
            df['wind_speed'] = df['wind_speed'].where(df['daytime'] == 1, np.nan)
            df['wind_speed'] = df['wind_speed'].where(df['u*'] > 0.15, np.nan)
            df.dropna(subset=['wind_dir', 'x_90%', 'x_70%', 'x_50%', 'x_30%','wind_speed'], inplace=True)
            wind_direction_deg = df['wind_dir']
            distance_90 = df['x_90%']
            distance_70 = df['x_70%']
            distance_50 = df['x_50%']
            distance_30 = df['x_30%']
            scaler = MinMaxScaler()
            df['wind_speed_normalized'] = scaler.fit_transform (df[['wind_speed']])
            wind_speed = df['wind_speed_normalized']
            # Convert degrees to radians for polar plot
            wind_direction_rad = np.deg2rad(wind_direction_deg)
            fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(6,6))
            # Scatter plot
            ax.scatter(wind_direction_rad, distance_90, s=wind_speed*750, alpha=0.045, label='90%',color='red')
            ax.scatter(wind_direction_rad, distance_70, s=wind_speed*750, alpha=0.045, label='70%',color='green')
            ax.scatter(wind_direction_rad, distance_50, s=wind_speed*750, alpha=0.045, label='50%',color='blue')
            ax.scatter(wind_direction_rad, distance_30, s=wind_speed*750, alpha=0.045, label='30%',color='orange')
            ax.set_theta_zero_location('N')  # Zero at north
            ax.set_theta_direction(-1)  # Clockwise rotation
            ax.set_rlabel_position(225)  # Move radial labels away
            ax.set_ylim(0, max(df['x_90%'].quantile(0.85),350))  # Set radial limits from 0 to 10
            ax.legend()
            plt.title('Fetch distance, direction and its frequency')
            st.pyplot(fig)
            plt.close()
            st.write("Fetch distance (m): Scale shown on South-West line; Circle size: Scales with wind speed;")
            st.write("Color transparency: Scales with fetch's distance & direction frequency; ")
            st.write("Data filtered for daytime and friction velocity (u*) > 0.15 m/s;")
        except:
            pass

    def plot_rads_ratios(self):
        df = self.df.copy()
        df['LWIN_1_1_1'] = df['LWIN_1_1_1'].where(df['daytime'] == 1, np.nan)
        df['LWOUT_1_1_1'] = df['LWOUT_1_1_1'].where(df['daytime'] == 1, np.nan)
        df['SWIN_1_1_1'] = df['SWIN_1_1_1'].where(df['daytime'] == 1, np.nan)
        df['SWOUT_1_1_1'] = df['SWOUT_1_1_1'].where(df['daytime'] == 1, np.nan)
        df['SW_Delta'] = df['SWIN_1_1_1'] - df['SWOUT_1_1_1']
        df['LW_Delta'] = df['LWOUT_1_1_1'] - df['LWIN_1_1_1']

        

        df.dropna(subset=['SWIN_1_1_1', 'SWOUT_1_1_1', 'LWIN_1_1_1', 'LWOUT_1_1_1', 'LW_Delta', 'SW_Delta'], inplace=True)

        LW_Delta_median = df['LW_Delta']#.resample('D').median()
        SW_Delta_median = df['SW_Delta']#.resample('D').median()

        LW_mean_value = round(LW_Delta_median.mean(),2)
        SW_mean_value = round(SW_Delta_median.mean(),2)

        fig, ax = plt.subplots(figsize=self.plot_size)

        ax.plot(LW_Delta_median.index, LW_Delta_median, linewidth=0.5, color="blue", label=f'LW Delta: {LW_mean_value}')
        ax.plot(SW_Delta_median.index, SW_Delta_median, linewidth=0.5, color="green", label=f'SW Delta: {SW_mean_value}')

        # ax.xaxis.set_major_locator(mdates.DayLocator(interval=self.ticks))

        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))

        plt.xticks(rotation=-45, ha="right")

        ax.grid(True)
        ax.legend(loc = 'upper right')
        ax.set_xlim(self.df.index.min(), self.df.index.max())

        ax.set_ylim(-100, 800)
        ax.set_title(

            self.selected_site_bold + ": SWIN - SWOUT; LWOUT-LWIN (W/$m^2$)" 
        )

        st.pyplot(fig)
        plt.close()

    def sf3_json_stats(self):
        
        # Color function for column A
        def color_col_a(val):
            if val < 20:
                return 'background-color: lightgreen'
            elif val < 40:
                return 'background-color: yellow'
            else:
                return 'background-color: salmon'

        # Color function for column B
        def color_col_b(val):
            if val < 10:
                return 'background-color: lightblue'
            elif val < 30:
                return 'background-color: orange'
            else:
                return 'background-color: pink'

        # Color function for column C
        def color_col_c(val):
            if val < 250:
                return 'background-color: lavender'
            else:
                return 'background-color: lightgray'
            
        # Color function for Name column 
        def color_col_M(val):
            if val == 'text 3':
                return 'background-color: red'
            else:
                return 'background-color: lightgreen'

        # Sample dataframe
        data = {
            'DateTime': pd.date_range('2025-09-01 13:00:00', periods=5),
            'USB free space': [10, 25, 30, 45, 50],
            'Offset': [5, 15, 25, 35, 45],
            'GPS detected': [100, 200, 300, 400, 500],
            'D': [10, 25, 30, 45, 50],
            'E': [5, 15, 25, 35, 45],
            'F': [100, 200, 300, 400, 500],
            'G': [10, 25, 30, 45, 50],
            'H': [5, 15, 25, 35, 45],
            'I': [100, 200, 300, 400, 500],
            'J': [10, 25, 30, 45, 50],
            'K': [5, 15, 25, 35, 45],
            'L': [100, 200, 300, 400, 500],
            'M': ['text 1', 'text 2', 'text 3', 'text 4', 'text 5'],
            'n': [10, 25, 30, 45, 50],
            'o': [5, 15, 25, 35, 45],
            'p': [100, 200, 300, 400, 500],
            'Q': [10, 25, 30, 45, 50],
            'R': [5, 15, 25, 35, 45],
            'S': [100, 200, 300, 400, 500],
            'T': [10, 25, 30, 45, 50],
            'U': [5, 15, 25, 35, 45],
            'V': [100, 200, 300, 400, 500],
            'W': [10, 25, 30, 45, 50],
            'X': [5, 15, 25, 35, 45],
            'Y': [100, 200, 300, 400, 500],
            'Z': ['text 1', 'text 2', 'text 3', 'text 4', 'text 5'],
            'A1': [10, 25, 30, 45, 50],
            'B1': [5, 15, 25, 35, 45],
            'C1': [100, 200, 300, 400, 500],
            'D1': [10, 25, 30, 45, 50],
            'E1': [5, 15, 25, 35, 45],
            'F1': [100, 200, 300, 400, 500],
            'G1': [10, 25, 30, 45, 50],
            'H1': [5, 15, 25, 35, 45],
            'I1': [100, 200, 300, 400, 500],
            'J1': [10, 25, 30, 45, 50],
            'K1': [5, 15, 25, 35, 45],
            'L1': [100, 200, 300, 400, 500],
            'M1': ['text 1', 'text 2', 'text 3', 'text 4', 'text 5'],
            'n1': [10, 25, 30, 45, 50],
            'o1': [5, 15, 25, 35, 45],
            'p1': [100, 200, 300, 400, 500],
            'Q1': [10, 25, 30, 45, 50],
            'R1': [5, 15, 25, 35, 45],
            'S1': [100, 200, 300, 400, 500],
            'T1': [10, 25, 30, 45, 50],
            'U1': [5, 15, 25, 35, 45],
            'V1': [100, 200, 300, 400, 500],
            'W1': [10, 25, 30, 45, 50],
            'X1': [5, 15, 25, 35, 45],
            'Y1': [100, 200, 300, 400, 500],
            'Z1': ['text 1', 'text 2', 'text 3', 'text 4', 'text 5'],                
                    
                    
                            }
        df = pd.DataFrame(data)
        df = df.sort_values('DateTime', ascending=False).reset_index(drop=True)

        # Apply styling per column
        styled_df = df.style.map(color_col_a, subset=['USB free space'])\
                            .map(color_col_b, subset=['Offset'])\
                            .map(color_col_c, subset=['GPS detected'])\
                            .map(color_col_M, subset = ['M'])
                            

        # Display in Streamlit
        st.divider()
        st.markdown("### Stats from SF3 JSON file (daily values):-")
        st.markdown("Mode is computed if value is of string type; Median is computed if the value is a number.")
        st.dataframe(styled_df)
        st.divider()
