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

    def __init__(self, df, PLOT_SIZE, site_name):
        self.df = df
        self.PLOT_SIZE = PLOT_SIZE
        self.selected_site = site_name
        self.selected_site_bold =  f"$\\mathbf{{{site_name}}}$"
        self.ticks = 3


    def nan_check(self, col_name):
        """Check if a column is all NaNs."""
        if col_name not in self.df.columns:
            return True

        if self.df[col_name].isnull().all():
            return True
        else:
            return False


    def clean_column(self, col_name):
        window=24
        threshold=2.5
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


    def plot_et(self):

        global date_range
        """Display bar plot of daily values (from hourly summed values.)"""

        if self.nan_check("ET"):
            return

        self.df.loc[self.df["ET"]<0,"ET"] = np.nan
        ET_Daily = self.df["ET"].resample("D").sum()

        ET_Daily = ET_Daily / 2

        fig, ax = plt.subplots(figsize=self.PLOT_SIZE)

        bar_width = np.clip((0.0023 * ET_Daily.shape[0] + 0.0204), 0.025, 0.2)

        ax.bar(

            ET_Daily.index,

            ET_Daily.values,

            label="Cumulative daily ET (mm).",

            width=bar_width,
        )


        ax.xaxis.set_major_locator(mdates.DayLocator(interval=self.ticks))

        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d-%y"))

        plt.xticks(rotation=45, ha="right")

        ax.set_xlim(self.df.index.min(), self.df.index.max())
        ax.set_ylim(0, 10)
        ax.set_title(self.selected_site_bold + ": Daily cumulative ET (mm).")

        ax.legend(loc="lower left")

        st.pyplot(fig)
        plt.close()

    def plot_RH_regression(self):
        merged_df = self.df.copy()

        required_columns = ["RH", "RH_1_1_1"]
        missing_columns = [col for col in required_columns if col not in merged_df.columns]

        if missing_columns:
            print(f"Missing columns: {missing_columns}")
            return 
        
        if merged_df["RH"].isnull().all():
            return
        if merged_df["RH_1_1_1"].isnull().all():
            return

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
        ax.plot(

            (0, 100),

            (0, 100),

            linewidth=2,

            color="gray",

            linestyle="dashdot",

            label="1:1 line",
        )

        ax.scatter(X, y, s=1, label="Data points")

        ax.plot(X, huber_predicted_y, color="brown", linewidth=2, label="Huber Regression")

        ax.plot(X, linear_predicted_y, color="blue", linewidth=2, label="Linear Regression")

        # ax.plot((0,100), (0,100), color="gray", linewidth=2, label="1:1 Line")

        ax.set_xlabel("Relative humidity from Vaisala (%)")

        ax.set_ylabel("Relative humidity from LI-7500 (%)")
        ax.set_title(

            self.selected_site_bold

            + ": "

            + "RH comparisoin Vaisala (X-axis) Vs Gas analyzer RH (Y-axis)"
        )

        ax.text(0.05, 0.72, f"{linear_regression}", color="blue", transform=ax.transAxes)

        ax.text(0.05, 0.58, f"{huber_regression}", color="brown", transform=ax.transAxes)
        ax.text(

            0.12,

            0.05,

            "Regression equation is calculated using 'Huber-Regresion'\n"

            + "to ignore larger outliers (e.g. explore Cora).",

            transform=ax.transAxes,
        )

        ax.set_ylim(-5, 115)

        ax.set_xlim(-5, 115)

        ax.legend()

        ax.grid(True)

        return fig
    
    def plot_closure(self):
        df = self.df.copy()
        required_columns = ["RN_1_1_1", "SHF_3_1_1", "LE","H","daytime","u*","RN_G","LE_H"]
        missing_columns = [col for col in required_columns if col not in df.columns]

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

            components["RN_G"] = self.clean_column( "RN_1_1_1") - self.clean_column( "SHF_1_1_1")

            components["LE_H"] = self.clean_column("LE") + self.clean_column("H")

            # components["RN_G"] = Yvals

            # components["LE_H"] = Xvals

            components["daytime"] = self.clean_column("daytime")

            components["u*"] = self.clean_column( "u*")

            eb_ratio_array = components["LE_H"].to_numpy() / components["RN_G"].to_numpy()

            eb_ratio_array = np.clip(eb_ratio_array, -0.5, 1.5)

            components["EB_Ratio"] = eb_ratio_array
            components = components[components['EB_Ratio']>0]


            eb_ratio = round((components["LE_H"].sum() / components["RN_G"].sum()), 4)

            components = components[(components["daytime"] > 0)]

            components = components[(components["u*"] >= 0.15)]

            components = components.dropna()

            slope, intercept, rvalue, _,_ = linregress(components.RN_G, components.LE_H)
            rvalue = np.asarray(rvalue)

            x_fit = np.linspace(

                components.RN_G.min(), components.RN_G.max(), components.shape[0]
            )

            y_fit = slope * x_fit + intercept

            scatter = ax.scatter(

                x = components['RN_G'].to_numpy(),

                y = components['LE_H'].to_numpy(),

                c=components["EB_Ratio"].to_numpy(),
                cmap="plasma",

                label="Energy balance closure",

                s=2,
            )

            plt.colorbar(scatter, ax=ax, label="EBR (unitless)")

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

            # ax.set_xlim(-50,maxval*1.1)

            # ax.set_ylim(-50, maxval*1.1)

            ax.set_xlim(-100, 900)

            # ax.set_ylim(-100, 900)

        except Exception as ex:
            print(ex)
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
            print(f"Missing columns: {missing_columns}")
            return

        if self.df["wind_dir"].isnull().all():
            return
        if self.df["daytime"].isnull().all():
            return
        if self.df["wind_speed"].isnull().all():
            return

        # Create a windrose plot

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

            st.text("Variable not found")
            print(f"An error occurred: {e}")
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

            st.text("Variable not found")
            print(f"An error occurred: {e}")
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

                # st.markdown(f"Satellite image path: {satellite_img_path}")

                image = Image.open(satellite_img_path)


                st.image(image, caption="")

        except Exception as e:

            st.text("Variable not found")
            print(f"An error occurred: {e}")
            pass

    def plot_evap_fraction(self):
        df_copy = self.df.copy()
        LE = self.clean_column("LE")
        H = self.clean_column("H")
        EF = LE / (LE + H)
        df_copy['evap_fraction'] = EF
        df_copy.drop(['date', 'time'], axis=1, inplace=True)
        df_copy = df_copy.resample('3h').median()
        df_copy.loc[df_copy["daytime"] == 0, 'evap_fraction']= np.nan
        df_copy.loc[df_copy["SWIN_1_1_1"] <= 25, 'evap_fraction']= np.nan
        col_name = "evap_fraction"

        fig, ax = plt.subplots(figsize=self.PLOT_SIZE)
        df_copy['zero_line'] = 0
        df_copy['0.5_line'] = 0.5
        # ax.plot(df_copy.index, df_copy['zero_line'], linewidth=1, color="black", label="Zero line")
        if df_copy['evap_fraction'].min() < 0.5:
            ax.plot(df_copy.index, df_copy['0.5_line'], linewidth=0.5, color="red", label="Investigate: EF values too low.")

        ax.plot(df_copy.index, df_copy['evap_fraction'], linewidth=0.5, color="blue", label=col_name, marker='o', markersize=2)
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=self.ticks))

        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d-%y"))

        plt.xticks(rotation=45, ha="right")

        ax.grid(True)
        ax.legend()
        ax.set_xlim(self.df.index.min(), self.df.index.max())

        ax.set_ylim(-2, 2)
        ax.set_title(

            self.selected_site_bold + ": " + "Evaporative Fraction" + " (Daytime values; SWIN >50 W/m2; de-spiked)"
        )

        st.pyplot(fig)
        plt.close()


    def plot_fetch(self):
        # Example data
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
        ax.set_rlabel_position(135)  # Move radial labels away
        ax.set_ylim(0, max(df['x_90%'].quantile(0.85),350))  # Set radial limits from 0 to 10
        ax.legend()
        plt.title('Fetch distance, direction and its frequency')
        st.pyplot(fig)
        plt.close()
        st.write("Fetch distance (m): Scale shown on SW line; Circle size: Scales with wind speed;")
        st.write("Color transparency: Scales with fetch's distance & direction frequency; ")
        st.write("Data filtered for daytime and friction velocity (u*) > 0.15 m/s;")

    def plot_rads_ratios(self):
        # Example data
        df = self.df.copy()
        df['SWIN_1_1_1'] = df['SWIN_1_1_1'].where(df['daytime'] == 1, np.nan)
        df['SWOUT_1_1_1'] = df['SWOUT_1_1_1'].where(df['daytime'] == 1, np.nan)
        df['albedo'] = df['SWOUT_1_1_1'] / df['SWIN_1_1_1']
        df['albedo'] = df['albedo'].where(df['daytime'] == 1, np.nan)
        df['LWIN_1_1_1'] = df['LWIN_1_1_1'].where(df['daytime'] == 1, np.nan)
        df['LWOUT_1_1_1'] = df['LWOUT_1_1_1'].where(df['daytime'] == 1, np.nan)

        df['LW_ratio'] =  df['LWOUT_1_1_1']/df['LWIN_1_1_1']

        

        df.dropna(subset=['SWIN_1_1_1', 'SWOUT_1_1_1', 'LWIN_1_1_1', 'LWOUT_1_1_1', 'albedo', 'LW_ratio'], inplace=True)

        LW_ratio_median = df['LW_ratio'].resample('D').median()
        LW_ratio_mean_value = round(LW_ratio_median.mean(),2)
        albedo_median = df['albedo'].resample('D').median()
        albedo_mean_value = round(albedo_median.mean(),2)

        fig, ax = plt.subplots(figsize=self.PLOT_SIZE)

        ax.plot(LW_ratio_median.index, LW_ratio_median, linewidth=0.5, color="blue", label=f'LW Ratio mean: {LW_ratio_mean_value}', marker='o', markersize=2)
        ax.plot(albedo_median.index, albedo_median, linewidth=0.5, color="green", label=f'Albedo mean: {albedo_mean_value}', marker='o', markersize=2)

        ax.xaxis.set_major_locator(mdates.DayLocator(interval=self.ticks))

        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d-%y"))

        plt.xticks(rotation=45, ha="right")

        ax.grid(True)
        ax.legend()
        ax.set_xlim(self.df.index.min(), self.df.index.max())

        ax.set_ylim(0, 2)
        ax.set_title(

            self.selected_site_bold + ": Albedo (SWout/SWin) & LW Ratio (LWout/LWin)" 
        )

        st.pyplot(fig)
        plt.close()
