import sqlite3
import glob
import os
import datetime
import pandas as pd
import numpy as np


today_date = datetime.datetime.today().date()
# date_range = (mod_time_readable, today_date)
columnlist = [
    "air_pressure",
    "air_temperature",
    "ALB_1_1_1",
    "bowen_ratio",
    "co2_signal_strength_7500_mean",
    "co2_molar_density",
    "date",
    "daytime",
    "e",
    "es",
    "ET",
    "H",
    "L",
    "LE",
    "LWIN_1_1_1",
    "LWOUT_1_1_1",
    "P_RAIN_1_1_1",
    "PPFD_1_1_1",
    "RH",
    "RH_1_1_1",
    "RN_1_1_1",
    "SHF_1_1_1",
    "SHF_2_1_1",
    "SHF_3_1_1",
    "sonic_temperature",
    "SWC_1_1_1",
    "SWC_2_1_1",
    "SWC_3_1_1",
    "SWC_4_1_1",
    "SWC_5_1_1",
    "SWC_6_1_1",
    "SWIN_1_1_1",
    "SWOUT_1_1_1",
    "TA_1_1_1",
    "TC_1_1_1",
    "TCNR4_C_1_1_1",
    "time",
    "TS_1_1_1",
    "TS_2_1_1",
    "TS_3_1_1",
    "TS_4_1_1",
    "TS_5_1_1",
    "TS_6_1_1",
    "TS_7_1_1",
    "TS_8_1_1",
    "TS_9_1_1",
    "u*",
    "VPD",
    "water_vapor_density",
    "wind_dir",
    "wind_speed"
]

sites = [
    "Olathe",
    "Cora",
    "Boulder",
    "NAPI",
    "LaPlata",
    "Cortez",
    "Baggs",
    "FtBridger",
    "Gunnison",
    # "HUC_12",
    # "GrantNE",
    # "Sutherland_Beans",
    # "Holbrook"
]
windroses = {
    "LaPlata":r"Data/WindRose/WindRose_LaPlata.png",
    "NAPI":r"Data/WindRose/WindRose_NAPI.png",
    "Olathe":r"Data/WindRose/WindRose_Olathe.png",
    "Baggs":r"Data/WindRose/WindRose_Baggs.png",
    "Cora":r"Data/WindRose/WindRose_Cora.png",
    "Cortez":r"Data/WindRose/WindRose_Cortez.png",
    "Gunnison":r"Data/WindRose/WindRose_Gunnison.png",
    "Boulder":r"Data/WindRose/WindRose_Boulder.png",
    "HUC_12":r"Data/WindRose/WindRose_HUC12.png",
    "GrantNE":r"Data/WindRose/WindRose_GrantNE.png",
    "Sutherland_Beans":r"Data/WindRose/WindRose_Sutherland.png",
    "Holbrook":r"Data/WindRose/WindRose_Holbrook.png",
    "FtBridger":r"Data/WindRose/WindRose_FtBridger.png"
}
satellite_images = {
    "LaPlata":r"Data/SatelliteImage/LaPlata.jpg",
    "NAPI":r"Data/SatelliteImage/NAPI.jpg",
    "Olathe":r"Data/SatelliteImage/Olathe.jpg",
    "Baggs":r"Data/SatelliteImage/Baggs.jpg",
    "Cora":r"Data/SatelliteImage/Cora.jpg",
    "Cortez":r"Data/SatelliteImage/Cortez.jpg",
    "Gunnison":r"Data/SatelliteImage/Gunnison.jpg",
    "Boulder":r"Data/SatelliteImage/Boulder.jpg",
    "HUC_12":r"Data/SatelliteImage/HUC12.jpg",
    "GrantNE":r"Data/SatelliteImage/Grant.jpg",
    "Sutherland_Beans":r"Data/SatelliteImage/Sutherland.jpg",
    "Holbrook":r"Data/SatelliteImage/Holbrook.jpg",
    "FtBridger":r"Data/SatelliteImage/FtBridger.jpg"
}

def fill_nans(df):
    for col_name in df.columns:
        if df[col_name].isnull().all():
            # print("all values NAN in ", col_name)
            df[col_name]  = np.nan
    return df

def rn_time_lag(df: pd.DataFrame):
    df['RN_1_1_1'] = df['SWIN_1_1_1'] - df['SWOUT_1_1_1'] + df['LWIN_1_1_1'] - df['LWOUT_1_1_1'].shift(-2)
    df['LWOUT_1_1_1'] = df['LWOUT_1_1_1'].shift(-2)
    return df

def get_dataframe(filePath):
    """Get the dataframe from the selected daily summary file for the pre-selected columns only."""
    tempPath = os.path.basename(filePath)
    year_int = int(tempPath.split("-")[0])
    if year_int < datetime.datetime.today().year-1:
        return pd.DataFrame()
    # print(filePath)
    header = pd.read_csv(filePath, nrows=0, sep="\t").columns.tolist()
    # Filter the desired columns to include only those present in the file
    actual_columns = [col for col in columnlist if col in header]
    df = pd.read_csv(filePath, sep="\t", skiprows=[1], usecols=actual_columns)
    df["DateTime"] = pd.to_datetime(df["date"] + " " + df["time"])
    df.set_index("DateTime", inplace=True)
    df.sort_index(inplace=True)
    for column in columnlist:
        if not column in df.columns:
            df[column]=np.nan
    df.drop(columns=["date", "time"], inplace=True)
    df = fill_nans(df)
    return df

def read_db(selected_site):
    # selected_site = sites[10]
    todaysDate = datetime.datetime.today().date()
    data_path1 = r"D:\OneDrive - University of Nebraska-Lincoln\UNL\All EC Tower Data"
    data_path2 = r"C:\Users\ashish\OneDrive - University of Nebraska-Lincoln\UNL\All EC Tower Data"
    if os.path.exists(data_path1):
        main_path = data_path1
    else:
        main_path = data_path2
    script_path = os.getcwd()
    db_path = f"{script_path}/Data/{selected_site}/summaries/{selected_site}.db"

    mod_time_readable = datetime.datetime(month=1, day=1, year=1900).date()
    if os.path.exists(db_path):
        mod_time = os.path.getmtime(db_path)
        mod_time_readable = datetime.datetime.fromtimestamp(mod_time).date()


    if mod_time_readable <= todaysDate:
        file_pattern = f"{main_path}/{selected_site}/summaries/*Summary.txt"
        summary_files = glob.glob(file_pattern, recursive=True)
        if len(summary_files) == 0:
            return
        counter = 0
        for filePath in summary_files:
            if os.path.getsize(filePath) == 0:
                continue
            df = get_dataframe(filePath)
            if df.shape[0] < 1:
                continue
            if counter == 0:
                merged_df = df
                counter = 1
            else:
                merged_df = pd.concat([merged_df, df], axis=0)

        merged_df.sort_index(inplace=True)
        merged_df.drop_duplicates(inplace=True)
        merged_df = fill_nans(merged_df)
        # merged_df =rn_time_lag(merged_df)
        conn = sqlite3.connect(db_path)
        merged_df.to_sql(name="summary", con=conn, if_exists="replace", index=True)
        print(f"Writing DB {selected_site} at {datetime.datetime.now().strftime('%H:%M:%S')}")
        # Close the connection
        conn.close()
        merged_df = merged_df.iloc[0:0]

def main():
    print("\n")
    for selected_site in sites:
        print(f"Reading {selected_site} at: {datetime.datetime.now().strftime('%H:%M:%S')}")
        read_db(selected_site)
    # print("\n")

if __name__ == "__main__":
    main()
