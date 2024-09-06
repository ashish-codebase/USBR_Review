import pandas as pd
import glob
import os
import datetime
import sqlite3

mod_time_readable = datetime.datetime(month=1, day=1, year=1900).date()
today_date = datetime.datetime.today().date()
date_range = (mod_time_readable, today_date)
columnlist = [
    "air_pressure",
    "air_temperature",
    "ALB_1_1_1",
    "co2_signal_strength_7500_mean",
    "co2_molar_density",
    "date",
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
    "VPD",
    "water_vapor_density",
    "wind_dir",
    "wind_speed"
]
sites = [
    "LaPlata",
    "NAPI",
    "Olathe",
    "Baggs",
    "Cora",
    "Cortez",
    "Gunnison",
    "Boulder",
    "HUC_12",
    "GrantNE",
    "Sutherland_Beans",
    "Holbrook"
]
windroses = {
    "LaPlata":"https://bil-ec-towers.vercel.app/WindRose_LaPlata.png",
    "NAPI":"https://bil-ec-towers.vercel.app/WindRose_NAPI.png",
    "Olathe":"https://bil-ec-towers.vercel.app/WindRose_Uncompahgre.png",
    "Baggs":"https://bil-ec-towers.vercel.app/WindRose_Baggs.png",
    "Cora":"https://bil-ec-towers.vercel.app/WindRose_Cora.png",
    "Cortez":"https://bil-ec-towers.vercel.app/WindRose_Cortez.png",
    "Gunnison":"https://bil-ec-towers.vercel.app/WindRose_Gunnison.png",
    "Boulder":"https://bil-ec-towers.vercel.app/WindRose_Cora.png",
    "HUC_12":"",
    "GrantNE":"",
    "Sutherland_Beans":"",
    "Holbrook":""
}

def get_dataframe(filePath):
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
    df.drop(columns=["date", "time"], inplace=True)
    df.sort_index(inplace=True)
    # merged_df = merged_df[(merged_df.index >= datetime.datetime(datetime.datetime.today().year,1,1 ))]
    return df

def read_db(selected_site):
    todaysDate = datetime.datetime.today().date()
    main_path = r"D:\OneDrive - University of Nebraska-Lincoln\UNL\All EC Tower Data"
    script_path = os.getcwd()
    db_path = f"{script_path}/Data/{selected_site}/summaries/{selected_site}.db"
    print(f"DB path string {db_path}")    
    if os.path.exists(db_path):
        mod_time = os.path.getmtime(db_path)
        mod_time_readable = datetime.datetime.fromtimestamp(mod_time).date()

    if mod_time_readable < todaysDate:

        file_pattern = f"{main_path}/{selected_site}/summaries/*Summary.txt"
        print(f"Summary files search path: {file_pattern}")
        print("")
        print("")
        print("")
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
        start_time = datetime.datetime.now()
        conn = sqlite3.connect(db_path)
        merged_df.to_sql(name="summary", con=conn, if_exists="replace", index=True)
        # Close the connection
        conn.close()
        merged_df = pd.DataFrame()
        
def main():
    for selected_site in sites:
        read_db(selected_site)

if __name__ == "__main__":
    main()
