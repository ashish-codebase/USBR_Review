import pandas as pd
import glob
import os
import datetime
import sqlite3

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
    "Holbrook",
]


def get_dataframe(filePath):
    tempPath = os.path.basename(filePath)
    year_int = int(tempPath.split("-")[0])
    if year_int < datetime.datetime.today().year:
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


def main():
    todaysDate = datetime.datetime.today().date()
    main_path = r"D:\OneDrive - University of Nebraska-Lincoln\UNL\All EC Tower Data"
    script_path = os.getcwd()
    for selected_site in sites:
        summary_sqlite = f"{script_path}/Data/{selected_site}/summaries/summary.db"
        mod_time_readable = datetime.datetime(month=1, day=1, year=1900).date()
        if os.path.exists(summary_sqlite):
            mod_time = os.path.getmtime(summary_sqlite)
            mod_time_readable = datetime.datetime.fromtimestamp(mod_time).date()

        if mod_time_readable < todaysDate:

            file_pattern = f"{main_path}/{selected_site}/summaries/*Summary.txt"
            csv_files = glob.glob(file_pattern, recursive=True)
            if len(csv_files) == 0:
                continue
            counter = 0
            for filePath in csv_files:
                if os.path.getsize(filePath) == 0:
                    continue
                df = get_dataframe(filePath)
                if df.shape[0] < 3:
                    continue
                if counter == 0:
                    merged_df = df
                    counter = 1
                else:
                    merged_df = pd.concat([merged_df, df], axis=0)

            merged_df = merged_df[
                (
                    merged_df.index
                    >= datetime.datetime(datetime.datetime.today().year, 1, 1)
                )
            ]
            merged_df.sort_index(inplace=True)
            merged_df.drop_duplicates(inplace=True)
            start_time = datetime.datetime.now()
            conn = sqlite3.connect(summary_sqlite)
            merged_df.to_sql("summary", con=conn, if_exists="replace", index=True)
            # Close the connection
            conn.close()

            delta = datetime.datetime.now() - start_time
            performance = f"Time difference for {selected_site}: {delta}"

            with open("performance.txt", "a") as file:
                print(performance)
                # Write a line to the file
                file.write(performance + "\n")


if __name__ == "__main__":
    main()
