import glob
import os

import sqlite3
import time
from datetime import datetime
import shutil

# import matplotlib.pyplot as plt

# import matplotlib.dates as mdates
import pandas as pd

import numpy as np

from tqdm import tqdm


class DataValidation:


    def __init__(self) -> None:

        self.db_path = ".\\Data\\ValidationData.db"

        if self.is_database_empty():

            self.create_new_database()

        else:

            self.ValidationData = pd.read_sql_query("SELECT * FROM ValidationData", self.db_connection)


    def is_database_empty(self) -> bool:

        with sqlite3.connect(self.db_path) as conn:

            cursor = conn.cursor()

            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")

            tables = cursor.fetchall()

            return len(tables) == 0
        

    def create_new_database(self):

        self.db_connection = sqlite3.connect(self.db_path)

        self.ValidationData = pd.DataFrame(columns=['DateTime', 'Logged'])


        self.ValidationData.to_sql("ValidationData", self.db_connection, if_exists="replace", index=False)

        self.db_connection.close()


class GenerateData:
    """
    Creates a database for each site and stores the data from the DAQM files.\n
    """

    columnlist = [

        'air_pressure',

        'air_temperature',

        'ALB_1_1_1',

        'DATE',

        'LWIN_1_1_1',

        'LWOUT_1_1_1',

        'P_RAIN_1_1_1',

        'PPFD_1_1_1',

        'RH_1_1_1',

        'RN_1_1_1',

        'SHF_1_1_1',

        'SHF_2_1_1',

        'SHF_3_1_1',

        'SWC_1_1_1',

        'SWC_2_1_1',

        'SWC_3_1_1',

        'SWC_4_1_1',

        'SWC_5_1_1',

        'SWC_6_1_1',

        'SWIN_1_1_1',

        'SWOUT_1_1_1',

        'TA_1_1_1',

        'TC_1_1_1',

        'TCNR4_C_1_1_1',

        'TIME',

        'TS_1_1_1',

        'TS_2_1_1',

        'TS_3_1_1',

        'TS_4_1_1',

        'TS_5_1_1',

        'TS_6_1_1',

        'TS_7_1_1',

        'TS_8_1_1',

        'TS_9_1_1',

        'VPD',

        ]

    sites = [

        "Baggs",

        "Boulder",

        "Cora",

        "Cortez",

        "Gunnison",

        "LaPlata",

        "NAPI",

        "Olathe",

        "FtBridger",

        "Farson",
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


    # def __init__(self)->None:

    #     self.execute()


    def execute(self):

        merged_df = pd.DataFrame()

        for site in tqdm(self.sites):

            current_dir = os.path.dirname(os.path.abspath(__file__))

            db_path = os.path.join(current_dir, 'Data', f"{site}.db")

            if os.path.exists(db_path):

                modification_time = os.path.getmtime(db_path)

                current_time = time.time()

                time_difference_hours = (current_time - modification_time) / 3600

                if time_difference_hours < 2:
                    pass

                    continue


            db_connection = sqlite3.connect(db_path)

            daqm_path1 = r"C:\Users\ashish\OneDrive - University of Nebraska-Lincoln\UNL\All EC Tower Data"

            daqm_path2 = r"D:\OneDrive - University of Nebraska-Lincoln\UNL\All EC Tower Data"


            if os.path.isdir(daqm_path1):

                file_pattern = f"{daqm_path1}\\{site}\\daqm\\*daqm.zip"

            elif os.path.isdir(daqm_path2):

                file_pattern = f"{daqm_path2}\\{site}\\daqm\\*daqm.zip"

            else:

                break


            daqm_paths = glob.glob(file_pattern, recursive=True)

            counter = 0

            for filePath in daqm_paths:

                daqm_data = self.get_dataframe(filePath)

                if daqm_data.empty:
                    continue

                if daqm_data.shape[0]<30:

                    continue

                dh_hryl = daqm_data.resample('30min').mean()
                dh_hryl['P_RAIN_1_1_1'] = daqm_data['P_RAIN_1_1_1'].resample("30min").sum()
                if counter==0:

                    merged_df = dh_hryl

                    counter=1

                else:

                    merged_df = pd.concat([merged_df,dh_hryl], axis=0)


            merged_df.to_sql(site, db_connection, if_exists="replace", index=True)

            db_connection.close()
            shutil.copy(db_path, r'D:\OneDrive - University of Nebraska-Lincoln\UNL\StreamlitProjects\FastHTML_Project\static\data')


    def get_dataframe(self, daqm_path):

        df = pd.read_csv(daqm_path, sep="\t", skiprows=[1], encoding='utf-8', encoding_errors='ignore', on_bad_lines='skip')

        try:
            df['DateTime'] = pd.to_datetime(df['DATE'] + ' ' + df['TIME'], format='%Y-%m-%d %H:%M:%S')
        except Exception as ex:
            print(f"Error parsing date in {daqm_path}: {ex}")
            return pd.DataFrame()
        
        df_year = df['DateTime'].min().date().year
        
        min_year = datetime.today().year-2

        if df_year < min_year:
            return pd.DataFrame()

        df.set_index('DateTime', inplace=True)
            
        df.drop(columns=['DATE', 'TIME'], inplace=True)


        df.sort_index(inplace=True)

        subsetCols = list(set(self.columnlist) & set(df.columns))

        df = df[subsetCols]


        return df


    def remove_spikes(self, df, col_name, window=24, threshold=2.5):
        """

        Removes outliers to make sure the plotted chart doesn't show spikes.\n

        Spikes can occur and have a meaningful significance,\n

        but for reviewing sensors functionality they are removed.
        """

        if col_name == "P_RAIN_1_1_1":

            return df[col_name]

        if df[col_name].isnull().all():

            df[col_name] = np.nan

            return df[col_name]

        try:

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

if __name__ == "__main__":
    obj = GenerateData()
    obj.execute()