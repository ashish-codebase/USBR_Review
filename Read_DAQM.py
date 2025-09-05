import glob
import os
import sqlite3
import time
from datetime import datetime
from Parameters import Parameters
import pandas as pd
import numpy as np
from tqdm import tqdm
import Read_Summary
from multiprocessing import Pool
DAQM_columns = Parameters.DAQM_columns
sites = Parameters.sites
import Read_Raw

class GenerateData:
    """
    Creates a database for each site and stores the data from the DAQM files.\n
    """

    def execute(self, limit = 96):
        """Execute the data generation process."""

        merged_df = pd.DataFrame()

        for site in tqdm(sites):

            current_dir = os.path.dirname(os.path.abspath(__file__))

            db_path = os.path.join(current_dir, 'Data', f"{site}.db")

            if os.path.exists(db_path):
                db_connection =sqlite3.connect(db_path)
                db_query = f"SELECT * FROM {site} ORDER BY DateTime DESC LIMIT 1"
                pd_data = pd.read_sql_query(db_query, db_connection)
                db_connection.close()
                last_modification_time_string = pd_data['DateTime'][0] # Reading only 1 line, i.e. the last record at end of the table.
                dt_format = "%Y-%m-%d %H:%M:%S"
                last_modification_time = datetime.strptime(last_modification_time_string, dt_format)
                last_modification_time_seconds = last_modification_time.timestamp()

                current_time = time.time()

                time_difference_hours = (current_time - last_modification_time_seconds) / 3600

                if time_difference_hours < limit:
                    # pass

                    continue


            db_connection = sqlite3.connect(db_path)

            daqm_path1 = r"C:\Users\ashish\OneDrive - University of Nebraska-Lincoln\UNL\All EC Tower Data"

            daqm_path2 = r"D:\OneDrive - University of Nebraska-Lincoln\UNL\All EC Tower Data"


            if os.path.isdir(daqm_path1):

                file_pattern = f"{daqm_path1}\\{site}\\daqm\\*daqm.zip"

            elif os.path.isdir(daqm_path2):

                file_pattern = f"{daqm_path2}\\{site}\\daqm\\*daqm.zip"

            else:
                print(f"Directory for {site} not found in either path.")
                db_connection.close()

                break


            daqm_paths = glob.glob(file_pattern, recursive=True)

            counter = 0

            with Pool(os.cpu_count()) as p:
                daqm_frames = p.map(self.get_dataframe, daqm_paths)

            for daqm_data in daqm_frames:

                # daqm_data = self.get_dataframe(file_path)

                if daqm_data.empty:
                    continue

                if daqm_data.shape[0]<30:

                    continue

                dh_hryl = daqm_data.resample('30min').mean(numeric_only=True)
                dh_hryl['P_RAIN_1_1_1'] = daqm_data['P_RAIN_1_1_1'].resample("30min").sum()
                if counter==0:

                    merged_df = dh_hryl

                    counter=1

                else:

                    merged_df = pd.concat([merged_df,dh_hryl], axis=0)


            merged_df.to_sql(site, db_connection, if_exists="replace", index=True)
            db_connection.execute('VACUUM;')
            db_connection.commit()
            db_connection.close()
            # shutil.copy(db_path, r'D:\OneDrive - University of Nebraska-Lincoln\UNL\StreamlitProjects\FastHTML_Project\static\data')


    def get_dataframe(self, daqm_path):

        df = pd.read_csv(daqm_path, sep="\t", skiprows=[1], encoding='utf-8', encoding_errors='ignore', on_bad_lines='skip')

        try:
            df['DateTime'] = pd.to_datetime(df['DATE'] + ' ' + df['TIME'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
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

        subsetCols = list(set(DAQM_columns) & set(df.columns))

        df = df[subsetCols]

        return df


    def remove_spikes(self, df, col_name:str, window=24, threshold=2.5):
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
    obj.execute(limit=0)  # Force execution without time limit for generating new database and plots.
    for site in sites:
        Read_Summary.create_summary_db(site)
        Read_Raw.parallel_process(site)
