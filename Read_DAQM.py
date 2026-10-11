import glob
import os
from datetime import datetime, timedelta
from Parameters import Parameters
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
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

    def filter_dates(self, daqm_paths):
        filtered_files = []
        format_string = '%Y-%m-%d-%H%M%S'
        cutoff_date = datetime.today() - timedelta(days=60)
        for daqm_path in daqm_paths:
            file_name = os.path.basename(daqm_path)
            time_string = file_name[:17]
            time_stamp = datetime.strptime(time_string,format_string)
            if time_stamp>cutoff_date:
                filtered_files.append(daqm_path)
        return filtered_files


    def execute(self, limit = 96):
        """Execute the data generation process."""
        compilation_time = datetime.now()

        with open("last_compiled.txt", "w") as f:
            f.write(compilation_time.strftime("%Y-%m-%d %H:%M:%S"))
        merged_df = pd.DataFrame()

        for site in tqdm(sites):

            daqm_path1 = r"C:\Users\ashish\OneDrive - University of Nebraska\UNL\All EC Tower Data"

            daqm_path2 = r"D:\OneDrive - University of Nebraska\UNL\All EC Tower Data"


            if os.path.isdir(daqm_path1):

                file_pattern = f"{daqm_path1}\\{site}\\daqm\\*daqm.zip"

            elif os.path.isdir(daqm_path2):

                file_pattern = f"{daqm_path2}\\{site}\\daqm\\*daqm.zip"

            else:
                print(f"Directory for {site} not found in either path.")
                break


            daqm_paths = glob.glob(file_pattern, recursive=True)
            daqm_paths = self.filter_dates(daqm_paths=daqm_paths)

            counter = 0

            with Pool(os.cpu_count()) as p:
                daqm_frames = p.map(self.get_dataframe, daqm_paths)

            for daqm_data in daqm_frames:

                # daqm_data = self.get_dataframe(file_path)

                if daqm_data.empty:
                    continue

                row_count = daqm_data.shape[0]
                if row_count<30:
                    continue

                data_hourly = daqm_data.resample('30min').mean(numeric_only=True)
                data_hourly['P_RAIN_1_1_1'] = daqm_data['P_RAIN_1_1_1'].resample("30min").sum()

                if counter==0:
                    merged_df = data_hourly
                    counter=1

                else:
                    merged_df = pd.concat([merged_df,data_hourly], axis=0)

            # merged_df['DateTime'] = merged_df.index
            merged_df.index.name = 'DateTime'

            merged_df.to_parquet(f"./Data/{site}.parquet", engine='pyarrow',index=True)



    def get_dataframe(self, daqm_path):

        try:
            df = pd.read_csv(daqm_path, sep="\t", skiprows=[1], encoding='utf-8', encoding_errors='ignore', on_bad_lines='skip')
            df['DateTime'] = pd.to_datetime(df['DATE'].astype(str) + ' ' + df['TIME'].astype(str), format='%Y-%m-%d %H:%M:%S', errors='coerce')
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

def DeleteUnnecessaryFiles():
    unnecessaryFiles = glob.glob("./**/*-DESKTOP-*.*",recursive=True)
    for file in unnecessaryFiles:
        if os.path.isfile(file):
            os.remove(file)

if __name__ == "__main__":
    DeleteUnnecessaryFiles()
    obj = GenerateData()
    obj.execute(limit=0)  # Force execution without time limit for generating new database and plots.
    for site in sites:
        Read_Summary.create_summary_db(site)
        Read_Raw.parallel_process_ghg(site)
