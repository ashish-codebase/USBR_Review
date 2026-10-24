import glob
import os
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
from Parameters import Parameters

DAQM_columns = Parameters.DAQM_columns
sites = Parameters.sites
DAYS_LIMIT = 61
def summary_base_path()->str|None:
    daqm_path1 = r"C:\Users\ashish\OneDrive - University of Nebraska\UNL\All EC Tower Data"

    daqm_path2 = r"D:\OneDrive - University of Nebraska\UNL\All EC Tower Data"
    if os.path.isdir(daqm_path1):

        return daqm_path1

    elif os.path.isdir(daqm_path2):

        return daqm_path2

    else:
        print("Neither directory exists.")
        return None

def get_summary_files(site):
    """
    Reads all the summary text files stored in the relevant site's folder.
    Returns ONLY the files within 60 days time frame.
    """
    base_path = summary_base_path()
    if base_path:
        file_pattern = f"{base_path}\\{site}\\summaries\\*-Summary.txt"
        summar_files = glob.glob(file_pattern, recursive=False)
    else:
        return None
    filtered_files = remove_older_summary_files(summar_files)
    return filtered_files
    
def remove_older_summary_files(summary_files):
    if not summary_files:
        return None
    today = datetime.now().date()
    filtered_files = []
    limit = DAYS_LIMIT
    for file in summary_files:
        if os.path.getsize(file) == 0:
            continue
        file_name = os.path.basename(file)
        date_str = file_name.split('_')[0]
        file_time = datetime.strptime(date_str, '%Y-%m-%d').date()
        if today - file_time <= pd.Timedelta(days=limit):
            filtered_files.append(file)

    return filtered_files

def open_connection(site_name):
    root = Path(__file__).parent
    db_path = root / "Data" / f"{site_name}_summary.db"
    connection = sqlite3.connect(db_path)
    return connection


def create_summary_db(site):
    print(f"Start reading: {site} {datetime.now()}")    
    summary_files = get_summary_files(site)
    if not summary_files :
        return
    start_time = datetime.now()
    full_data = pd.DataFrame()
    for summary_file in summary_files:
        fname = summary_file.split('\\')[-1]
        if os.path.getsize(summary_file) == 0 :
            print(f"Summary file {fname} is empty. Skipping.")
            continue
        try:
            df = pd.read_csv(summary_file, sep='\t', skiprows=[1], usecols=["date","time","H","LE","sonic_temperature","air_temperature","ET","RH","VPD","wind_speed","wind_dir","bowen_ratio","x_10%","x_30%","x_50%","x_70%","x_90%","co2_signal_strength_7500_mean", "daytime","u*"])
            df['timestamp'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['time'].astype(str), format='%Y-%m-%d %H:%M:%S')
            full_data = pd.concat([full_data, df], ignore_index=True)
        except Exception as e:
            print(f"Error reading {fname}: {e}")
            continue

        if full_data.empty:
            print(f"No data in: {summary_file}")
            continue
    full_data.set_index('timestamp', inplace=True)
    full_data = full_data[~full_data.index.duplicated(keep='first')]
    full_data.sort_index(inplace=True)
    start_date = datetime.now().date() - timedelta(days=61)
    end_date = datetime.now().date() + timedelta(days=1)
    date_range = pd.date_range(start= start_date, end= end_date, freq='30min')
    blank_df = pd.DataFrame(index=date_range)
    full_data = pd.concat([blank_df, full_data], axis=1)
    full_data.index.name = 'DateTime'
    connection = open_connection(site)
    if not connection:
        print(f"Could not open database connection for {site}.")
        return
    full_data.to_sql(site, connection, if_exists="replace", index=True)
    print(f"Data for {site} written to {site}_summary.db")
    connection.execute('VACUUM;')
    connection.commit()
    connection.close()
    end_time = datetime.now()
    print(f"Time for reading and compressing DB file ==> {end_time - start_time} seconds.")

def db2df(site):
    connection = open_connection(site)
    if not connection:
        print(f"Could not open database connection for {site}.")
        return pd.DataFrame()
    query = f"SELECT * FROM {site}"
    df = pd.read_sql(query, connection, parse_dates=['DateTime'], index_col='DateTime')
    connection.close()
    return df

if __name__ == "__main__":
    for site in sites:
        create_summary_db(site)

