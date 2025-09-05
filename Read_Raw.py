import glob
import os
import pandas as pd
import zipfile
import json
from datetime import datetime
from matplotlib import pyplot as plt
import multiprocessing
import matplotlib.dates as mdates
from multiprocessing import get_context
import numpy as np


sites = [
        "Baggs",
        "Boulder",
        "Cora",
        "Cortez",
        "FtBridger",
        "LaPlata",
        "NAPI",
        "Olathe",
        "Gunnison",
        "Farson",
    ]

def extract_zip_file(ghg_path):
    """Extracts SF23_status json file from GHG file."""
    base_name = os.path.basename(ghg_path)
    file_time_string = base_name.split('_')[0]
    file_time = datetime.strptime(file_time_string, '%Y-%m-%dT%H%M%S')
    if '2025-08-01T02300' in base_name:
        pass
    try:
        with zipfile.ZipFile(ghg_path, 'r') as z:
            with z.open(r"system_config/sf23_status.json") as f:
                json_file = json.load(f) 
                ghg_time = json_file['ghg_epoch_time']
                sf3_time = json_file['sf23_epoch_time']
                time_offset = json_file['delta']
                # print("\n")
                # print(ghg_time, sf3_time)
                return file_time, base_name, ghg_time, sf3_time, time_offset
        
    except Exception as e:
        print(f"Error extracting {ghg_path}: {e}")        
        return file_time,base_name, None, None, -2

def parallel_process(site_name):
    try:
        if not site_name:
            return
        # site_name = 'Baggs'
        base_raw_path = f"D:\\OneDrive - University of Nebraska-Lincoln\\UNL\\All EC Tower Data\\{site_name}\\raw"
        raw_files = glob.glob(f"{base_raw_path}\\2025\\*\\*.ghg", recursive=True)
        # Parallel execute extract_zip_file for all raw_files
        cpu_count = os.cpu_count() or 2
        results = []
        with get_context("spawn").Pool(processes=cpu_count) as pool:
            results = pool.map(extract_zip_file, raw_files)
            
        df = pd.DataFrame(results, columns=['FileTime','File Name', 'GHG Time', 'SF3 Time', 'Time Offset (s)'])

        df.sort_values(by='FileTime', inplace=True)
        # Convert dates to datetime objects first
        start_time = datetime(2025, 1, 1)
        end_time = datetime.today()
        start_date = start_time.date()
        end_date = end_time.date()
        # Set x limits using matplotlib dates

        df = df.set_index( 'FileTime')
        df = df.loc[start_time : end_time]
        df.to_csv(f'.//Data//Offset//{site_name}_offsets.csv', index=True)
        
        fig, ax = plt.subplots(figsize = (12,4))
        val1 = df.loc[df['Time Offset (s)']==-2,'Time Offset (s)']
        val2 = df.loc[df['Time Offset (s)']==-1,'Time Offset (s)']
        val3 = df.loc[df['Time Offset (s)']==0,'Time Offset (s)']
        val4 = df.loc[df['Time Offset (s)']==1,'Time Offset (s)']

        ax.scatter(val1.index, val1.to_numpy(), color='red', s = 2, label='Missing timestamps')
        ax.scatter(val2.index, val2.to_numpy(), color='brown', s = 2, label='Bad timestamps')
        ax.scatter(val3.index, val3.to_numpy(), color='limegreen', s = 2, label='Good timestamps')
        ax.scatter(val4.index, val4.to_numpy(), color='green', s = 2, label='OK timestamps')
        ax.legend(loc='lower left')
        ax.set_ylim(-2.5, 1.5)
        ax.set_xlim(start_time,end_time)
        ax.set_xlabel(f"Date range: {start_date} - {end_date}")
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

        ax.set_title(f"{site_name}: Timing offsets GHG & SF3")
        plt.tight_layout()
        plt.savefig(f".//Data//Offset//{site_name}_offset.png")
        print(f"Offset figure created for: {site_name}")
        # plt.show()
    except Exception as e:
        print(f"Following error occured: {e}")
def main():
    for site in sites:
        parallel_process(site)


if __name__ == "__main__":
    main()
