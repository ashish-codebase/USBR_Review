import glob
import os
import pandas as pd
import zipfile
import json
from datetime import datetime
from matplotlib import pyplot as plt
import numpy as np
import multiprocessing
# multiprocessing.freeze_support()
# multiprocessing.set_start_method('spawn')
import matplotlib.colors as mcolors

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
    if not site_name:
        return
    # site_name = 'Baggs'
    base_raw_path = f"D:\\OneDrive - University of Nebraska-Lincoln\\UNL\\All EC Tower Data\\{site_name}\\raw"
    raw_files = glob.glob(f"{base_raw_path}\\2025\\*\\*.ghg", recursive=True)
    # Parallel execute extract_zip_file for all raw_files
    cpu_count = os.cpu_count() or 2
    results = []
    with multiprocessing.Pool(processes=cpu_count) as pool:
        results = pool.map(extract_zip_file, raw_files)

    # for raw_file in raw_files:
    #     results.append(extract_zip_file(raw_file))
        
    df = pd.DataFrame(results, columns=['FileTime','File Name', 'GHG Time', 'SF3 Time', 'Time Offset (s)'])

    df.sort_values(by='FileTime', inplace=True)
    start_time = '2025-01-01'
    end_time = datetime.today()

    df = df.set_index( 'FileTime')
    df = df.loc[start_time : end_time]
    df.to_csv(f'{site_name}_Raw_File_Times.csv', index=True)
    print("Extraction complete.")
    plt.figure(figsize = (24,6))
    val1 = df.loc[df['Time Offset (s)']==-2,'Time Offset (s)']
    val2 = df.loc[df['Time Offset (s)']==-1,'Time Offset (s)']
    val3 = df.loc[df['Time Offset (s)']==0,'Time Offset (s)']
    val4 = df.loc[df['Time Offset (s)']==1,'Time Offset (s)']

    plt.scatter(val1.index, val1.to_numpy(), color='red', s = 1, label='Missing timestamps')
    plt.scatter(val2.index, val2.to_numpy(), color='brown', s = 1, label='Bad timestamps')
    plt.scatter(val3.index, val3.to_numpy(), color='limegreen', s = 1, label='Good timestamps')
    plt.scatter(val4.index, val4.to_numpy(), color='green', s = 1, label='OK timestamps')
    plt.legend(loc='lower left')
    plt.ylim(-2.5, 1.5)
    plt.title(f"{site_name}: Timing offsets GHG & SF3")
    plt.tight_layout()
    plt.savefig(f"{site_name}_offset.png")
    # plt.show()

def main():
    for site in sites:
        parallel_process(site)


if __name__ == "__main__":
    main()



