
import json
from datetime import datetime
import zipfile
# import line_profiler
import os

# @line_profiler.profile
def extract_zip_file(ghg_path):
    """Extracts SF23_status json file from GHG file."""
    base_name = os.path.basename(ghg_path)
    file_time_string = base_name.split('_')[0]
    file_time = datetime.strptime(file_time_string, '%Y-%m-%dT%H%M%S')
    try:
        with zipfile.ZipFile(ghg_path, 'r') as z:
            with z.open(r"system_config/sf23_status.json") as f:
                json_file = json.load(f) 
                ghg_time = json_file['ghg_epoch_time']
                sf3_time = json_file['sf23_epoch_time']
                time_offset = json_file['delta']
                return file_time, base_name, ghg_time, sf3_time, time_offset
        
    except Exception as e:
        # print(f"Error extracting {ghg_path}: {e}")        
        return file_time,base_name, None, None, -2