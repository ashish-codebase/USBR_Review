import pandas as pd
import streamlit as st
from pathlib import Path
import os
from datetime import datetime
from Read_Summary import db2df
from time import process_time

class DB_Reader:
    def __init__(self) -> None:
        self.baggs = self.Baggs()
        self.boulder = self.Boulder()
        self.cora = self.Cora()
        self.cortez = self.Cortez()
        self.farson = self.Farson()
        self.ftBridger = self.FtBridger()
        self.gunnison = self.Gunnison()
        self.laPlata = self.LaPlata()
        self.nAPI = self.NAPI()
        self.olathe = self.Olathe()

    def read_parquet(self, site_name):
        root = Path(__file__).parent
        db_path = root / "Data" / f"{site_name}.parquet"

        if os.path.getsize(db_path)==0:        
            return pd.DataFrame()
        
        daqm_data = pd.read_parquet(db_path, engine='pyarrow')
        # daqm_data.set_index('DateTime', inplace=True)


        date_range = pd.date_range(start= datetime.now().date() - pd.Timedelta(days=61), end= datetime.now().date() + pd.Timedelta(days=1), freq='30min')
        filtered_df = daqm_data[(daqm_data.index >= date_range[0]) & (daqm_data.index <= date_range[-1])]
        blank_df = pd.DataFrame(index=date_range)
        blank_df.index.name = 'DateTime'
        full_data = blank_df.merge(filtered_df, left_index=True, right_index=True, how='left')
        # full_data.index.name = 'DateTime'
        
        sonic_data = db2df(site_name)
        combined_data = full_data.merge(sonic_data, left_index=True, right_index=True, how='left')
        combined_data['RN_G'] = combined_data['RN_1_1_1'] - combined_data['SHF_1_1_1']
        combined_data['LE_H'] = combined_data['LE'] + combined_data['H']
        # combined_data['EB_Array] = combined_data['LE'] + combined_data['H']
        combined_data.drop(['date','time'],axis=1, inplace=True)
        return combined_data
        
    def get_db(self,  site_name) -> pd.DataFrame:
        start_time = process_time()
        site_map = {
            'Baggs': self.baggs,
            'Boulder': self.boulder,
            'Cora': self.cora,
            'Cortez': self.cortez,
            'Farson': self.farson,
            'FtBridger': self.ftBridger,
            'Gunnison': self.gunnison,
            'LaPlata': self.laPlata,
            'NAPI': self.nAPI,
            'Olathe': self.olathe
            }
        
        end_time = process_time()
        print(f'{site_name} read time: {(end_time-start_time):.8f} seconds.')
        return site_map.get(site_name, pd.DataFrame()) 


    @st.cache_data
    def Baggs(_self):
        return _self.read_parquet('Baggs')
    
    @st.cache_data
    def Boulder(_self):
        return _self.read_parquet('Boulder')
    
    @st.cache_data
    def Cora(_self):
        return _self.read_parquet('Cora')
    
    @st.cache_data    
    def Cortez(_self):
        return _self.read_parquet('Cortez')
    
    @st.cache_data        
    def Farson(_self):
        return _self.read_parquet('Farson')
    
    
    @st.cache_data
    def FtBridger(_self):
        return _self.read_parquet('FtBridger')
    
    @st.cache_data    
    def Gunnison(_self):
        return _self.read_parquet('Gunnison')
    
    @st.cache_data    
    def LaPlata(_self):
        return _self.read_parquet('LaPlata')
    
    @st.cache_data    
    def NAPI(_self):
        return _self.read_parquet('NAPI')
    
    @st.cache_data    
    def Olathe(_self):
        return _self.read_parquet('Olathe')
    

if __name__ == "__main__":
    DB_Reader()