import os
import pandas as pd
import numpy as np
import time
from stndata import ONEMINUTESTNDTYPE, ONEMINUTESTNNAMES

file = r'C:\W10Dev\nhi-tsed\nhi-tsed\data\allevents\results\visual_storm_types_datetime.csv'
storm = pd.read_csv(file)
storm['datetime'] = storm['datetime'].astype('datetime64[ns]')

time_end = pd.DataFrame(storm.groupby(["stnNum"])["datetime"].max())
time_begin = pd.DataFrame(storm.groupby(["stnNum"])["datetime"].min())
station_period = (time_end['datetime'] - time_begin['datetime']).dt.days

storm.loc[(storm["stormType"] == "Thunderstorm") | (storm["stormType"] == "Front up") | (storm["stormType"] == "Front down"), ["stormType"]] = "con"
storm.loc[(storm["stormType"] == "Synoptic storm") | (storm["stormType"] == "Synoptic front") | (storm["stormType"] == "Storm-burst"), ["stormType"]] = "non"

type_num = pd.DataFrame(storm.groupby(["stnNum", "stormType"])["datetime"].count())
con = type_num.xs("con", level='stormType')
non = type_num.xs("non", level='stormType')
# non = type_num.xs("Synoptic front", level='stormType')
con = con.rename(columns={'datetime': 'con'})
non = non.rename(columns={'datetime': 'non'})
# print(con.loc[[1019, 14984, 300060], :])
# print(non.loc[[1019, 14984, 300060], :])

station_period = pd.merge(pd.merge(station_period, con, how ='left', on =['stnNum']), non, how ='left', on =['stnNum'])
station_period["con"] = station_period["con"].fillna(0)
station_period["non"] = station_period["non"].fillna(0)
# print(station_period.loc[[1019, 14984, 300060], :])
# os._exit(0)
print(station_period.columns)
station_period = station_period.rename(columns={'datetime': 'time_span'})

allstnfile = r"X:\georisk\HaRIA_B_Wind\data\raw\from_bom\2023\1-minute\HD01D_StationDetails.txt"
allstndf = pd.read_csv(allstnfile, sep=',', index_col='stnNum',
                       names=ONEMINUTESTNNAMES,
                       keep_default_na=False,
                       converters={
                            'stnName': str.strip,
                            'stnState': str.strip,
                            'stnDataStartYear': lambda s: int(float(s.strip() or 0)),
                            'stnDataEndYear': lambda s: int(float(s.strip() or 0))
                        })

allstndf = pd.merge(allstndf, station_period, how ='left', on =['stnNum'])
print(allstndf.columns)
allstndf['time_span'] = allstndf['time_span'] * allstndf['pctComplete'] * allstndf['pctY'] / 10000.

allstndf["con_prop"] = allstndf["con"] / (allstndf["con"] + allstndf["non"])
allstndf["non_prop"] = allstndf["non"] / (allstndf["con"] + allstndf["non"])

allstndf["con_days"] = allstndf["con"] / allstndf["time_span"] * 365.25
allstndf["non_days"] = allstndf["non"] / allstndf["time_span"] * 365.25

allstndf['time_span'] = allstndf['time_span'] / 365.25

# non convective visual storm max gust wind
file = r'H:\HL\visual_storm\2023\visual_storm_maxgustwind_GPD_nonconvective_8.csv'
names = ['stnNum', 'non_ARI100']
dtype = {'stnNum':int, 'non_ARI100':float}
storm = pd.read_csv(file, header=None, dtype=dtype, names=names, skiprows=1, usecols=[0,9])
allstndf = pd.merge(allstndf, storm, how ='left', on =['stnNum'])
print(allstndf.columns)
    
# convective visual storm max gust wind
file = r'H:\HL\visual_storm\2023\visual_storm_maxgustwind_GPD_convective_8.csv'
names = ['stnNum', 'con_ARI100']
dtype = {'stnNum':int, 'con_ARI100':float}
storm = pd.read_csv(file, header=None, dtype=dtype, names=names, skiprows=1, usecols=[0,9])
allstndf = pd.merge(allstndf, storm, how ='left', on =['stnNum'])
print(allstndf.columns)

# allstndf.loc[allstndf['time_span'] < 8, ["con_prop","non_prop","con_days", "non_days", "con_ARI100", "non_ARI100"]] = ""
allstndf.loc[allstndf['time_span'] < 8, ["con_ARI100", "non_ARI100"]] = ""
print(allstndf.columns)

station = allstndf[['stnNum','stnLon','stnLat','con_prop','non_prop','con_days','non_days','con_ARI100', 'non_ARI100']].copy()
station.to_csv(r"H:\HL\visual_storm\2023\station_location_storm_frequency_ARI100.csv")







