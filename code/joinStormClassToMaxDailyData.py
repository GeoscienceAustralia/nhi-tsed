"""
Join storm classification data to daily maximum gust data

The station details are extracted to a GeoJSON and text file in
`extractStationDetails.py`. This is essentially the information provided
by BoM in their station details files.

We join the outputs of the storm classification process to the daily
maximum wind gust data and the binary classification from El Rafei
et al, based on gust ratio values.

Input:
- `StationDetails.geojson`
- `stormclass.csv`
- `dailymax/<stationNumber>.pkl`

The `StationDetails.geojson is indexed by `stnNum`, `stormclass.csv`
is indexed by `stnNum` and `date`, and each daily max station file is
indexed by `datetime`

The `StationDetails.geojson` is used to get the list of stations to work with.

Created: 2023-11-17
Author: Craig Arthur
"""

import os
import sys
import getpass
import pandas as pd
import geopandas as gpd
from datetime import datetime
from prov.model import ProvDocument

from files import flStartLog, flGitRepository
from files import flModDate, flPathTime
DATEFMT = "%Y-%m-%d %H:%M:%S"

starttime = datetime.now().strftime(DATEFMT)
commit, tag, dt, url = flGitRepository(sys.argv[0])
prov = ProvDocument()
prov.set_default_namespace("")
prov.add_namespace("prov", "http://www.w3.org/ns/prov#")
prov.add_namespace("xsd", "http://www.w3.org/2001/XMLSchema#")
prov.add_namespace("foaf", "http://xmlns.com/foaf/0.1/")
prov.add_namespace("void", "http://vocab.deri.ie/void#")
prov.add_namespace("dcterms", "http://purl.org/dc/terms/")
prov.add_namespace("git", "https://github.com/GeoscienceAustralia/nhi-tsed")
prov.add_namespace("tsed", "https://www.ga.gov.au/hazards")
provlabel = "tsed:joinStormClass"
provtitle = "Join storm class to daily max data"

codeent = prov.entity(
    sys.argv[0],
    {
        "dcterms:type": "prov:SoftwareAgent",
        "git:commit": commit,
        "git:tag": tag,
        "dcterms:date": dt,
        "git:url": url,
    },
)

# We use the current user as the primary agent:
useragent = prov.agent(
    f":{getpass.getuser()}",
    {"prov:type": "prov:Person"}
)

orgagent = prov.agent(
    "GeoscienceAustralia",
    {
        "prov:type": "prov:Organisation",
        "foaf:name": "Geoscience Australia"
    },
)

prov.wasAssociatedWith(codeent, useragent)
prov.actedOnBehalfOf(useragent, orgagent)


def loadData(stnNum: int, filename: str) -> pd.DataFrame:
    """
    Load event data for a given station. Missing values are interpolated
    linearly - if values are missing at the start or end they are backfilled
    from the nearest valid value.

    This data has been extracted by `extractStationData.py`, and is stored in
    pickle files, so there should be no issues around type conversions, when
    used on the same machine.

    :param stnNum: BoM station number
    :type stnNum: int
    :return: DataFrame holding the data of all gust events for a station
    :rtype: `pd.DataFrame`
    """

    LOGGER.debug(f"Loading event data from {fname}")
    try:
        df = pd.read_pickle(fname)
    except FileNotFoundError:
        LOGGER.warning(f"Cannot load {fname}")
        return None
    df["date"] = pd.to_datetime(df["date"])

    df["stnNum"] = stnNum
    df.reset_index(inplace=True)
    df.set_index(["stnNum", "date"], inplace=True)

    return df


BASEPATH = os.path.dirname(os.getcwd())
DATAPATH = os.path.join(BASEPATH, 'data')
DAILYMAXPATH = os.path.join(DATAPATH, "allevents", "dailymax")
GUSTRATIOPATH = os.path.join(DATAPATH, "training", "gustratio")
OUTPUTPATH = os.path.join(DATAPATH, "allevents", "results")

# Fields to retain, and the mapping to final attribute names
FIELDS = {
    'datetime': 'datetimeUTC',
    'datetimeLST': 'datetimeLST',
    'stnNum': 'stnNum',
    'windgust': 'windgust',
    'windspd': 'windspeed',
    'winddir': 'winddir',
    'temp': 'temperature',
    'dewpoint': 'dewpoint',
    'prerain': 'rainfallPre',
    'postrain': 'rainfallPost',
    'pretemp': 'tempPre',
    'posttemp': 'tempPost',
    'mslp': 'meanSLP',
    'stnp': 'stnPressure',
    'gustratio': 'gustRatio',
    'emergence': 'emergence',
    'stormType': 'stormType',
    'category': 'category'
    }

LOGGER = flStartLog(
    os.path.join(BASEPATH, "output", "joinStormClass.log"),
    logLevel="INFO",
    verbose=True
    )

fullStationFile = os.path.join(DATAPATH, "StationDetails.geojson")
LOGGER.info(f"Loading station details from {fullStationFile}")
stnDetails = gpd.read_file(fullStationFile)
stnDetails.set_index("stnNum", inplace=True)
stnDetails['stnWMOIndex'] = stnDetails['stnWMOIndex'].astype('Int64')
prov.entity(
    "tsed:GeospatialStationData",
    {
        "prov:location": fullStationFile,
        "dcterms:type": "void:dataset",
        "dcterms:description": "Geospatial station information",
        "dcterms:created": flModDate(fullStationFile),
        "dcterms:format": "GeoJSON",
    },
)

LOGGER.info(f"Loaded {len(stnDetails)} stations")

stormClassFile = os.path.join(
    DATAPATH, "allevents", "results", "stormclass.csv"
    )

prov.entity(
    "tsed:StormClassFile",
    {
        "prov:location": stormClassFile,
        "dcterms:type": "void:dataset",
        "dcterms:description": "Storm classificaiton information",
        "dcterms:created": flModDate(stormClassFile),
        "dcterms:format": "comma-separated values",
    },
)

# Reset the index so that the date element is a `pd.datetime` object:
stormClass = pd.read_csv(stormClassFile)
stormClass.drop('idx', axis=1, inplace=True)
stormClass['date'] = pd.to_datetime(stormClass.date)
stormClass.set_index(['stnNum', 'date'], inplace=True, drop=False)
stormClass['idx'] = stormClass.index

LOGGER.info("Loading daily max data")
dailyMaxEnt = prov.collection(
    "tsed:dailyMaxWindGustData",
    {
        "prov:location": DAILYMAXPATH,
        "dcterms:type": "prov:Collection",
        "dcterms:title": "Daily maximum wind gust data",
        "dcterms:created": flPathTime(DAILYMAXPATH),
        },
)
dflist = []
for stn in stnDetails.index:
    fname = os.path.join(DAILYMAXPATH, f"{stn:06d}.pkl")
    df = loadData(stn, fname)
    if df is None:
        next
    else:
        dflist.append(df)
        entity = prov.entity(
           f":{os.path.basename(fname)}",
           {
               "prov.atLocation": DAILYMAXPATH,
               "dcterms:created": flModDate(fname)
           }
        )
        prov.hadMember(dailyMaxEnt, entity)

# Daily maximum data dataframe:
dailyMax = pd.concat(dflist)
dailyMax["idx"] = dailyMax.index

LOGGER.info("Loading gust ratio data")
gustRatioEnt = prov.collection(
    "tsed:dailyGustRatioData",
    {
        "prov:location": GUSTRATIOPATH,
        "dcterms:type": "prov:Collection",
        "dcterms:title": "Daily gust ratio data",
        "dcterms:location": flPathTime(GUSTRATIOPATH),
        },
)
gustlist = []
for stn in stnDetails.index:
    fname = os.path.join(GUSTRATIOPATH, f"{stn:06d}.pkl")
    df = loadData(stn, fname)
    if df is None:
        next
    else:
        gustlist.append(df)
        entity = prov.entity(
           f"tsed:{os.path.basename(fname)}",
           {
               "prov:location": GUSTRATIOPATH,
               "dcterms:created": flModDate(fname)
           }
        )
        prov.hadMember(gustRatioEnt, entity)

gustClass = pd.concat(gustlist)
gustClass = gustClass['category']

# Merge the datasets:
LOGGER.info("Merging datasets")
interimData = dailyMax.merge(gustClass, left_index=True, right_index=True)
outputData = interimData.merge(stormClass)

LOGGER.info("Dropping unused columns")
for col in outputData.columns:
    if col not in FIELDS.keys():
        outputData.drop(col, axis=1, inplace=True)

LOGGER.info("Renaming and sorting columns")
outputData.rename(columns=FIELDS, inplace=True)
outputData = outputData[list(FIELDS.values())]

LOGGER.info("Saving output data")
outputFile = os.path.join(OUTPUTPATH, "storm_classification_data.csv")

outputData.to_csv(outputFile, index=False)

stormClassEnt = prov.entity(
    "tsed:ClassifiedDailyStorms",
    {
        "prov:location": outputFile,
        "dcterms:title": "Daily classified storm data",
        "dcterms:description": "Daily storm data with storm classes",
        "dcterms:type": "void:Dataset",
        "dcterms:created": datetime.now().strftime(DATEFMT),
    }
)

LOGGER.info("Saving provenance data")
prov.wasGeneratedBy(stormClassEnt, provlabel)
prov.wasDerivedFrom(stormClassEnt, dailyMaxEnt)
prov.wasDerivedFrom(stormClassEnt, gustRatioEnt)
prov.used(provlabel, dailyMaxEnt)
prov.used(provlabel, gustRatioEnt)
prov.used(provlabel, "tsed:StormClassFile")
prov.used(provlabel, "tsed:GeospatialStationData")

prov.serialize(os.path.join(OUTPUTPATH, "stormclassdata.xml"), format='xml')

LOGGER.info("Completed")
