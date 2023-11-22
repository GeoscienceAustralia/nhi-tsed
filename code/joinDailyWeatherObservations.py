"""
Join the daily weather observations to the storm data

The daily weather data contains details on the daily maximum wind gust, daily
minimum and maximum temperatures and the observed weather codes reported at
each wather station on a three-hourly cycle.

We can join the data based on the station number and the time of the maximum
daily gust event. This is recorded in the daily max data file for each record.

Input:
- `storm_classification_data.csv`
- Daily maximum wind speed data files
- `StationDetails.geojson`

The `StationDetails.geojson` is used to get the list of stations to work with.

The `storm_classification_data.csv` provides the storm classes

Author: Craig Arthur
Created: 2023-11-21

"""

import os
import sys
import getpass
import warnings
import pandas as pd
import geopandas as gpd
from datetime import datetime, timedelta
from prov.model import ProvDocument

from stndata import DAILYNAMES
from stndata import WXCODES

from files import flStartLog, flGitRepository
from files import flModDate, flPathTime

warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)
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
provlabel = ":joinWeatherObservations"
provtitle = "Join storm class to weather observation data"

TZ = {
    "QLD": 10,
    "NSW": 10,
    "VIC": 10,
    "TAS": 10,
    "SA": 9.5,
    "NT": 9.5,
    "WA": 8,
    "ANT": 0,
}

# List of columns to be used in the merge with the storm database:
COLUMNS = ['datetimeUTC', 'stnNum',
           'PresentWeatherCode', 'PastWeatherCode',
           'thunder', 'hail', 'duststorm']

BASEPATH = os.path.dirname(os.getcwd())
DATAPATH = os.path.join(BASEPATH, 'data')
DAILYMAXPATH = os.path.join(DATAPATH, "allevents", "dailymax")
OUTPUTPATH = os.path.join(DATAPATH, "allevents", "results")
WXDATAPATH = r"X:\georisk\HaRIA_B_Wind\data\raw\from_bom\2023\daily"


LOGGER = flStartLog(
    os.path.join(BASEPATH, "output", "joinStormClass.log"),
    logLevel="INFO",
    verbose=True
    )

codeent = prov.entity(
    sys.argv[0],
    {
        "dcterms:type": "prov:SoftwareAgent",
        "git:commit": commit,
        "git:tag": tag,
        "dcterms:date": dt,
        "prov:url": url,
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


def findPresentWx(row):
    """
    Find the present weather code (if reported) and return the weather
    description.

        Usage:

        df.apply(findPresentWx, axis=1)

    :param row: Row of a DataFrame that includes a datetime and present
    weather codes
    :return: Weather description (if the code exists), or empty string
    :rtype: string
    """
    hour = row['datetimeLST'].hour
    nearest_hour = min(range(0, 24, 3), key=lambda x: abs(x - hour))
    nearest_presentWx_column = f'PresentWx{str(nearest_hour).zfill(2)}'
    try:
        code = int(row[nearest_presentWx_column])
        return WXCODES[code]
    except ValueError:
        return row[nearest_presentWx_column]


def findPastWx(row):
    """
    Find the past weather code that corresponds to the time of the maximum
    daily gust (if reported) and return the weather description

    Usage:

        df.apply(findPastWx, axis=1)

    :param row: Row of a DataFrame
    :return: Weather description (if the code exists) or empty string
    :rtype: string
    """
    hour = row['datetimeLST'].hour
    pastWx_hours = [int(col[6:8]) for col in row.index if col.startswith('PastWx')]  # noqa
    # Find the next PastWx hour
    next_hour = min(filter(lambda x: x > hour, pastWx_hours), default=min(pastWx_hours))  # noqa

    next_pastWx_column = f'PastWx{str(next_hour).zfill(2)}'
    try:
        code = int(row[next_pastWx_column])
        return WXCODES[code]
    except ValueError:
        return row[next_pastWx_column]


def loadDailyData(filename: str, stnState: str) -> pd.DataFrame:
    """
    Load a daily max wind gust file, determine the present and past weather
    code (if they are recorded) and return a `pd.DataFrame`

    :param filename: Full path to a station's daily data file
    :type filename: str
    :param stnState: State the station is located in
    :type stnState: str
    :return: `pd.DataFrame` that holds the daily data, where there is a
    maximum daily wind gust recorded. Only a subset of variables are
    returned.
    :rtype: pd.DataFrame
    """
    try:
        df = pd.read_csv(filename, sep=",",
                         index_col=False,
                         names=DAILYNAMES,
                         header=0,
                         parse_dates={"datetimeLST": [2, 3, 4, 15]},
                         keep_date_col=True,
                         date_format="%Y %m %d %H%M",
                         na_values=["####"],
                         skipinitialspace=True,)
    except FileNotFoundError:
        LOGGER.warning(f"Cannot load {filename} - cannot find the file")
        return None

    # Remove any records where the time is not indicated:
    df.dropna(axis=0, subset=['HHMM'], inplace=True)
    if len(df) == 0:
        LOGGER.debug(f"No valid observations")
        return None
    # Convert time data
    df['datetimeLST'] = pd.to_datetime(df['datetimeLST'],
                                       format="%Y %m %d %H%M")
    df["datetimeUTC"] = df.datetimeLST - timedelta(hours=TZ[stnState])
    df['datetimeUTC'] = df.datetimeUTC.dt.tz_localize("UTC")
    df["date"] = df.datetimeUTC.dt.date

    # Extract the weather codes:
    df['PresentWeatherCode'] = df.apply(findPresentWx, axis=1)
    df['PastWeatherCode'] = df.apply(findPastWx, axis=1)
    df = df[COLUMNS]
    return df


fullStationFile = os.path.join(DATAPATH, "StationDetails.geojson")
LOGGER.info(f"Loading station details from {fullStationFile}")
stnDetails = gpd.read_file(fullStationFile)
stnDetails.set_index("stnNum", inplace=True)
stnDetails['stnWMOIndex'] = stnDetails['stnWMOIndex'].astype('Int64')
prov.entity(
    ":GeospatialStationData",
    {
        "dcterms:type": "void:dataset",
        "dcterms:description": "Geospatial station information",
        "prov:atLocation": fullStationFile,
        "prov:GeneratedAt": flModDate(fullStationFile),
        "dcterms:format": "GeoJSON",
    },
)

LOGGER.info(f"Loaded {len(stnDetails)} stations")

LOGGER.info("Loading storm class data")
stormClassFile = os.path.join(OUTPUTPATH, "storm_classification_data.csv")
stormClassEnt = prov.entity(
    ":ClassifiedDailyStorms",
    {
        "dcterms:title": "Daily classified storm data",
        "dcterms:description": "Daily storm data with storm classes",
        "dcterms:type": "void:Dataset",
        "prov:atLocation": stormClassFile,
        "prov:GeneratedAt": flModDate(stormClassFile),
        "dcterms:format": "comma-separated values"
    }
)

stormData = pd.read_csv(stormClassFile)
# Ensure datetimes are actual pandas datetime objects:
stormData['datetimeUTC'] = pd.to_datetime(stormData.datetimeUTC)
stormData['datetimeLST'] = pd.to_datetime(stormData.datetimeLST)

LOGGER.info("Loading weather description data")
wxDescEnt = prov.collection(
    ":dailyWeatherDescData",
    {
        "dcterms:type": "prov:Collection",
        "dcterms:title": "Daily gust ratio data",
        "prov:atLocation": WXDATAPATH,
        "prov:GeneratedAt": flPathTime(WXDATAPATH),
        },
)

wxdatalist = []
for stn in stnDetails.index:
    LOGGER.debug(f"Loading daily weather data for station {stn}")
    stnState = stnDetails.loc[stn, 'stnState']
    fname = os.path.join(
        WXDATAPATH,
        f"DC02D_Data_{stn:06d}_9999999910405892.txt"
    )
    df = loadDailyData(fname, stnState)
    if df is None:
        next
    else:
        wxdatalist.append(df)
        entity = prov.entity(
           f":{os.path.basename(fname)}",
           {
               "prov.atLocation": WXDATAPATH,
               "dcterms:created": flModDate(fname)
           }
        )
        prov.hadMember(wxDescEnt, entity)

wxData = pd.concat(wxdatalist)

LOGGER.info("Merging storm classifications with weather codes")
outputData = stormData.merge(wxData, how='left', on=['datetimeUTC', 'stnNum'])
LOGGER.info("Saving output data")
outputFile = os.path.join(OUTPUTPATH, "storm_classification_wxcodes.csv")

outputData.to_csv(outputFile, index=False)

stormClassWxCodesEnt = prov.entity(
    ":ClassifiedDailyStormsWxCodes",
    {
        "dcterms:title": "Daily classified storm data with weather codes",
        "dcterms:description": ("Daily storm data with"
                                "storm classes and weather codes"),
        "dcterms:type": "void:Dataset",
        "prov:atLocation": outputFile,
        "prov:GeneratedAt": datetime.now().strftime(DATEFMT),
    }
)

LOGGER.info("Doing cross tabulations of weather codes and storm types")
colorder = ['Synoptic storm', 'Synoptic front',
            'Storm-burst', 'Thunderstorm',
            'Front up', 'Front down', 'Spike',]
presentWxCodesTable = pd.crosstab(outputData['PresentWeatherCode'], outputData['stormType'])[colorder]  # noqa
presentWxCodesTable.to_excel(
    os.path.join(OUTPUTPATH, "storm_classification_presentwxcodes.xlsx")
)

pastWxCodesTable = pd.crosstab(outputData['PastWeatherCode'], outputData['stormType'])[colorder]  # noqa
pastWxCodesTable.to_excel(
    os.path.join(OUTPUTPATH, "storm_classification_pastwxcodes.xlsx")
)

thunderCodeTable = pd.crosstab(outputData['stormType'], outputData['thunder'])
hailCodeTable = pd.crosstab(outputData['stormType'], outputData['hail'])
dustCodeTable = pd.crosstab(outputData['stormType'], outputData['duststorm'])

thunderCodeTable.to_excel(
    os.path.join(OUTPUTPATH, "storm_classification_pastwxthunder.xlsx")
)

hailCodeTable.to_excel(
    os.path.join(OUTPUTPATH, "storm_classification_pastwxhail.xlsx")
)

dustCodeTable.to_excel(
    os.path.join(OUTPUTPATH, "storm_classification_pastwxdust.xlsx")
)

LOGGER.info("Saving provenance data")
prov.wasGeneratedBy(stormClassWxCodesEnt, provlabel)
prov.wasDerivedFrom(stormClassWxCodesEnt, stormClassEnt)
prov.wasDerivedFrom(stormClassWxCodesEnt, wxDescEnt)
prov.used(provlabel, stormClassEnt)
prov.used(provlabel, wxDescEnt)
prov.used(provlabel, ":StormClassFile")
prov.used(provlabel, ":GeospatialStationData")

prov.serialize(
    os.path.join(OUTPUTPATH, "stormclassdatawxcodes.xml"),
    format='xml'
    )

LOGGER.info("Completed")
