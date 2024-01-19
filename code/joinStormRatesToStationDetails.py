"""
Join storm rate data to station details and store output in GeoJSON file

The station details are extracted to a GeoJSON and text file in
`extractStationDetails.py`. This is essentially the information provided
by BoM in their station details files.

We join the outputs of the storm classification process in terms of proportion
of different storm types, annual mean storm rates and the basic 1:100 AEP wind
speed for each storm type (where there is sufficient data)

Input:
- `StationDetails.geojson`
- `station_location_storm_frequency_ARI100.csv`

Both files should be indexed by the `stnNum` attribute.

Created: 2023-11-16
Author: Craig Arthur
"""

import os
import sys
import getpass
import pandas as pd
import geopandas as gpd
from datetime import datetime
from prov.model import ProvDocument
from prov.dot import prov_to_dot

from files import flStartLog, flGitRepository
from files import flModDate
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
provlabel = "tsed:joinStormRates"
provtitle = "Join storm rates to station details"

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

BASEDIR = os.path.dirname(os.getcwd())
DATAPATH = os.path.join(BASEDIR, 'data')

LOGGER = flStartLog(
    os.path.join(BASEDIR, "output", "joinStormRates.log"),
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

stormFreqFile = os.path.join(
    DATAPATH,
    "station_location_storm_frequency_ARI100.csv"
    )

prov.entity(
    "tsed:StormFrequencyFile",
    {
        "prov:location": stormFreqFile,
        "dcterms:type": "void:dataset",
        "dcterms:description": "Storm frequency information",
        "dcterms:created": flModDate(stormFreqFile),
        "dcterms:format": "comma-separated values",
    },
)

stormData = pd.read_csv(stormFreqFile, index_col="stnNum")
stormData.drop(columns=['Unnamed: 0', 'stnLon', 'stnLat'], inplace=True)

LOGGER.info("Merging station details with storm data")
outputData = stnDetails.merge(stormData, left_index=True, right_index=True)

LOGGER.info("Renaming columns")
outputData.rename(
    columns={
        "con_ARI100": "conARI100",
        "non_ARI100": "nonARI100",
        "con_prop": "propConv",
        "non_prop": "propNonConv",
        "con_days": "meanConvRate",
        "non_days": "meanNonConvRate"
    },
    inplace=True
)

# Replace NaN values with zeros:
outputData.fillna(
    {
        "conARI100": 0,
        "nonARI100": 0
    },
    inplace=True
)
outputFile = os.path.join(DATAPATH, "StationStormData.geojson")
LOGGER.info(f"Saving data to {outputFile}")
outputData.to_file(outputFile, driver="GeoJSON")

stnstormdata = prov.entity(
    "tsed:StationStormData",
    {
        "prov:location": outputFile,
        "dcterms:type": "void:dataset",
        "dcterms:description": "Geospatial station details with storm data",
        "dcterms:created": datetime.now().strftime(DATEFMT),
        "dcterms:format": "GeoJSON",
    },
)

endtime = datetime.now().strftime(DATEFMT)
prov.activity(":joinStormRates", startTime=starttime, endTime=endtime)
prov.wasGeneratedBy(
    stnstormdata,
    "tsed:joinStormRates",
    time=datetime.now().strftime(DATEFMT)
)
prov.used("tsed:joinStormRates", "tsed:GeospatialStationData")
prov.used("tsed:joinStormRates", "tsed:StormFrequencyFile")
prov.wasAssociatedWith("tsed:joinStormRates", codeent)
prov.serialize(os.path.join(DATAPATH, "station_storm_data.xml"), format="xml")
dot = prov_to_dot(prov)
dot.write_png(os.path.join(DATAPATH, "station_storm_data.png"))
LOGGER.info("Completed")
