"""
Objective event classification using gust ratio

Classify wind gust events into convective or non-convective based on the wind
gust ratio, as defined in El Rafei et al. 2023

If r_1 = V_G/V_1 < 2.0 and r_2 = V_G/V_2 < 2.0, then the event is considered
synoptic. Otherwise, the event is classed as convective

V_G = peak gust wind speed
V_1 = mean gust wind speed for the 2 hour time period before the gust event
V_2 = mean gust wind speed for the 2 hour time period after the gust event.

El Rafei, M., S. Sherwood, J. Evans, and A. Dowdy, 2023: Analysis and
characterisation of extreme wind gust hazards in New South Wales,
Australia. *Nat Hazards*, **117**, 875–895,
https://doi.org/10.1007/s11069-023-05887-1.

"""

import os
import sys
import re
import glob
import getpass
import argparse
import logging
from os.path import join as pjoin
from datetime import datetime, timedelta
from configparser import ConfigParser, ExtendedInterpolation
from prov.model import ProvDocument
import pandas as pd
import geopandas as gpd
import numpy as np
import warnings

from process import pAlreadyProcessed, pWriteProcessedFile, pArchiveFile, pInit
from files import flStartLog, flGetStat, flSize, flGitRepository, flModDate
from stndata import ONEMINUTESTNNAMES, ONEMINUTEDTYPE, ONEMINUTENAMES

warnings.simplefilter("ignore", RuntimeWarning)
pd.set_option("mode.chained_assignment", None)

LOGGER = logging.getLogger()
PATTERN = re.compile(r".*Data_(\d{6}).*\.txt")
STNFILE = re.compile(r".*StnDet.*\.txt")
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

DATEFMT = "%Y-%m-%d %H:%M:%S"

prov = ProvDocument()
prov.set_default_namespace("")
prov.add_namespace("prov", "http://www.w3.org/ns/prov#")
prov.add_namespace("xsd", "http://www.w3.org/2001/XMLSchema#")
prov.add_namespace("foaf", "http://xmlns.com/foaf/0.1/")
prov.add_namespace("void", "http://vocab.deri.ie/void#")
prov.add_namespace("dcterms", "http://purl.org/dc/terms/")
provlabel = ":stormGustClassification"
provtitle = "Storm gust classification"


def start():
    """
    Parse command line arguments, initiate processing module (for tracking
    processed files) and start the main loop.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_file",
                        help="Configuration file")
    parser.add_argument("-v", "--verbose",
                        help="Verbose output",
                        action="store_true")
    args = parser.parse_args()

    configFile = args.config_file
    verbose = args.verbose
    config = ConfigParser(allow_no_value=True,
                          interpolation=ExtendedInterpolation())
    config.optionxform = str
    config.read(configFile)
    config.configFile = configFile

    pInit(configFile)
    main(config, verbose)


def main(config, verbose=False):
    """
    Start logger and call the loop to process source files.

    :param config: `ConfigParser` object with configuration loaded
    :param boolean verbose: If `True`, print logging messages to STDOUT

    """

    global g_stations

    logfile = config.get("Logging", "LogFile")
    loglevel = config.get("Logging", "LogLevel", fallback="INFO")
    verbose = config.getboolean("Logging", "Verbose", fallback=verbose)
    datestamp = config.getboolean("Logging", "Datestamp", fallback=False)
    LOGGER = flStartLog(logfile, loglevel, verbose, datestamp)
    outputDir = config.get("Output", "Path", fallback="")
    starttime = datetime.now().strftime(DATEFMT)
    commit, tag, dt, url = flGitRepository(sys.argv[0])

    prov.agent(
        sys.argv[0],
        {
            "dcterms:type": "prov:SoftwareAgent",
            "prov:Revision": commit,
            "prov:tag": tag,
            "dcterms:date": dt,
            "prov:url": url,
        },
    )

    # We use the current user as the primary agent:
    prov.agent(f":{getpass.getuser()}", {"prov:type": "prov:Person"})

    prov.agent(
        "GeoscienceAustralia",
        {
            "prov:type": "prov:Organisation",
            "foaf:name": "Geoscience Australia"
        },
    )

    prov.agent(
        "BureauOfMeteorology",
        {
            "prov:type": "prov:Organization",
            "foaf:name": "Bureau of Meteorology Climate Data Services",
            "foaf:mbox": "climatedata@bom.gov.au",
        },
    )

    configent = prov.entity(
        ":configurationFile",
        {
            "dcterms:title": "Configuration file",
            "dcterms:type": "foaf:Document",
            "dcterms:format": "Text file",
            "prov:atLocation": os.path.basename(config.configFile),
        },
    )

    ListAllFiles(config)
    LoadStationFile(config)
    processFiles(config)

    endtime = datetime.now().strftime(DATEFMT)
    extractionact = prov.activity(
        provlabel,
        starttime,
        endtime,
        {"dcterms:title": provtitle, "dcterms:type": "void:Dataset"},
    )
    prov.actedOnBehalfOf(extractionact, f":{getpass.getuser()}")
    prov.actedOnBehalfOf(f":{getpass.getuser()}", "GeoscienceAustralia")
    prov.used(provlabel, configent)
    prov.used(provlabel, ":GeospatialStationData")
    prov.wasAssociatedWith(extractionact, sys.argv[0])

    prov.serialize(pjoin(outputDir, "gustextraction.xml"), format="xml")

    for key in g_files.keys():
        LOGGER.info(f"Processed {len(g_files[key])} {key} files")
    LOGGER.info("Completed")


def LoadStationFile(config):
    """
    Load a list of stations from a previously-processed GeoJSON file

    :param config: `ConfigParser` object

    """
    global g_stations

    stationFile = config.get("ObservationFiles", "StationFile")
    g_stations = gpd.read_file(stationFile)
    g_stations.set_index("stnNum", inplace=True)
    prov.entity(
        ":GeospatialStationData",
        {
            "dcterms:type": "void:dataset",
            "dcterms:description": "Geospatial station information",
            "prov:atLocation": stationFile,
            "prov:GeneratedAt": flModDate(stationFile),
            "dcterms:format": "GeoJSON",
        },
    )


def ListAllFiles(config):
    """
    For each item in the 'Categories' section of the configuration file, load
    the specification (glob) for the files, then pass to `expandFileSpecs`

    :param config: `ConfigParser` object

    Example:

    [Categories]
    1=CategoryA
    2=CategoryB

    [CategoryA]
    OriginDir=C:/inputpath/
    Option2=OtherValue
    *.csv
    *.zip


    """
    global g_files
    g_files = {}
    categories = config.items("Categories")
    for idx, category in categories:
        specs = []
        items = config.items(category)
        for k, v in items:
            if v == "":
                specs.append(k)
        expandFileSpecs(config, specs, category)


def expandFileSpec(config, spec, category):
    """
    Given a file specification and a category, list all files that match the
    spec and add them to the :dict:`g_files` dict.
    The `category` variable corresponds to a section in the configuration file
    that includes an item called 'OriginDir'.
    The given `spec` is joined to the `category`'s 'OriginDir' and all matching
    files are stored in a list in :dict:`g_files` under the `category` key.

    :param config: `ConfigParser` object
    :param str spec: A file specification. e.g. '*.*' or 'IDW27*.txt'
    :param str category: A category that has a section in the source
    configuration file
    """
    if category not in g_files:
        g_files[category] = []

    origindir = config.get(
        category, "OriginDir", fallback=config.get("Defaults", "OriginDir")
    )
    spec = pjoin(origindir, spec)
    files = glob.glob(spec)
    LOGGER.info(f"{len(files)} {spec} files to be processed")
    for file in files:
        if os.stat(file).st_size > 0:
            if file not in g_files[category]:
                g_files[category].append(file)


def expandFileSpecs(config, specs, category):
    for spec in specs:
        expandFileSpec(config, spec, category)


def getStationList(stnfile: str) -> pd.DataFrame:
    """
    Extract a list of stations from a station file

    :param str stnfile: Path to a station file

    :returns: :class:`pd.DataFrame`
    """
    LOGGER.debug(f"Retrieving list of stations from {stnfile}")
    df = pd.read_csv(
        stnfile,
        sep=",",
        index_col="stnNum",
        names=ONEMINUTESTNNAMES,
        keep_default_na=False,
        converters={"stnName": str.strip, "stnState": str.strip},
    )
    LOGGER.debug(f"There are {len(df)} stations")
    return df


def processFiles(config):
    """
    Process a list of files in each category
    """
    global g_files
    global LOGGER
    unknownDir = config.get("Defaults", "UnknownDir")
    defaultOriginDir = config.get("Defaults", "OriginDir")
    deleteWhenProcessed = config.getboolean(
        "Files", "DeleteWhenProcessed", fallback=False
    )
    archiveWhenProcessed = config.getboolean(
        "Files", "ArchiveWhenProcessed", fallback=True
    )
    outputDir = config.get("Output", "Path", fallback=unknownDir)
    LOGGER.debug(f"DeleteWhenProcessed: {deleteWhenProcessed}")
    LOGGER.debug(f"Output directory: {outputDir}")
    if not os.path.exists(unknownDir):
        os.mkdir(unknownDir)

    if not os.path.exists(pjoin(outputDir, "gustratio")):
        os.makedirs(pjoin(outputDir, "gustratio"))

    category = "ObservationFiles"
    originDir = config.get(category, "OriginDir", fallback=defaultOriginDir)
    LOGGER.debug(f"Origin directory: {originDir}")

    for f in g_files[category]:
        LOGGER.info(f"Processing {f}")
        directory, fname, md5sum, moddate = flGetStat(f)
        if pAlreadyProcessed(directory, fname, "md5sum", md5sum):
            LOGGER.info(f"Already processed {f}")
        else:
            if processFile(f, config):
                LOGGER.info(f"Successfully processed {f}")
                pWriteProcessedFile(f)
                if archiveWhenProcessed:
                    pArchiveFile(f)
                elif deleteWhenProcessed:
                    os.unlink(f)

    gustent = prov.entity(
        ":DailyGustClassification",
        {
            "dcterms:type": "void:Dataset",
            "dcterms:description": "Gust classification of daily max wind gust",  # noqa: E501
            "prov:atLocation": pjoin(outputDir, "gustratio"),
            "prov:GeneratedAt": datetime.now().strftime(DATEFMT),
        },
    )

    prov.wasGeneratedBy(gustent, provlabel)


def processFile(filename: str, config) -> bool:
    """
    process a file and store output in the given output directory

    :param str filename: path to a station data file
    :param str outputDir: Output path for data & figures to be saved
    """

    global g_stations

    outputDir = config.get("Output", "Path")
    outputFormat = config.get("Output", "Format", fallback="pickle")
    ext = "pkl" if outputFormat == "pickle" else "csv"
    outfunc = "to_pickle" if outputFormat == "pickle" else "to_csv"

    LOGGER.info(f"Loading data from {filename}")
    LOGGER.debug(f"Data will be written to {outputDir}")
    m = PATTERN.match(filename)
    stnNum = int(m.group(1))
    stnState = g_stations.loc[stnNum, "stnState"]
    stnName = g_stations.loc[stnNum, "stnName"]
    LOGGER.info(f"{stnName} - {stnNum} ({stnState})")
    filesize = flSize(filename)
    if filesize == 0:
        LOGGER.warning(f"Zero-sized file: {filename}")
        rc = False
    else:
        basename = f"{stnNum:06d}.{ext}"
        dfmax = extractGustRatio(
            filename,
            stnState,
        )
        if dfmax is None:
            LOGGER.warning(f"No data returned for {filename}")
        else:
            outputFile = pjoin(outputDir, "gustratio", basename)
            LOGGER.debug(f"Writing data to {outputFile}")
            getattr(dfmax, outfunc)(outputFile)
            e1 = prov.entity(
                filename,
                {
                    "dcterms:type": "void:dataset",
                    "dcterms:description": "Gust event information",
                    "prov:atLocation": pjoin(outputDir, "events", basename),
                    "prov:GeneratedAt": datetime.now().strftime(DATEFMT),
                    "dcterms:format": outputFormat,
                },
            )
            prov.wasDerivedFrom(provlabel, e1)
        rc = True
    return rc


def extractGustRatio(filename, stnState, variable="windgust"):
    """
    Extract daily maximum value of `variable` from 1-minute observation records
    contained in `filename` and evaluate gust ratio

    :param filename: str, path object or file-like object
    :param stnState: str, station State (for determining local time zone)
    :param str variable: the variable to extract daily maximum values
         default "windgust"

    :returns: `pandas.DataFrame`

    """

    LOGGER.debug(f"Reading station data from {filename}")
    try:
        df = pd.read_csv(
            filename,
            sep=",",
            index_col=False,
            dtype=ONEMINUTEDTYPE,
            names=ONEMINUTENAMES,
            header=0,
            parse_dates={"datetime": [7, 8, 9, 10, 11]},
            na_values=["####"],
            skipinitialspace=True,
        )
    except Exception as err:
        LOGGER.exception(f"Cannot load data from {filename}: {err}")
        return None

    LOGGER.debug("Filtering on quality flags")
    for var in [
        "temp",
        "temp1max",
        "temp1min",
        "wbtemp",
        "dewpoint",
        "rh",
        "windspd",
        "windmin",
        "winddir",
        "windsd",
        "windgust",
        "mslp",
        "stnp",
    ]:
        df.loc[~df[f"{var}q"].isin(["Y"]), [var]] = np.nan

    # Hacky way to convert from local standard time to UTC:
    df["datetimeLST"] = pd.to_datetime(df.datetime, format="%Y %m %d %H %M")
    LOGGER.debug("Converting from local to UTC time")
    df["datetime"] = df.datetimeLST - timedelta(hours=TZ[stnState])
    df["date"] = df.datetime.dt.date
    df.set_index("datetime", inplace=True)
    df.set_index(df.index.tz_localize(tz="UTC"), inplace=True)

    LOGGER.debug("Determining daily maximum wind speed record")
    dfmax = df.loc[df.groupby(["date"])[variable].idxmax().dropna()]
    dfdata = pd.DataFrame(
        columns=["v1", "v2", "r1", "r2", "category"], index=dfmax.index
    )
    for idx, row in dfmax.iterrows():
        startdt = idx - timedelta(hours=2)
        enddt = idx + timedelta(hours=2)
        maxgust = row["windgust"]
        v1 = df.loc[startdt:idx]["windgust"].mean()
        v2 = df.loc[idx:enddt]["windgust"].mean()
        r1 = maxgust / v1
        r2 = maxgust / v2
        if r1 < 2.0 and r2 < 2.0:
            category = "synoptic"
        else:
            category = "convective"
        dfdata.loc[idx] = [v1, v2, r1, r2, category]

    LOGGER.debug("Joining other observations to daily maximum wind data")
    dfmax = dfmax.join(dfdata)
    return dfmax


start()
