"""
Train and run an automatic classification algorithm for determining the likely
phenomena that generated an severe wind gust.

Based on the temporal evolution of wind gusts, temperature, dew point and
station pressure, we can classify storms into synoptic or convective storms
(and then further sub-categories). We follow an approach similar to Cook (2023)
where a set of high-quality stations are used to develop a training dataset,
which is then used to train a machine-learning algorithm that can classify the
full set of events extracted.


"""
import sys
import getpass
from os.path import join as pjoin
import logging
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib import patheffects
import pandas as pd
import geopandas as gpd
import numpy as np
import seaborn as sns
from prov.model import ProvDocument


from sklearn.preprocessing import StandardScaler
from sktime.classification.kernel_based import RocketClassifier
import sktime

from stndata import ONEMINUTESTNNAMES
from files import flStartLog, flGitRepository
from files import flModDate, flPathTime

np.random.seed(1000)
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
prov.add_namespace("sioc", "http://rdfs.org/sioc/ns#")
prov.add_namespace("git", "https://github.com/GeoscienceAustralia/nhi-tsed")
prov.add_namespace("tsed", "http://ga.gov.au/hazards")
provlabel = "tsed:stormDataClassification"
provtitle = "Storm data classification"

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

sktimeent = prov.entity(
    "sktime",
    {
        "prov:location": "http://doi.org/10.5281/zenodo.3749000",
        "sioc:latest_version": sktime.__version__,
    }
)

pandasent = prov.entity(
    "pandas",
    {
        "prov:location": "https://doi.org/10.5281/zenodo.3509134",
        "sioc:latest_version": pd.__version__,
    }
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

# Following can be put into command line args or config file:
BASEDIR = r"..\data"
TRAINDIR = pjoin(BASEDIR, "training")
OUTPUTPATH = pjoin(BASEDIR, "allevents", "results")
PLOTPATH = pjoin(BASEDIR, "allevents", "plots")
LOGGER = flStartLog(r"..\output\classifyTimeSeries.log", "INFO", verbose=True)
hqstndf = pd.read_csv(pjoin(BASEDIR, "hqstations.csv"), index_col="stnNum")
fullStationFile = pjoin(BASEDIR, "StationDetails.geojson")
eventFile = pjoin(TRAINDIR, "visual_storm_types.csv")

# Station location data (high-quality stations)
stnent = prov.entity(
    "tsed:HQstationLocation",
    {
        "prov:location": pjoin(BASEDIR, "hqstations.csv"),
        "dcterms:title": "High-quality station information",
        "dcterms:type": "void:Dataset",
        "dcterms:created": flModDate(pjoin(BASEDIR, "hqstations.csv")),
    }
)

# This is the visually classified training dataset:
LOGGER.info("Loading visually classified storm data")
stormdf = pd.read_csv(
    eventFile,
    usecols=[1, 2, 3],
    parse_dates=["date"],
    dtype={"stnNum": int, "stormType": "category"},
)

# Storm classification definition entity:
stdef = prov.entity(
    "tsed:stormClassificationTrainingSet",
    {
        "prov:location": eventFile,
        "dcterms:title": "Storm classification training set",
        "dcterms:type": "void:Dataset",
        "dcterms:created": flModDate(eventFile)
    },
)

stormdf.set_index(["stnNum", "date"], inplace=True)
nevents = len(stormdf)
LOGGER.info(f"{nevents} visually classified storms loaded from {eventFile}")

# To demonstrate the performance of the algorithm, we take a random selection
# of 200 storms to test against:
test_storms = stormdf.sample(200)
train_storms = stormdf.drop(test_storms.index)
ntrain = len(train_storms)


def loadData(stnNum: int, datapath: str) -> pd.DataFrame:
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
    fname = pjoin(datapath, "events", f"{stnNum:06d}.pkl")
    LOGGER.debug(f"Loading event data from {fname}")
    df = pd.read_pickle(fname)
    df["date"] = pd.to_datetime(df["date"])
    vars = ["windgust", "tempanom", "stnpanom",
            "dpanom", "windspd", "uanom", "vanom"]
    for var in vars:
        df[var] = df[var].interpolate(method="linear").fillna(method="bfill")

    df["stnNum"] = stnNum
    df.reset_index(inplace=True)
    df.set_index(["stnNum", "date"], inplace=True)
    return df


def plotEvent(df: pd.DataFrame, stormType: str):
    """
    Plot the mean profile of an event

    :param df: DataFrame containing timeseries of temp, wind gust, etc.
    :type df: pd.DataFrame
    :param stormType: Name of the storm class
    :type stormType: str
    """
    pe = patheffects.withStroke(foreground="white", linewidth=5)

    fig, ax = plt.subplots(figsize=(12, 8))
    axt = ax.twinx()
    axp = ax.twinx()
    ax.set_zorder(1)
    ax.patch.set_visible(False)
    lnt = axt.plot(
        df.tdiff,
        df.tempanom,
        label=r"Temperature anomaly [$^o$C]",
        color="r",
        marker="^",
        markerfacecolor="None",
        lw=2,
        path_effects=[pe],
        zorder=1,
        markevery=5,
    )
    lnd = axt.plot(
        df.tdiff,
        df.dpanom,
        color="orangered",
        marker=".",
        markerfacecolor="None",
        lw=1,
        path_effects=[pe],
        zorder=1,
        markevery=5,
        label=r"Dew point anomaly [$^o$C]",
    )
    lnp = axp.plot(
        df.tdiff,
        df.stnpanom,
        color="purple",
        lw=2,
        path_effects=[pe],
        ls="--",
        label="Station pressure anomaly [hPa]",
    )
    lnw = ax.plot(
        df.tdiff,
        df.windgust,
        label="Gust wind speed [km/h]",
        lw=3,
        path_effects=[pe],
        markerfacecolor="None",
        zorder=100,
    )

    axt.spines[["right"]].set_color("r")
    axt.yaxis.label.set_color("r")
    axt.tick_params(axis="y", colors="r")
    axt.set_ylabel(r"Temperature/dewpoint anomaly [$^o$C]")

    ax.set_ylabel("Gust wind speed [km/h]")

    axp.spines[["right"]].set_position(("axes", 1.075))
    axp.spines[["right"]].set_color("purple")
    axp.yaxis.label.set_color("purple")
    axp.tick_params(axis="y", colors="purple")
    axp.set_ylabel("Pressure anomaly [hPa]")

    gmin, gmax = ax.get_ylim()
    pmin, pmax = axp.get_ylim()
    tmin, tmax = axt.get_ylim()
    ax.set_ylim((0, max(gmax, 100)))
    ax.set_xlabel("Time from gust peak [minutes]")
    axp.set_ylim((min(-2.0, pmin), max(pmax, 2.0)))
    axt.set_ylim((min(-2.0, tmin), max(tmax, 2.0)))

    ax.grid(True)
    axt.grid(False)
    axp.grid(False)

    lns = lnw + lnt + lnd + lnp
    labs = [ln.get_label() for ln in lns]
    ax.set_title(stormType)
    ax.legend(lns, labs)
    plt.text(
        1.0,
        -0.05,
        f"Created: {datetime.now():%Y-%m-%d %H:%M %z}",
        transform=ax.transAxes,
        ha="right",
        va="top",
    )
    plt.savefig(pjoin(PLOTPATH, f"{stormType}.png"), bbox_inches="tight")


# Load all the events into a single dataframe. We'll then pick out the events
# based on whether they are in the training set or the test set, using the
# index from the storm classification data:
LOGGER.info("Creating dataframe with all visually classified storm data")
dflist = []
for stn in hqstndf.index:
    df = loadData(stn, TRAINDIR)
    dflist.append(df)

# Visual classification dataframe:
vcdf = pd.concat(dflist)
vcdf["idx"] = vcdf.index

vars = ["windgust", "tempanom", "stnpanom", "dpanom"]
nvars = len(vars)

# Apply a standard scaler (zero mean and unit variance):
scaler = StandardScaler()
vcdf[vars] = scaler.fit_transform(vcdf[vars].values)

# Split into a preliminary training and test dataframes:
LOGGER.info("Splitting the visually classified data into test and train sets")
traindf = vcdf.loc[train_storms.index]
testdf = vcdf.loc[test_storms.index]

trainset = traindf.reset_index().set_index(["idx", "tdiff"])[vars]
trainarray = np.moveaxis(trainset.values.reshape((ntrain, 121, nvars)), 1, -1)
y = np.array(train_storms["stormType"].values)

testset = testdf.reset_index().set_index(["idx", "tdiff"])[vars]
testarray = np.moveaxis(testset.values.reshape((200, 121, nvars)), 1, -1)

# Here we use the full set of visually classified events for training
# the classifier:
fulltrain = (
    vcdf.loc[stormdf.index].
    reset_index().
    set_index(["idx", "tdiff"])[vars])

# noqa: E501
fulltrainarray = np.moveaxis(
    (fulltrain.values.reshape((len(stormdf), 121, nvars))), 1, -1
)

# Create array of storm types from the visually classified data:
fully = np.array(
    list(stormdf.loc[
        fulltrain.reset_index()["idx"].unique()
        ]["stormType"].values)  # noqa: E501
)

# First start with the training set:
LOGGER.info("Running the training set with 10,000 kernels")
rocket = RocketClassifier(num_kernels=10000)
rocket.fit(trainarray, y)
y_pred = rocket.predict(testarray)
results = pd.DataFrame(data={"prediction": y_pred, "visual": test_storms["stormType"]})  # noqa: E501
score = rocket.score(testarray, test_storms["stormType"])
LOGGER.info(f"Accuracy of the classifier for the training set: {score}")

# Now run the classifier on the full event set:
LOGGER.info("Running classifier for all visually-classified events")
rocket = RocketClassifier(num_kernels=10000)
rocket.fit(fulltrainarray, fully)
newclass = rocket.predict(fulltrainarray)
results = pd.DataFrame(data={"prediction": newclass, "visual": fully})
score = rocket.score(fulltrainarray, fully)
LOGGER.info(f"Accuracy of the classifier: {score}")
stormclasses = [
    "Synoptic storm",
    "Synoptic front",
    "Storm-burst",
    "Thunderstorm",
    "Front up",
    "Front down",
    "Spike",
    "Unclassified",
]

(
    pd.crosstab(results["visual"], results["prediction"])
    .reindex(stormclasses)[stormclasses]
    .to_excel(pjoin(TRAINDIR, "crosstab.xlsx"))
)

allstndf = gpd.read_file(fullStationFile)
allstndf.set_index("stnNum", inplace=True)
allstndf['stnWMOIndex'] = allstndf['stnWMOIndex'].astype('Int64')
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

alldatadflist = []
LOGGER.info("Loading all events with maximum gust > 60 km/h")
for stn in allstndf.index:
    try:
        df = loadData(stn, pjoin(BASEDIR, "allevents"))
    except FileNotFoundError:
        LOGGER.debug(f"No data for station: {stn}")
        pass
    else:
        alldatadflist.append(df)

alldatadf = pd.concat(alldatadflist)
alldatadf["idx"] = alldatadf.index

alldatadf[vars] = scaler.transform(alldatadf[vars].values)
allX = alldatadf.reset_index().set_index(["idx", "tdiff"])[vars]

naidx = []
LOGGER.info("Removing storms with insufficient data:")
for ind, tmpdf in allX.groupby(level="idx"):
    if len(tmpdf) < 121:
        naidx.append(ind)
        LOGGER.info(f"< 121 obs: {ind}, {len(tmpdf)}")
    if tmpdf.isna().sum().sum() > 0:
        # Found NAN values in the data (usually dew point)
        naidx.append(ind)
        LOGGER.info(f"NaN values: {ind}, {len(tmpdf)}")

allXupdate = allX.drop(naidx, level="idx")
nstorms = int(len(allXupdate) / 121)
vars = ["windgust", "tempanom", "stnpanom", "dpanom"]
nvars = len(vars)
allXX = np.moveaxis(allXupdate.values.reshape((nstorms, 121, nvars)), 1, -1)
LOGGER.info(f"Running the classifier for all {nstorms} events")
stormclass = rocket.predict(allXX)

LOGGER.info("Reset the scaling to plot data")
allXupdate[vars] = scaler.inverse_transform(allXupdate[vars])

outputstormdf = pd.DataFrame(
    data={"stormType": stormclass},
    index=(allXupdate.index.get_level_values(0).unique()),
)

LOGGER.debug("Writing storm value counts to file")
outputstormdf.stormType.value_counts().to_excel(pjoin(OUTPUTPATH, "stormcounts.xlsx"))  # noqa: E501
stormcounttbl = prov.entity(
    "tsed:stormCountTable",
    {
        "prov:location": pjoin(OUTPUTPATH, "stormcounts.xlsx"),
        "dcterms:title": "Storm count table",
        "dcterms:type": "Spreadsheet",
        "dcterms:created": datetime.now().strftime(DATEFMT),
    }
)

LOGGER.debug("Plotting storm counts")
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
colors = sns.color_palette("viridis", n_colors=8)
outputstormdf.stormType.value_counts().loc[stormclasses].plot(
    kind="bar", ax=ax, color=colors)
plt.text(
    1.0,
    -0.05,
    f"Created: {datetime.now():%Y-%m-%d %H:%M %z}",
    transform=ax.transAxes,
    ha="right",
    va="top",
)
plt.savefig(pjoin(PLOTPATH, "stormcounts.png"), bbox_inches="tight")
stormcountfig = prov.entity(
    "tsed:stormCountFigure",
    {
        "prov:location": pjoin(PLOTPATH, "stormcounts.png"),
        "dcterms:titile": "Storm count figure",
        "dcterms:type": "Figure",
        "dcterms:created": datetime.now().strftime(DATEFMT),

    },
)

# Now we are going to plot the mean profile of the events:
allXupdate["idx"] = allXupdate.index.get_level_values(0)

pltents = []
pltacts = []
for storm in stormclasses:
    LOGGER.info(f"Plotting mean profile for {storm} class events")
    stidx = outputstormdf[outputstormdf["stormType"] == storm].index
    stevents = allXupdate[allXupdate.index.get_level_values("idx").isin(stidx)]
    meanst = stevents.groupby("tdiff").mean(numeric_only=True).reset_index()
    plotEvent(meanst, storm)
    pltent = prov.entity(
        f"tsed:mean{storm}Class",
        {
            "prov:location": pjoin(PLOTPATH, f"{storm}.png"),
            "dcterms:title": f"Mean storm profile for {storm}",
            "dcterms:type": "Figure",
            "dcterms:created": datetime.now().strftime(DATEFMT),
        },
    )
    pltact = prov.activity(
        f":plotMean{storm}",
        datetime.now().strftime(DATEFMT)
    )
    pltents.append(pltent)
    pltacts.append(pltact)

outputFile = pjoin(OUTPUTPATH, "stormclass.pkl")
LOGGER.info(f"Saving storm classification data to {outputFile}")
outputstormdf["stnNum"], outputstormdf["date"] = zip(
    *outputstormdf.reset_index()["idx"]
)

outputstormdf.to_pickle(outputFile)

LOGGER.info("Saving provenance information")
endtime = datetime.now().strftime(DATEFMT)

outputstdef = prov.entity(
    "tsed:stormClassifcationSet",
    {
        "prov:location": outputFile,
        "dcterms:title": "Storm classifications",
        "dcterms:type": "dcterms:Dataset",
    },
)

prov.wasDerivedFrom(outputstdef, stdef, time=datetime.now().strftime(DATEFMT))

classact = prov.activity(provlabel, starttime, endtime)
prov.used(classact, stdef)
prov.used(classact, stnent)
prov.used(classact, "tsed:GeospatialStationData")
prov.wasGeneratedBy(classact, codeent)
prov.wasDerivedFrom(codeent, sktimeent)
prov.wasDerivedFrom(codeent, pandasent)
prov.wasGeneratedBy(outputstdef, classact, time=starttime)
prov.wasGeneratedBy(stormcounttbl, classact)
prov.wasGeneratedBy(stormcountfig, classact)
for pltent, pltact in zip(pltents, pltacts):
    prov.wasGeneratedBy(pltent, pltact)
    prov.wasDerivedFrom(pltent, outputstdef)


prov.serialize(pjoin(OUTPUTPATH, "classification.xml"), format="xml")

LOGGER.info("Completed")
