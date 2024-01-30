"""
Convert storm data to geospatial data and plot rates of occurrence, proportion
of storm types, etc. on maps.

"""
import sys
import getpass
from os.path import join as pjoin
from datetime import datetime

import pandas as pd
import geopandas as gpd
from stndata import ONEMINUTESTNNAMES

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
prov.add_namespace("tsed", "https://ga.gov.au/hazards")
provlabel = "tsed:stormDataConversion"
provtitle = "Storm data conversion"

BASEDIR = r"..\data\allevents"
OUTPUTPATH = pjoin(BASEDIR, "results")
LOGGER = flStartLog(r"..\output\convertStormCounts.log", "INFO", verbose=True)

LOGGER.info(f"Loading storm classifications from {pjoin(OUTPUTPATH, 'stormclass.pkl')}")  # noqa
df = pd.read_pickle(pjoin(OUTPUTPATH, "stormclass.pkl"))

fullStationFile = pjoin(r"..\data", "StationDetails.geojson")
LOGGER.info(f"Loading station details from {fullStationFile}")
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

classent = prov.entity(
    "tsed:stormClassifcationSet",
    {
        "prov:location": pjoin(OUTPUTPATH, "stormclass.pkl"),
        "dcterms:title": "Storm classifications",
        "dcterms:type": "dcterms:Dataset",
        "dcterms:created": flModDate(pjoin(OUTPUTPATH, "stormclass.pkl")),
    }
)

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

# Assume both the start and end year have data:
LOGGER.debug("Calculating time span for stations")
allstndf['timeSpan'] = allstndf.stnDataEndYear - allstndf.stnDataStartYear + 1

stormclasses = ["Synoptic storm", "Synoptic front",
                "Storm-burst", "Thunderstorm",
                "Front up", "Front down",
                "Spike", "Unclassified"]

groupdf = (df.reset_index(drop=True)
           .groupby(['stnNum', 'stormType'])
           .size()
           .reset_index(name='count'))

LOGGER.info("Grouping by station and storm type")
pivotdf = groupdf.pivot_table(index='stnNum', columns='stormType',
                              values='count', fill_value=0)

LOGGER.info("Calculating count by storm type")
fulldf = pivotdf.join(allstndf, on='stnNum', how='left')
fulldf['Convective'] = fulldf[['Thunderstorm', 'Front up', 'Front down']].sum(axis=1)  # noqa
fulldf['Non-convective'] = fulldf[['Synoptic storm', 'Synoptic front', 'Storm-burst']].sum(axis=1)  # noqa
fulldf['stormCount'] = fulldf[stormclasses].sum(axis=1)
fulldf['ConvectiveRate'] = fulldf['Convective'].div(fulldf['timeSpan'], axis=0)
fulldf['Non-convectiveRate'] = fulldf['Non-convective'].div(fulldf['timeSpan'], axis=0)  # noqa

pd.options.mode.copy_on_write = True

LOGGER.info("Calculating proportions of storm types")
propdf = fulldf
propdf[stormclasses] = fulldf[stormclasses].div(fulldf['stormCount'], axis=0)
propdf['Convective'] = fulldf['Convective'].div(fulldf['stormCount'], axis=0)
propdf['Non-convective'] = fulldf['Non-convective'].div(fulldf['stormCount'],
                                                        axis=0)
propdf.to_csv(pjoin(OUTPUTPATH, 'propstorms.csv'))

LOGGER.info("Converting data to GeoDataFrame")
gdf = gpd.GeoDataFrame(fulldf,
                       geometry=gpd.points_from_xy(fulldf.stnLon,
                                                   fulldf.stnLat),
                       crs='epsg:7844')

propgdf = gpd.GeoDataFrame(propdf,
                           geometry=gpd.points_from_xy(
                               propdf.stnLon, propdf.stnLat
                               ),
                           crs='epsg:7844')
propgdf.to_file(pjoin(OUTPUTPATH, "propstorms.json"), driver="GeoJSON")

propent = prov.entity(
    "tsed:propStormGeospatialData",
    {
        "prov:location": pjoin(OUTPUTPATH, "propstorms.json"),
        "dcterms:description": "Geospatial storm proportions information",
        "dcterms:type": "void:dataset",
        "dcterms:created": flModDate(pjoin(OUTPUTPATH, "propstorms.json")),
        "dcterms:format": "GeoJSON",
    }
    )

prov.wasDerivedFrom(propent, "tsed:GeospatialStationData", )
prov.wasDerivedFrom(propent, classent)

LOGGER.info("Plotting maps of storm type proportions")
# Note: cartopy is imported here to avoid a conflict with geopandas/GDAL
from cartopy import crs as ccrs  # noqa
import matplotlib.pyplot as plt  # noqa
import matplotlib  # noqa
matplotlib.rcParams['font.sans-serif'] = 'Arial'
import cartopy.feature as cfeature  # noqa

states = cfeature.NaturalEarthFeature(
        category='cultural',
        name='admin_1_states_provinces_lines',
        scale='10m',
        facecolor='none')

gax = plt.axes(projection=ccrs.PlateCarree())
gax.figure.set_size_inches(15, 12)
propgdf.plot(column='Convective',
             legend=True,
             scheme='quantiles',
             k=7, ax=gax)

gax.coastlines(resolution='10m')
gax.add_feature(states, edgecolor='0.15', linestyle='--')
gax.set_extent([110, 160, -45, -10])
gl = gax.gridlines(draw_labels=True, linestyle=":")
gl.top_labels = False
gl.right_labels = False
gax.set_title("Convective storms")
plt.text(1.0, -0.05,
         f"Created: {datetime.now():%Y-%m-%d %H:%M %z}",
         transform=gax.transAxes, ha='right')
plt.savefig(pjoin(OUTPUTPATH, "convective_map.png"), bbox_inches='tight')
plt.close()

gax = plt.axes(projection=ccrs.PlateCarree())
gax.figure.set_size_inches(15, 12)
propgdf.plot(column='Non-convective',
             legend=True,
             scheme='quantiles',
             k=7, ax=gax)

gax.coastlines(resolution='10m')
gax.add_feature(states, edgecolor='0.15', linestyle='--')
gax.set_extent([110, 160, -45, -10])
gl = gax.gridlines(draw_labels=True, linestyle=":")
gl.top_labels = False
gl.right_labels = False
gax.set_title("Non-convective storms")
plt.text(1.0, -0.05, f"Created: {datetime.now():%Y-%m-%d %H:%M %z}",
         transform=gax.transAxes, ha='right')
plt.savefig(pjoin(OUTPUTPATH, "nonconvective_map.png"), bbox_inches='tight')
plt.close()

gax = plt.axes(projection=ccrs.PlateCarree())
gax.figure.set_size_inches(15, 12)
gdf.plot(column='ConvectiveRate', legend=True, scheme='quantiles',
         k=7, ax=gax)

gax.coastlines(resolution='10m')
gax.add_feature(states, edgecolor='0.15', linestyle='--')
gax.set_extent([110, 160, -45, -10])
gl = gax.gridlines(draw_labels=True, linestyle=":")
gl.top_labels = False
gl.right_labels = False
gax.set_title("Convective storm rate")
plt.text(1.0, -0.05, f"Created: {datetime.now():%Y-%m-%d %H:%M %z}",
         transform=gax.transAxes, ha='right')
convFig = pjoin(OUTPUTPATH, "convectiverate_map.png")
plt.savefig(convFig, bbox_inches='tight')
plt.close()

gax = plt.axes(projection=ccrs.PlateCarree())
gax.figure.set_size_inches(15, 12)
gdf.plot(column='Non-convectiveRate',
         legend=True, scheme='quantiles',
         k=7, ax=gax)

gax.coastlines(resolution='10m')
gax.add_feature(states, edgecolor='0.15', linestyle='--')
gax.set_extent([110, 160, -45, -10])
gl = gax.gridlines(draw_labels=True, linestyle=":")
gl.top_labels = False
gl.right_labels = False
gax.set_title("Non-convective storm rate")
plt.text(1.0, -0.05, f"Created: {datetime.now():%Y-%m-%d %H:%M %z}",
         transform=gax.transAxes, ha='right')
nonconvFig = pjoin(OUTPUTPATH, "nonconvectiverate_map.png")
plt.savefig(nonconvFig, bbox_inches='tight')
plt.close()

prov.entity(
    "tsed:convectiveRateMap",
    {
        "prov:location": convFig,
        "dcterms:type": "void:dataset",
        "dcterms:description": "Convective storm rate map",
        "dcterms:created": flModDate(convFig),
        "dcterms:format": "PNG",
    }
)

prov.entity(
    "tsed:nonconvectiveRateMap",
    {
        "prov:location": nonconvFig,
        "dcterms:type": "void:dataset",
        "dcterms:description": "Non-convective storm rate map",
        "dcterms:created": flModDate(nonconvFig),
        "dcterms:format": "PNG",
    }
)

LOGGER.info("Saving provenance information")
endtime = datetime.now().strftime(DATEFMT)
conv = prov.activity(provlabel, starttime, endtime)

prov.used(conv, codeent)
prov.used(conv, classent)
prov.used(conv, "tsed:GeospatialStationData")
prov.wasGeneratedBy(propent, conv)
prov.wasGeneratedBy("tsed:convectiveRateMap", conv)
prov.wasDerivedFrom("tsed:convectiveRateMap", propent)
prov.wasGeneratedBy("tsed:nonconvectiveRateMap", conv)
prov.wasDerivedFrom("tsed:nonconvectiveRateMap", propent)
prov.serialize(pjoin(OUTPUTPATH, "conversion.xml"), format="xml")
dot = prov_to_dot(prov)
dot.write_png(pjoin(OUTPUTPATH, "conversion.png"))
