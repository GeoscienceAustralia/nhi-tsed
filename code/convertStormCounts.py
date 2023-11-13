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
provlabel = ":stormDataConversion"
provtitle = "Storm data conversion"

BASEDIR = r"..\data\allevents"
OUTPUTPATH = pjoin(BASEDIR, "results")
LOGGER = flStartLog(r"..\output\convertStormCounts.log", "INFO", verbose=True)

LOGGER.info(f"Loading storm classifications from {pjoin(OUTPUTPATH, 'stormclass.pkl')}")
df = pd.read_pickle(pjoin(OUTPUTPATH, "stormclass.pkl"))

fullStationFile = pjoin(r"..\data", "StationDetails.geojson")
LOGGER.info(f"Loading station details from {fullStationFile}")
allstndf = gpd.read_file(fullStationFile)
allstndf.set_index("stnNum", inplace=True)
allstndf['stnWMOIndex'] = allstndf['stnWMOIndex'].astype('Int64')
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

classent = prov.entity(
    ":stormClassifcationSet",
    {
        "dcterms:title": "Storm classifications",
        "dcterms:type": "dcterms:Dataset",
        "prov:generatedAtTime": flModDate(pjoin(OUTPUTPATH, "stormclass.pkl")),
        "prov:atLocation": pjoin(OUTPUTPATH, "stormclass.pkl"),
    }
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

"""
allstnfile = r"X:\georisk\HaRIA_B_Wind\data\raw\from_bom\2022\1-minute\HD01D_StationDetails.txt"  # noqa
allstndf = pd.read_csv(
    allstnfile, sep=',', index_col='stnNum',
    names=ONEMINUTESTNNAMES,
    keep_default_na=False,
    converters={
        'stnName': str.strip,
        'stnState': str.strip,
        'stnDataStartYear': lambda s: int(float(s.strip() or 0)),
        'stnDataEndYear': lambda s: int(float(s.strip() or 0))
    })
"""

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
    ":propStormGeospatialData",
    {
        "dcterms:type": "void:dataset",
        "dcterms:description": "Geospatial storm proportions information",
        "prov:atLocation": pjoin(OUTPUTPATH, "propstorms.json"),
        "prov:GeneratedAt": flModDate(pjoin(OUTPUTPATH, "propstorms.json")),
        "dcterms:format": "GeoJSON",
    }
    )

prov.wasDerivedFrom(propent, ":GeospatialStationData", )
prov.wasDerivedFrom(propent, classent)

LOGGER.info("Plotting maps of storm type proportions")
# Note: cartopy is imported here to avoid a conflict with geopandas/GDAL
from cartopy import crs as ccrs
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = 'Arial'
import cartopy.feature as cfeature

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
plt.savefig(pjoin(OUTPUTPATH, "convectiverate_map.png"), bbox_inches='tight')
plt.close()

gax = plt.axes(projection=ccrs.PlateCarree())
gax.figure.set_size_inches(15, 12)
gdf.plot(column='Non-convectiveRate', legend=True, scheme='quantiles',
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
plt.savefig(pjoin(OUTPUTPATH, "nonconvectiverate_map.png"),
            bbox_inches='tight')
plt.close()

prov.entity(
    ":convectiveRateMap",
    {
        "dcterms:type": "void:dataset",
        "dcterms:description": "Convective storm rate map",
        "prov:atLocation": pjoin(OUTPUTPATH, "convectiverate_map.png"),
        "prov:GeneratedAt": flModDate(pjoin(OUTPUTPATH, "convectiverate_map.png")),
        "dcterms:format": "PNG",
    }
)

prov.entity(
    ":nonconvectiveRateMap",
    {
        "dcterms:type": "void:dataset",
        "dcterms:description": "Non-convective storm rate map",
        "prov:atLocation": pjoin(OUTPUTPATH, "nonconvectiverate_map.png"),
        "prov:GeneratedAt": flModDate(pjoin(OUTPUTPATH, "nonconvectiverate_map.png")),
        "dcterms:format": "PNG",
    }
)

LOGGER.info("Saving provenance information")
endtime = datetime.now().strftime(DATEFMT)
conv = prov.activity(provlabel, starttime, endtime)

prov.used(conv, codeent)
prov.used(conv, classent)
prov.used(conv, ":GeospatialStationData")
prov.wasGeneratedBy(propent, conv)
prov.wasGeneratedBy(":convectiveRateMap", conv)
prov.wasDerivedFrom(":convectiveRateMap", propent)
prov.wasGeneratedBy(":nonconvectiveRateMap", conv)
prov.wasDerivedFrom(":nonconvectiveRateMap", propent)
prov.serialize(pjoin(OUTPUTPATH, "conversion.xml"), format="xml")
from prov.dot import prov_to_dot
dot = prov_to_dot(prov)
dot.write_png(pjoin(OUTPUTPATH, "conversion.png"))
