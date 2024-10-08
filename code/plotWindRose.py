import os
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import geopandas as gpd
import numpy as np
from windrose import WindroseAxes, plot_windrose
from files import flStartLog, flGitRepository
from datetime import datetime


BASEDIR = os.path.dirname(os.getcwd())
DATAPATH = os.path.join(BASEDIR, 'data')
fullStationFile = os.path.join(DATAPATH, "StationDetails.geojson")
LOGGER = flStartLog(
    os.path.join(BASEDIR, "output", "plotWindRose.log"),
    logLevel="INFO",
    verbose=True
    )

fullStationFile = os.path.join(DATAPATH, "StationDetails.geojson")
LOGGER.info(f"Loading station details from {fullStationFile}")
stnDetails = gpd.read_file(fullStationFile)
stnDetails.set_index("stnNum", inplace=True)
stnDetails['stnWMOIndex'] = stnDetails['stnWMOIndex'].astype('Int64')

stormClassFile = os.path.join(
    DATAPATH, "allevents", "results", "storm_classification_wxcodes.csv"
    )

def plot_windrose_subplots(data, *, direction, var, color=None, **kwargs):
    """wrapper function to create subplots per axis"""
    ax = plt.gca()
    ax = WindroseAxes.from_ax(ax=ax, theta_labels=["E", "NE", "N", "NW", "W", "SW", "S", "SE"])
    plot_windrose(direction_or_df=data[direction], var=data[var], ax=ax, **kwargs)


def plotClassifiedWindRose(df, stnNum, stnData):
    g = sns.FacetGrid(
        data=df,
        # the column name for each level a subplot should be created
        col="stormType",
        # place a maximum of 3 plots per row
        col_wrap=3,
        col_order=['Synoptic storm', 'Synoptic front', 'Storm-burst',
                   'Thunderstorm', 'Front up', 'Front down'],
        subplot_kws={"projection": "windrose"},
        sharex=False,
        sharey=False,
        despine=False,
        height=3.5,
    )

    g.map_dataframe(
        plot_windrose_subplots,
        data=df,
        direction="winddir",
        var="windgust",
        normed=True,
        # manually set bins, so they match for each subplot
        bins=(0.1, 60, 70, 80, 90, 100, 120, 140),
        calm_limit=0.1,
        kind="bar",
    )

    y_ticks = range(20, 65, 20)
    for ax in g.axes:
        ax.set_rgrids(y_ticks, y_ticks)

    plt.subplots_adjust(bottom=0.15, left=0.0, right=1.0, top=0.85)
    g.axes[0].legend(bbox_to_anchor=(0.5, 0.0), bbox_transform=g.figure.transFigure, loc='lower center', ncols=4)
    g.figure.suptitle(stnData.stnName, )
    #plt.text(0.95, 0.025, f"Created: {datetime.now():%Y-%m-%d %H:%M %z}",
    #         transform=g.figure.transFigure, ha='right',
    #         fontsize='xx-small')
    #g.figure.tight_layout()
    outputfile=os.path.join(DATAPATH, "allevents", "plots", f"{stnNum}.WR.png")
    plt.savefig(outputfile, dpi=600)
    plt.close(g.figure)


def plotERWindRose(df, stnNum, stnData):
    try:
        g = sns.FacetGrid(
            data=df,
            # the column name for each level a subplot should be created
            col="category",
            # place a maximum of 3 plots per row
            col_wrap=2,
            subplot_kws={"projection": "windrose"},
            sharex=False,
            sharey=False,
            despine=False,
            height=3.5,
        )
    except ValueError as e:
        LOGGER.exception(e)
        LOGGER.exception("Appears there's only 1 class of storm here")
        return

    g.map_dataframe(
        plot_windrose_subplots,
        data=df,
        direction="winddir",
        var="windgust",
        normed=True,
        # manually set bins, so they match for each subplot
        bins=(0.1, 60, 70, 80, 90, 100, 120, 140),
        calm_limit=0.1,
        kind="bar",
    )

    y_ticks = range(10, 45, 10)
    for ax in g.axes:
        ax.set_rgrids(y_ticks, y_ticks)

    plt.subplots_adjust(bottom=0.2, left=0.0, right=1.0, top=0.8)
    g.axes[0].legend(bbox_to_anchor=(0.5, 0.0),
                     bbox_transform=g.figure.transFigure,
                     loc='lower center', ncols=4, fontsize='small')
    g.figure.suptitle(stnData.stnName)
    outputfile=os.path.join(
    DATAPATH, "allevents", "plots", f"{stnNum}.ER.png")
    plt.savefig(outputfile, dpi=600)
    plt.close(g.figure)

df = pd.read_csv(stormClassFile)

#stn=66037
#sdf = df[df.stnNum==stn]
#row=stnDetails.loc[stn]

#plotClassifiedWindRose(sdf, stn, row)
#plotERWindRose(sdf, stn, row)
for stn, row in stnDetails.iterrows():
    print(stn, row.stnName)
    row=stnDetails.loc[stn]
    sdf = df[df.stnNum==stn]
    if len(sdf)==0:
        continue
    else:
        plotClassifiedWindRose(sdf, stn, row)
        plotERWindRose(sdf, stn, row)
