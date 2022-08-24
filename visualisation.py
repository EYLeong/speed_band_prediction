import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString
import contextily as ctx

def load_json(filepath):
    '''
    Utility function to load a json file
    -----------------------------
    :params:
        Path/string filepath: Path of the file to be opened
    -----------------------------
    :returns:
        dict: Python dictionary representation of the json file
    '''
    with open(filepath) as f:
        return json.load(f)

def get_speedband(filepath, link_id):
    '''
    Retrieve the speedband from a specific json file of a specific road
    -----------------------------
    :params:
        Path/string filepath: Path of the file to be opened
        int link_id: ID of the road
    -----------------------------
    :returns:
        int: Speedband
    '''
    jjson = load_json(filepath)
    for link in jjson:
        if link["LinkID"] == str(link_id):
            return link["SpeedBand"]

def filename_to_time(filename):
    '''
    Convert the filename into a python date object
    -----------------------------
    :params:
        string filename: Filename of format %H:%M:%S
    -----------------------------
    :returns:
        date: Python date representation of the filename
    '''
    time_string = filename.split(".")[0]
    date_time = datetime.strptime(time_string, "%H:%M:%S")
    return date_time
        
def get_speedbands(daypath, link_id):
    '''
    Retrieve all the timestamps and corresponding speedbands for a specific road on a specific day
    -----------------------------
    :params:
        Path daypath: Path object of the directory where all json files for that day are stored
        int link_id: ID of the target road
    -----------------------------
    :returns:
        list: List of all timestamps as date objects
        list: List of all speedbands
    '''
    file_list = os.listdir(daypath)
    file_list.sort()
    time_list = []
    speedband_list = []
    for file in file_list:
        time = filename_to_time(file)
        time_list.append(time)
        speedband = get_speedband(daypath/file, link_id)
        speedband_list.append(speedband)
    return time_list, speedband_list

def plot_speedbands(daypath, link_id):
    '''
    Plot all the speedbands for a specified road on a specific day
    -----------------------------
    :params:
        Path daypath: Path object of the directory where all json files for that day are stored
        int link_id: ID of the target road
    -----------------------------
    :returns:
        None
    '''
    times, speedbands = get_speedbands(daypath, link_id)
    plt.plot(times, speedbands, label = daypath)
    ax = plt.gca()
    myFmt = mdates.DateFormatter('%H_%M')
    ax.xaxis.set_major_formatter(myFmt)
    ax.set_xlabel("Time")
    ax.set_ylabel("Speed Band")
    ax.set_title(link_id)
    ax.legend()
    plt.show()
    
def plot_all_speedbands(datadir, day, link_id):
    '''
    Plot all the speedbands for a specified road on all dates corresponding to that day (all Mondays, or Tuesdays, etc)
    -----------------------------
    :params:
        Path datadir: Path object of the data directory
        string day: Day of the week e.g. Mon, Tue, etc
        int link_id: ID of the target road
    -----------------------------
    :returns:
        None
    '''
    datelist = os.listdir(datadir)
    datelist = [date for date in datelist if date.startswith(day)]
    for date in datelist:
        times, speedbands = get_speedbands(datadir/date, link_id)
        plt.plot(times, speedbands, label = date)
    ax = plt.gca()
    myFmt = mdates.DateFormatter('%H_%M')
    ax.xaxis.set_major_formatter(myFmt)
    ax.set_xlabel("Time")
    ax.set_ylabel("Speed Band")
    ax.set_title(link_id)
    ax.legend()
    plt.show()
    
def loc_to_linestring(loc):
    '''
    Utility function to create shapely linestrings from road location data
    -----------------------------
    :params:
        string loc: Location data of format (start_lat start_lon end_lat end_lon)
    -----------------------------
    :returns:
        LineString: Corresponding LineString representing road
    '''
    coordArr = loc.split()
    coordArr = [float(coord) for coord in coordArr]
    return LineString([coordArr[1::-1], coordArr[3:1:-1]])

# Road Category letters to names, and colors for plotting
CATEGORIES = {
    "A":("Expressway", "tab:cyan"),
    "B":("Major Arterial Road", "tab:red"),
    "C":("Arterial Road", "tab:blue"),
    "D":("Minor Arterial Road", "tab:green"),
    "E":("Small Road", "tab:olive"),
    "F":("Slip Road", "tab:purple"),
    "G":("Other", "tab:brown"),
}

def plot_map(filepath, links=[]):
    '''
    Plot the geographical map of the road network with specific roads colored
    -----------------------------
    :params:
        string/Path filepath: Path of any json file of road links (just to get the location data of the roads)
        list links: IDs of the roads to highlight. If empty, the road categories are used for coloring
    -----------------------------
    :returns:
        None
    '''
    df = pd.read_json(filepath)
    linestrings = df["Location"].apply(loc_to_linestring)
    gdf = gpd.GeoDataFrame(df, geometry=linestrings, crs="EPSG:4326")
    gdf = gdf.to_crs('EPSG:3857')
    fig, ax = plt.subplots(figsize=(10, 10))
    if len(links) == 0:
        for k, v in CATEGORIES.items():
            trunc = gdf[gdf["RoadCategory"] == k]
            if len(trunc) != 0:
                ax = trunc.plot(ax=ax, color=v[1], label=v[0])
    else:
        cmap = plt.cm.get_cmap("brg", len(links)+1)
        for i in range(len(links)):
            trunc = gdf[gdf["LinkID"] == links[i]]
            if len(trunc) != 0:
                ax = trunc.plot(ax=ax, label=links[i], color=cmap(i))
        gdf[~gdf["LinkID"].isin(links)].plot(ax=ax, color=cmap(len(links)))

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend()
    ctx.add_basemap(ax)
    plt.show()

def get_avg_speedband(filepath):
    '''
    Retrieve the average speedband from a specific json file of all roads
    -----------------------------
    :params:
        Path/string filepath: Path of the file to be opened
    -----------------------------
    :returns:
        float: Average speedband
    '''
    jjson = load_json(filepath)
    total = 0
    count = 0
    for link in jjson:
        total += link["SpeedBand"]
        count += 1
    return total/count

def get_avg_speedbands(daypath):
    '''
    Retrieve all the timestamps and corresponding average speedbands for all roads on a specific day
    -----------------------------
    :params:
        Path daypath: Path object of the directory where all json files for that day are stored
    -----------------------------
    :returns:
        list: List of all timestamps as date objects
        list: List of all average speedbands
    '''
    file_list = os.listdir(daypath)
    file_list.sort()
    time_list = []
    speedband_list = []
    for file in file_list:
        time = filename_to_time(file)
        time_list.append(time)
        speedband = get_avg_speedband(daypath/file)
        speedband_list.append(speedband)
    return time_list, speedband_list

def get_all_avg_speedbands(datadir, days):
    '''
    Returns the average speedband across all roads by the hour for all of the days specified.
    -----------------------------
    :params:
        Path datadir: Path object of the data directory
        list days: List of the days to include in the average
    -----------------------------
    :returns:
        dict: Dictionary of hour -> average speedband across all roads for selected days
    '''
    datelist = os.listdir(datadir)
    datelist = [date for date in datelist if date.split("_")[0] in days]
    average_speedbands = {}
    average_speedbands_counts = {}
    for date in datelist:
        times, speedbands = get_avg_speedbands(datadir/date)
        for i in range(len(times)):
            if times[i].hour not in average_speedbands:
                average_speedbands[times[i].hour] = 0
                average_speedbands_counts[times[i].hour] = 0
            average_speedbands[times[i].hour] += speedbands[i]
            average_speedbands_counts[times[i].hour] += 1
    for k, v in average_speedbands.items():
        average_speedbands[k] = v/average_speedbands_counts[k]
    return average_speedbands