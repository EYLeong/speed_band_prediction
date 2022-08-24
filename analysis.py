import torch
import torch.nn as nn

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString
import contextily as ctx
import matplotlib.pyplot as plt

def rmse_per_link(predicted, actual):
    '''
    Calculates the RMSE of the speedbands for each road separately
    -----------------------------
    :params:
        list (3-dimensions of samples, roads, output timesteps) predicted: The predicted speedbands
        list (3-dimensions of samples, roads, output timesteps) actual: The actual speedbands
    -----------------------------
    :returns:
        list (2-dimensions of roads, output timesteps): rmse for each road across the different output timesteps
    '''
    predicted = torch.Tensor(predicted)
    actual = torch.Tensor(actual)
    rmses = []
    for i in range(predicted.shape[1]):
        rmses_timesteps = []
        for j in range(predicted.shape[2]):
            linkPreds = predicted[:,i,j]
            linkActs = actual[:,i,j]
            rmse = nn.MSELoss()(linkPreds, linkActs).sqrt()
            rmses_timesteps.append(rmse.item())
        rmses.append(rmses_timesteps)
    return rmses

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

def plot_geo_performance(metadata, rmses):
    '''
    Generates a geographical map of the roads color coded with their corresponding RMSE
    -----------------------------
    :params:
        dict metadata: Metadata linking road index to other road information
        list rmses: RMSE of each road
    -----------------------------
    :returns:
        None
    '''
    df = pd.DataFrame(metadata).transpose()
    df["RMSE"] = rmses
    loc = df["start_pos"] + " " + df["end_pos"]
    linestrings = loc.apply(loc_to_linestring)
    gdf = gpd.GeoDataFrame(df, geometry=linestrings, crs="EPSG:4326")
    gdf = gdf.to_crs('EPSG:3857')
    fig, ax = plt.subplots(figsize=(10, 10))
    gdf.plot(ax=ax, column="RMSE", legend=True, cmap="OrRd", legend_kwds={'label': 'RMSE'})
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ctx.add_basemap(ax)
    plt.show()
    
def plot_pred_actual(predicted, actual, idx, ts, timestamps, samples):
    '''
    Generates a plot of the predicted vs actual speedbands for a specific road, timestep, and time frame.
    -----------------------------
    :params:
        list (3-dimensions of samples, roads, output timesteps) predicted: The predicted speedbands
        list (3-dimensions of samples, roads, output timesteps) actual: The actual speedbands
        int idx: The index of the road that should be plotted
        list ts: The indices of the timesteps that should be plotted (0 is 5 min, 1 in 10 min, 2 is 15 min, etc)
        list (2-dimensions of samples, output timesteps) timestamps: The timestamps for the speedbands
        tuple (start, end) samples: The range of samples to plot. Indexed respective to the latest timestep
    -----------------------------
    :returns:
        None
    '''
    fig, ax = plt.subplots()
    
    # Plot the actual speedbands based on the latest timestep to allow earlier predictions to be plotted on the same graph
    num_timesteps = len(actual[0][0])
    timestamps = timestamps[samples[0]:samples[1],-1]
    ax.plot(timestamps, actual[samples[0]:samples[1],idx,-1], label="Actual")
    for timestep in ts:
        offset = num_timesteps - 1 - timestep
        ax.plot(timestamps, predicted[samples[0]+offset:samples[1]+offset,idx,timestep], label="Predicted {} min ago".format(5*(timestep+1)))
    ax.set_ylabel("Speedband")
    ax.set_xlabel("Timestep")
    ax.legend()
    plt.gcf().autofmt_xdate()
    plt.show()

def rmse_per_time(predicted, actual, timestamps, timeidx = 0, subpopulation = {}):
    '''
    Calculates the RMSE of the speedbands for each time period
    -----------------------------
    :params:
        list (3-dimensions of samples, roads, output timesteps) predicted: The predicted speedbands
        list (3-dimensions of samples, roads, output timesteps) actual: The actual speedbands
        list (2 dimensions of samples, output timesteps) timestamps: timestamps for the speedbands
        int timeidx: Index of the period of time to be analysed. Date strings are of format DAYOFWEEK_MTH_DAY_YEAR_H:M:S, hence a timeidx of 0 means splitting by day, 4 means splitting by hour, etc.
        dict (timeidx -> set of timeidx values to be included in calculation) subpopulation: Which subpopulations to include as part of the rmse calculation
    -----------------------------
    :returns:
        dict: Dictionary of time period to RMSE
        dict: Dictionary of time period to how many times it is represented in the test set
    '''
    def datetime_with_timeidx(datetime, timeidx):
        weekday = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        if timeidx == 0:
            return weekday[datetime.weekday()]
        if timeidx == 1:
            return datetime.month
        if timeidx == 2:
            return datetime.day
        if timeidx == 3:
            return datetime.year
        if timeidx == 4:
            return datetime.hour
        if timeidx == 5:
            return datetime.minute
        if timeidx == 6:
            return datetime.second
        
    predicted = torch.Tensor(predicted)
    actual = torch.Tensor(actual)
    time_period_counts = {}
    time_period_mses = {}
    for i in range(predicted.shape[0]):
        for j in range(predicted.shape[2]):
            valid = True
            for k, v in subpopulation.items():
                time_period = datetime_with_timeidx(timestamps[i][j], k)
                if time_period not in v:
                    valid = False
                    break
            if valid:
                time_period = datetime_with_timeidx(timestamps[i][j], timeidx)
                if time_period not in time_period_counts:
                    time_period_counts[time_period] = 0
                    time_period_mses[time_period] = 0
                time_period_counts[time_period] += 1
                time_period_mses[time_period] += nn.MSELoss()(predicted[i,:,j], actual[i,:,j])
                    
    for k,v in time_period_mses.items():
        time_period_mses[k] = (v.item()/time_period_counts[k]) ** 0.5
    return time_period_mses, time_period_counts

def rmse_per_category(metadata, rmse_per_link):
    '''
    Calculates the RMSE of the speedbands for each road category
    -----------------------------
    :params:
        dict metadata: Metadata linking road index to other road information
        list rmse_per_link (1-dimension of roads): Overall rmse across all output timesteps for each road link
    -----------------------------
    :returns:
        dict: Dictionary of road category to RMSE
        dict: Dictionary of road category to how many times it is represented in the test set
    '''
    mse_per_cat = {}
    cat_count = {}
    for i in range(len(rmse_per_link)):
        cat = metadata[str(i)]["RoadCategory"]
        if cat not in mse_per_cat:
            mse_per_cat[cat] = 0
            cat_count[cat] = 0
        mse_per_cat[cat] += rmse_per_link[i] ** 2
        cat_count[cat] += 1
    for k,v in mse_per_cat.items():
        mse_per_cat[k] = (v/cat_count[k]) ** 0.5
    return mse_per_cat, cat_count

def plot_rmse(rmses, xlabel):
    '''
    Generates a plot of the RMSE across some axis
    -----------------------------
    :params:
        dict rmses: Dictionary of axis item to RMSE
        string xlabel: Label of the axis
    -----------------------------
    :returns:
        None
    '''
    fig, ax = plt.subplots()
    x = []
    y = []
    for k, v in rmses.items():
        y.append(v)
        x.append(k)
    ax.plot(x, y)
    ax.set_ylabel("RMSE")
    ax.set_xlabel(xlabel)
    ax.set_title("RMSE against "+xlabel)
    plt.show()