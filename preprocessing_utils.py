import os
import sys
import numpy as np
import math
import json
import torch
from datetime import datetime, timedelta
from pathlib import Path, PurePath
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt

def process(files_dir, process_dir, overwrite=False):
    '''
    Processes traffic data in json files and generates metadata files and feature npy files
    If overwrite=False, do not process the data if the processed files already exist
    -----------------------------
    :params:
        str files_dir: the directory of the raw dataset
        str process_dir: the directory of the processed output
    -----------------------------
    :returns:
        None
    -----------------------------
    :file outputs:
        adj_u.py: undirected adjacency matrix of road network
        adj_d.py: directed adjacency matrix of road network
        metadata.json: mapping from road index in adjacency matrix to road metadata
        cat2index.json: mapping from road category to integer
        timestaps.json: mapping from timestamp indices to datetime strings
        features/{}.npy: array representation of features extracted from raw traffic data json. one npy file per traffic json.
    '''
    process_dir = Path(process_dir)
    (process_dir / "features").mkdir(parents=True, exist_ok=True)
    
    # check if files are already processed
    adj_u_path = process_dir / "adj_u.npy"
    adj_d_path = process_dir / "adj_d.npy"
    metadata_path = process_dir / "metadata.json"
    cat2index_path = process_dir / "cat2index.json"
    timestamps_path = process_dir / "timestamps.json"
    
    if (not overwrite
            and os.path.isfile(adj_u_path)
            and os.path.isfile(adj_d_path)
            and os.path.isfile(metadata_path)
            and os.path.isfile(cat2index_path)
            and os.path.isfile(timestamps_path)
       ):
        # do not run the function if both overwrite is false and processed files already exist
        return
    
    file_paths = get_ordered_file_path(files_dir)
    
    A_u, A_d, metadata, cat2index = get_adjacency(file_paths[0])
    
    timestamps = {}
    
    for i, data_file_path in enumerate(tqdm(file_paths)):
        
        # Extract featuers
        features = get_features(data_file_path, metadata, cat2index)
        features = features.astype(np.float32)
        npy_path = process_dir / "features" / (str(i)+".npy")
        np.save(npy_path, features)
        
        # Generate timestamps
        fileparts = PurePath(data_file_path).parts
        timestamps[i] = fileparts[-2] + "_" + fileparts[-1].split(".")[0]
        
    # save adjacency
    np.save(adj_u_path, A_u)
    np.save(adj_d_path, A_d)
    
    # save metadata
    with open(metadata_path, 'w') as outfile:
        json.dump(metadata, outfile, sort_keys=True, indent=4)
    
    with open(cat2index_path, 'w') as outfile:
        json.dump(cat2index, outfile, sort_keys=True, indent=4)
        
    with open(timestamps_path, 'w') as outfile:
        json.dump(timestamps, outfile, sort_keys=True, indent=4)
        
def mean_std(dataset_dir, idxs):
    '''
    Calculates the means and standard deviations of features in the dataset
    -----------------------------
    :params:
        str files_dir: the directory of the processed dataset
        list idxs: the indices of the files to be included in the calculation from the 'features' directory
    -----------------------------
    :returns:
        list: means of each feature
        list: standard deviations of each feature
    '''
    dataset_dir = Path(dataset_dir)
    sample = np.load(dataset_dir / "{}.npy".format(idxs[0]))
    means = np.zeros(sample.shape[1])
    stds = np.zeros(sample.shape[1])
    count = 0
    for i in tqdm(idxs):
        data = np.load(dataset_dir / "{}.npy".format(i))
        for features in data:
            count += 1
            for j in range(len(features)):
                new_mean = means[j] + (features[j] - means[j]) / count
                new_std = stds[j] + (features[j] - means[j]) * (features[j] - new_mean)
                means[j] = new_mean
                stds[j] = new_std
    stds = (stds / count) ** 0.5
    return means, stds

def load_metadata(process_dir):
    '''
    Loads the metadata from their respective files
    -----------------------------
    :params:
        str files_dir: the directory of the processed dataset
    -----------------------------
    :returns:
        list (2 dimensions of roads, roads): undirected adjacency matrix of road network
        list (2 dimensions of roads, roads): directed adjacency matrix of road network
        dict: mapping from road index in adjacency matrix to road metadata
        dict: mapping from road category to integer
        dict: mapping from timestamp indices to datetime strings
    '''
    process_dir = Path(process_dir)
    
    adj_u_path = process_dir / "adj_u.npy"
    adj_d_path = process_dir / "adj_d.npy"
    metadata_path = process_dir / "metadata.json"
    cat2index_path = process_dir / "cat2index.json"
    timestamps_path = process_dir / "timestamps.json"
    
    A_u = np.load(adj_u_path)
    A_d = np.load(adj_d_path)
    with open(metadata_path) as json_file:
        metadata = json.load(json_file)
    with open(cat2index_path) as json_file:
        cat2index = json.load(json_file)
    with open(timestamps_path) as json_file:
        timestamps = json.load(json_file)
    return A_u, A_d, metadata, cat2index, timestamps

def generate_samples(idxs, num_timesteps_input, num_timesteps_output, dir_path, features_dir, timestamps):
    '''
    Uses the sliding window method to generate training sample files from a processed dataset.
    -----------------------------
    :params:
         list idxs: incides of the files from the 'features' directory to be used
         int num_timesteps_input: number of input timesteps to the model
         int num_timesteps_output: number of output timesteps from the model (prediction horizon)
         str dir_path: path of the output directory
         str features_dir: path of the 'features' directory from the processing step
         dict timesamps: mapping from timestamp indices to datetime strings
    -----------------------------
    :returns:
        None
    -----------------------------
    :file outputs:
        inputs/{}.npy: array consisting of the concatenated features for input into the model
        targets/{}.npy: array consisting of the prediction target for the model
        input_timestamps.npy: array of timestamps corresponding to each sample in the inputs directory
        output_timestamps.npy: array of timestamps corresponding to each sample in the targets directory
    '''
    dir_path = Path(dir_path)
    features_dir = Path(features_dir)
    (dir_path / "inputs").mkdir(parents=True, exist_ok=True)
    (dir_path / "targets").mkdir(parents=True, exist_ok=True)
    count = 0
    input_timestamps = []
    output_timestamps = []
    for chunk in tqdm(idxs):
        for i in tqdm(range(len(chunk) - num_timesteps_input - num_timesteps_output + 1)):
            sample = []
            target = []
            sample_input_timestamps = []
            sample_output_timestamps = []
            for j in range(num_timesteps_input + num_timesteps_output):
                data = np.load(features_dir / "{}.npy".format(chunk[i+j]))
                key = str(chunk[i+j])
                timestamp = datetime.strptime(timestamps[key], "%a_%b_%d_%Y_%H:%M:%S")
                if j < num_timesteps_input:
                    sample.append(data)
                    sample_input_timestamps.append(timestamp)
                else:
                    target.append(data[:,0])
                    sample_output_timestamps.append(timestamp)
            sample = np.array(sample)
            target = np.array(target)
            np.save(dir_path / "inputs" / "{}.npy".format(count), sample)
            np.save(dir_path / "targets" / "{}.npy".format(count), target)
            input_timestamps.append(sample_input_timestamps)
            output_timestamps.append(sample_output_timestamps)
            count += 1
    input_timestamps = np.array(input_timestamps)
    output_timestamps = np.array(output_timestamps)
    np.save(dir_path / "input_timestamps.npy", input_timestamps)
    np.save(dir_path / "output_timestamps.npy", output_timestamps)
            
def get_symmetric_normalized_adj(A):
    """
    Calculates the symmetrically normalized adjacency matrix.
    Assumes that the adjacency matrix received is undirected,
    adds self loops, and returns D^-1/2 X A X D^-1/2
    -----------------------------
    :params:
         list (2 dimensions of roads, roads) A: undirected adjacency matrix from road network
    -----------------------------
    :returns:
        list (2 dimensions of roads, roads) : symmetrically normalised adjacency matrix
    """
    A = A + np.diag(np.ones(A.shape[0]))
    D_inv = np.diag(np.reciprocal(np.sum(A, axis=1) + 1))
    D_inv_sqrt = np.sqrt(D_inv)
    DAD = np.matmul(D_inv_sqrt, np.matmul(A, D_inv_sqrt))
    return DAD

def get_hybrid_normalized_adj(A):
    """
    Returns the hybrid normalized adjacency matrix.
    Assumes that the adjacency matrix received is directed,
    adds self loops, and returns D_in^-1/2 X A X D_out^-1/2
    -----------------------------
    :params:
         list (2 dimensions of roads, roads) A: directed adjacency matrix from road network
    -----------------------------
    :returns:
        list (2 dimensions of roads, roads): hybrid normalised adjacency matrix
    """
    A = A + np.diag(np.ones(A.shape[0]))
    D_out_inv = np.diag(np.reciprocal(np.sum(A, axis = 1)))
    D_in_inv = np.diag(np.reciprocal(np.sum(A, axis = 0)))
    doi_sqrt = np.sqrt(D_out_inv)
    dii_sqrt = np.sqrt(D_in_inv)
    DAD = np.matmul(dii_sqrt, np.matmul(A, doi_sqrt))
    return DAD

def get_hybrid_combined_normalized_adj(A):
    """
    Returns the hybrid combined normalized adjacency matrix.
    Assumes that the adjacency matrix received is directed,
    adds self loops, and returns D^-1/2 X A X D^-1/2
    -----------------------------
    :params:
         list (2 dimensions of roads, roads) A: directed adjacency matrix from road network
    -----------------------------
    :returns:
        list (2 dimensions of roads, roads): hybrid normalised adjacency matrix
    """
    A = A + np.diag(np.ones(A.shape[0]))
    D_inv = np.diag(np.reciprocal(np.sum(A, axis = 1) + np.sum(A, axis = 0)))
    D_inv_sqrt = np.sqrt(D_inv)
    DAD = np.matmul(D_inv_sqrt, np.matmul(A, D_inv_sqrt))
    return DAD

def get_adjacency(file_path):
    '''
    Generates the Adjacency matrix of the road network, together with other metadata
    -----------------------------
    :params:
        str file_path: the file path of the dataset
    -----------------------------
    :returns:
        list (2 dimensions of roads, roads): Adjacency matrix (undirected)
        list (2 dimensions of roads, roads): Adjacency matrix (directed)
        dict: Metadata (which index in the adjacency matrix corresponds to which road)
        dict: Road category to integer for use as feature
    '''
    with open(file_path, 'r') as traffic_data_file:
        traffic_records = json.load(traffic_data_file)

    # Get start, end, length, and find all road categories. Also remove all non-essential metadata features
    roadcategory_list = []
    nodes_params_dict = {}
    for (i, record) in enumerate(traffic_records):
        lat_long_positions = record['Location'].split()
        record['start_pos'] = ' '. join(lat_long_positions[0:2])
        record['end_pos'] = ' '. join(lat_long_positions[2:4])
        record['length'] = link_length(record['start_pos'], record['end_pos'])
        del record["Location"]
        del record["MaximumSpeed"]
        del record["MinimumSpeed"]
        del record["SpeedBand"]
        
        if record['RoadCategory'] not in roadcategory_list:
            roadcategory_list.append(record['RoadCategory'])
        
        nodes_params_dict[i] = record

    traffic_records.sort(key=lambda x: int(x.get('LinkID')))
    roadcategory_list.sort()
    RoadCat2Index = {}
    for i, cat in enumerate(roadcategory_list):
        RoadCat2Index[cat] = i
    
    # Generating adjacency matrix
    nodes_count = len(nodes_params_dict)
    A_u = np.zeros((nodes_count,nodes_count))
    A_d = np.zeros((nodes_count,nodes_count))
    # Finding the directed edges of the nodes
    for i, i_record in nodes_params_dict.items():
        for j, j_record in nodes_params_dict.items():
            if i_record['end_pos'] == j_record['start_pos']:
                A_u[i,j] = 1
                A_u[j,i] = 1
                A_d[i,j] = 1
                
    return A_u, A_d, nodes_params_dict, RoadCat2Index

def link_length(start_pos, end_pos):
    """
    Calculation of distance between two lat-long geo positions, using Haversine distance
    ------------------------------------
    :params:
        str start_pos: lat & long separated with a space
        str end_pos: lat & long separated with a space
    ------------------------------------
    :returns:
        float: total length of the link
    """
    lat1, lon1 = [float(pos) for pos in start_pos.split()]
    lat2, lon2 = [float(pos) for pos in end_pos.split()]
    radius = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) * math.sin(dlat / 2) +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) * math.sin(dlon / 2))
    d = radius * (2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)))
    return d


def get_features(file_path, metadata, cat2index):
    '''
    Generates a Feature matrix
    Note: Feature Matrix, X, would contain the output speedband as well.
    Positions of Features
        0. SpeedBand
        1. RoadCategory
        2. Length of Link
        3. Day
        4. Hour
    -----------------------------
    :params:
        str file_path: the file path of the dataset
    -----------------------------
    :returns:
        list: Feature matrix
    '''
        
    X = []
    parts = Path(file_path).parts
    hour = parts[-1].split(":")[0]
    day = parts[-2].split("_")[0]
    day2int = {
        "Mon":1,
        "Tue":2,
        "Wed":3,
        "Thu":4,
        "Fri":5,
        "Sat":6,
        "Sun":7
    }
    day_int = day2int[day]
    
    with open(file_path, 'r') as traffic_data_file:
        traffic_records = json.load(traffic_data_file)
        
    traffic_records.sort(key=lambda x: int(x.get('LinkID')))
    for i, record in enumerate(traffic_records):
        features = [record['SpeedBand'],cat2index[record['RoadCategory']],metadata[i]['length'],day_int,hour]
        X.append(features)
    
    return np.array(X)

def get_dates(dir_name):
    '''
    Converts a directory name into a datetime object
    -----------------------------
    :params:
        str dir_name: the name of the directory with the date as the name
    -----------------------------
    :returns:
        datetime: date representation of directory name
    '''
    date_str_format = "%a_%b_%d_%Y"
    my_date = datetime.strptime(dir_name, date_str_format)
    return my_date

def get_day_time(file_name):
    '''
    Converts a file name into a datetime object
    -----------------------------
    :params:
        str file_name: the name of the file with the time as the name
    -----------------------------
    :returns:
        datetime: date representation of file name
    '''
    date_str_format = "%H:%M:%S"
    my_date = datetime.strptime(file_name, date_str_format)
    return my_date

def processed_get_day_time(file_path):
    '''
    Gets the datetime representation of the file from a full file path
    -----------------------------
    :params:
        str file_path: the path of the file with the time as the name
    -----------------------------
    :returns:
        datetime: date representation of file name
    '''
    head_tail = os.path.split(file_path)
    file_name = head_tail[-1].split(".")[0]
    return get_day_time(file_name)

def get_ordered_file_path(dir_path):
    '''
    Sorts the full file paths by date
    -----------------------------
    :params:
        str dir_path: the path of the dir with all of the day subdirectories
    -----------------------------
    :returns:
        list: list of file paths sorted by time
    '''
    sorted_dir_name = sorted(os.listdir(dir_path), key=get_dates)
    sorted_dir = [os.path.join(dir_path, d) for d in sorted_dir_name]

    file_path = []
    for d in sorted_dir:
        list_file = [os.path.join(d,f) for f in os.listdir(d)]
        sorted_list_file = sorted(list_file, key=processed_get_day_time)
        file_path.extend(sorted_list_file)
    return file_path

def timestamp_to_datetime(timestamp):
    '''
    Converts a timestamp to a datetime object
    -----------------------------
    :params:
        str timestamp: timestamp string
    -----------------------------
    :returns:
        datetime: datetime object corresponding to the timestamp
    '''
    return datetime.strptime(timestamp, "%a_%b_%d_%Y_%H:%M:%S")

def is_consecutive(t1, t2, delta):
    '''
    Checks whether two timestamps are within some delta minutes of each other
    -----------------------------
    :params:
        str t1: first timestamp
        str t2: second timestamp
        int delta: target difference in minutes
    -----------------------------
    :returns:
        bool: whether the two timestamps are within delta of one another
    '''
    t1 = timestamp_to_datetime(t1)
    t2 = timestamp_to_datetime(t2)
    delta = timedelta(minutes=delta)
    return t2 < t1 + delta

def find_consecutive_chunks(timestamps, delta):
    '''
    Given the processed timestamps dict, split it into chunks of consecutive timestamps that differ by less than delta minutes
    -----------------------------
    :params:
        dict timestamps: mapping from timestamp indices to datetime strings
        int delta: target difference in minutes
    -----------------------------
    :returns:
        list: list of lists of indices. Timestamps in each sublist are within delta minutes of each other
    '''
    chunks = []
    chunk = []
    for i in range(len(timestamps)):
        if len(chunk) == 0:
            chunk.append(i)
        else:
            if is_consecutive(timestamps[str(i-1)], timestamps[str(i)], delta):
                chunk.append(i)
            else:
                chunks.append(chunk)
                chunk = [i]
    if len(chunk) != 0:
        chunks.append(chunk)
    return chunks

def chunk_len_proportion(chunks):
    '''
    Given a list of timestamp index chunks, find the proportion of each chunk
    -----------------------------
    :params:
        list chunks: list of lists of consecutive timestamps
    -----------------------------
    :returns:
        list: proportion of each chunk of the total dataset by length
    '''
    total_len = 0
    proportion = []
    for chunk in chunks:
        total_len += len(chunk)
    for chunk in chunks:
        proportion.append(len(chunk)/total_len)
    return proportion

def sub_chunk(chunk, proportions):
    '''
    Splits an existing chunk of timetamp indices into sub chunks, according to the specified proportions
    -----------------------------
    :params:
        list chunk: list of consecutive timestamps
        list proportions: list of proportions
    -----------------------------
    :returns:
        list: list of lists split by proportion
    '''
    sub_chunks = []
    sub_chunk = []
    thresholds = [x * len(chunk) for x in proportions]
    for i in range(len(thresholds)-1):
        thresholds[i+1] += thresholds[i]
        
    count = 0
    i = 0
    for i in range(len(chunk)):
        if i < thresholds[count]:
            sub_chunk.append(chunk[i])
        else:
            sub_chunks.append(sub_chunk)
            count += 1
            sub_chunk = []
    if len(sub_chunk) != 0:
        sub_chunks.append(sub_chunk)
    return sub_chunks

def distribution(X, indices):
    '''
    Plots the distribution of the given features across the given samples
    -----------------------------
    :params:
        list X: input sample
        list indices: indices of the features to be plot
    -----------------------------
    :returns:
        None
    '''
    for i in indices:
        values = X[:,i,:].flatten()
        plt.hist(values, bins="auto")
        plt.show()