# Define the find_npy_files function
import os
import numpy as np
import mne
import shutil
import pandas as pd
from nice.markers import PowerSpectralDensityEstimator
from nice.markers import PowerSpectralDensitySummary
from nice.markers import PowerSpectralDensity
from nice.markers import PermutationEntropy
from nice.markers import SymbolicMutualInformation
from nice.markers import KolmogorovComplexity
from scipy.stats import trim_mean
from sklearn.metrics import roc_curve, roc_auc_score
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from keras import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from keras.utils import to_categorical
from scipy.io import savemat
from keras import utils as np_utils
from keras.models import Sequential
from keras.layers import Conv2D, Dropout, Flatten, Dense
from arl_eegmodels.EEGModels import EEGNet
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping

# Load the dataframe
df = pd.read_csv('E:/Thesis DSS/TD_BRAIN/TDBRAIN-main/TDBRAIN_participants_V2.tsv', sep='\t')
df = df[["participants_ID", "indication"]]

search_label = "HEALTHY"

# Use pandas to filter the dataframe based on the label
filtered_df = df[df["indication"] == search_label]

# Get all the IDs that match the label
matching_ids = filtered_df["participants_ID"].tolist()

# Print the IDs
print("Number of matching entries: ", len(matching_ids))
print("The IDs for label {} are: {}".format(search_label, matching_ids))

original_path = "E:/Thesis DSS/TD_BRAIN/npyfilesEEGNET/open"
new_path = "E:/Thesis DSS/TD_BRAIN/npyfiles/indication/healthy_EO"

# Create the new path if it doesn't already exist
if not os.path.exists(new_path):
    os.makedirs(new_path)

# Iterate over items in the original path
for item in os.listdir(original_path):
    # Check if the item name contains any matching participant ID and does not contain 'BAD'
    if any(match_id in item for match_id in matching_ids) and 'BAD' not in item:
        old_path = os.path.join(original_path, item)
        new_item_path = os.path.join(new_path, item)
        try:
            if os.path.exists(old_path):
                if os.path.isdir(old_path):
                    shutil.move(old_path, new_item_path)
                    print(f"Directory {item} moved to {new_item_path}")
                elif os.path.isfile(old_path):
                    shutil.move(old_path, new_item_path)
                    print(f"File {item} moved to {new_item_path}")
            else:
                print(f"{item} does not exist in the current working directory")
        except shutil.Error as e:
            print(f"Error: {e}")
        except FileExistsError:
            print(f"{new_item_path} already exists")