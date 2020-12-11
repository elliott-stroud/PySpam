# Environment constants

# Entire data set will be used (no matter what sample size is set to)
# if and only if set to 'true'
#USE_ENTIRE_DATASET = 'true'
USE_ENTIRE_DATASET = 'false'

# Number of Ham messages is 6 times this value
# Also, number of Spam messages is 6 times this value
SAMPLE_SIZE = 50

# The root directory of the dataset
DIR_DATASET = '/development/EnronData/'

# The root directory of the dataset
DIR_DATASET = '/development/EnronData/'

# The root directory for the dataset split
DIR_DATA_SPLIT = '/development/EnronData/'

# The directory names for the training and testing splits
DIRNAME_TRAIN = 'trainenron'
DIRNAME_TEST = 'testenron'

# The directories that hold the train and test splits
DIR_DATASET_TRAIN = DIR_DATA_SPLIT + DIRNAME_TRAIN + '/'
DIR_DATASET_TEST = DIR_DATA_SPLIT + DIRNAME_TEST + '/'

# Stop editing from here on out

import os

for dir in [DIR_DATASET, DIR_DATASET_TRAIN, DIR_DATASET_TEST]:
    if not os.path.exists(dir): os.makedirs(dir)
