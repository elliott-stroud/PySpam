# This is code to separate Train Emails and Test Emails.
import os
import shutil
from shutil import copyfile
import random
import datetime

import config
import utils

utils.log('Begin execution')

shutil.rmtree(config.DIR_DATASET_TRAIN)
os.mkdir(config.DIR_DATASET_TRAIN)

shutil.rmtree(config.DIR_DATASET_TEST)
os.mkdir(config.DIR_DATASET_TEST)

utils.log('just deleted contents of two folders')

# input("Press Enter to continue...")
# utils.log('key pressed, now continuing')

# set seed to zero so that results are deterministic & repeatable
random.seed(0)
num_of_train_emails = 0
num_of_test_emails = 0

abc = ['ham', 'spam']
for x in abc:
    for y in range(6):
        path = config.DIR_DATASET+'enron'+str(y+1)+'/'+x+'/'
        list_length = len(os.listdir(path))
        sample_size = config.SAMPLE_SIZE
        if (config.USE_ENTIRE_DATASET =='true'):
            sample_size = list_length
        utils.log(f"[List Length] {path}: {list_length}")
        utils.log(f"[sample_size] : {sample_size}")        
        percent_train = (0.80)*sample_size
        a = []

        for j in range(sample_size):
            a.append(j)
        b = random.sample(a, int(percent_train))
        for i in b:
            copyfile(path+'/'+os.listdir(path)[i], config.DIR_DATASET_TRAIN+os.listdir(path)[i])
            num_of_train_emails += 1

        for v in range(sample_size):
            c = os.listdir(path)[v]
            train_path = config.DIR_DATASET_TRAIN
            h = os.listdir(train_path)
            if c not in h:
                copyfile(path+'/'+c, config.DIR_DATASET_TEST+c)
                num_of_test_emails += 1

utils.log("Number of train emails: {}".format(num_of_train_emails))
utils.log("Number of test emails: {}".format(num_of_test_emails))
utils.log("Size of {dir}: {size}".format(dir=config.DIR_DATASET_TRAIN, size=len(os.listdir(config.DIR_DATASET_TRAIN))))
utils.log("Size of {dir}: {size}".format(dir=config.DIR_DATASET_TEST, size=len(os.listdir(config.DIR_DATASET_TEST))))

utils.log('complete execution')
