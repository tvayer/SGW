#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 10:07:15 2019

@author: vayer
"""
import os,pickle
import argparse
import datetime
import dateutil.tz
import sys
import time


def wait_timeout(proc, seconds):
    """Wait for a process to finish, or raise exception after timeout"""
    start = time.time()
    end = start + seconds
    interval = min(seconds / 1000.0, .25)

    while True:
        result = proc.poll()
        if result is not None:
            return result
        if time.time() >= end:
            raise RuntimeError("Process timed out")
        time.sleep(interval)

def save_obj(obj, name,path='obj/' ):
    try:
        if not os.path.exists(path):
            print('Makedir')
            os.makedirs(path)
    except OSError:
        raise
    with open(path+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name,path):
    with open(path + name, 'rb') as f:
        return pickle.load(f)
        
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
              
def create_log_dir(FLAGS):
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')

    log_dir = FLAGS.log_dir + "/" + str(sys.argv[0][:-3]) + "_" + timestamp 
    print(log_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # save command line arguments
    with open(log_dir + "/hyperparameters_" + timestamp + ".csv", "w") as f:
        for arg in FLAGS.__dict__:
            f.write(arg + "," + str(FLAGS.__dict__[arg]) + "\n")

    return log_dir