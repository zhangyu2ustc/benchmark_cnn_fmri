#!/home/yuzhang/tensorflow-py3.6/bin/python3.6

# Author: Yu Zhang
# License: simplified BSD
# coding: utf-8

###default parameter settings

###task-fmri info
TR = 0.72
task_modality = 'MOTOR'
pathdata = "/home/yu/PycharmProjects/HCP_data/aws_s3_HCP1200/"
pathout = "/home/yu/PycharmProjects/HCP_data/temp_res_new2/"

##data pipe
block_dura = 1
nr_thread = 2
buffersize = 4

##model setting
test_sub_num = 0
test_size = 0.2
val_size = 0.1
randseed = 1234

Flag_Simple_RESNET = -1
Flag_CNN_Model = '3d'

batch_size = 1
nepoch = 100
learning_rate = 0.001
learning_rate_decay = 0.9
l2_reg = 0.0001
dataselect_percent = 1.0
Nlabels = 6
test_labels = []

###tensorflow
import os
import tensorflow as tf
num_cores = 24
USE_GPU_CPU =0
num_GPU =0
config = tf.ConfigProto(intra_op_parallelism_threads=num_cores, inter_op_parallelism_threads=2)
##session = tf.Session(config=config)


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Get rid of the AVX, SSE warnings
os.environ["OMP_NUM_THREADS"] = str(num_cores)
##os.environ["KMP_BLOCKTIME"] = "1"
os.environ["KMP_AFFINITY"] = "granularity=thread,compact,1,0"
os.environ["HOROVOD_STALL_CHECK_DISABLE"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
