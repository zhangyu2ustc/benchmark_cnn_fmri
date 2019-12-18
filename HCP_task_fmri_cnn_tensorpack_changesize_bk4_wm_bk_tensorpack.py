#!/home/yuzhang/tensorflow-py3.6/bin/python3.6

# Author: Yu Zhang
# License: simplified BSD
# coding: utf-8

## ps | grep python; pkill python

from pathlib import Path
import glob
import itertools
import lmdb
import h5py
import os
import sys
import time
import datetime
import shutil
import numpy as np
import pandas as pd
import nibabel as nib
from collections import Counter
from operator import itemgetter

import gc
import argparse
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Get rid of the AVX, SSE warnings
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

from nilearn import image, masking
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score, train_test_split,ShuffleSplit


import tensorflow as tf
from tensorflow.python.keras.layers import *
from tensorpack import dataflow
#from tensorpack.utils.serialize import dumps, loads

from tensorpack import InputDesc, SyncMultiGPUTrainer, SyncMultiGPUTrainerParameterServer
from tensorpack.callbacks import *
from tensorpack.utils import logger
from tensorpack.contrib.keras import KerasModel
from tensorpack.utils.gpu import get_num_gpu
from tensorpack.tfutils.common import get_tf_version_tuple

from tensorflow.python.keras.models import Model
from tensorflow.python.keras import regularizers
from tensorflow.python.keras import backend as K

import tensorflow.keras as keras
#import horovod.tensorflow.keras as hvd

#hvd.init()
#print('Hello from worker #{} of {}'.format(hvd.rank(), hvd.size()))

USE_GPU_CPU =1
num_GPU =2
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = "0,1" #str(hvd.local_rank())

#session = tf.Session(config=config)
#K.set_session(session)
'''
if (hvd.rank() == 0):  # Only print on worker 0
    print_summary = 1
    verbose = 1
    os.system("lscpu")
    #os.system("uname -a")
    print("TensorFlow version: {}".format(tf.__version__))
    print("Intel MKL-DNN is enabled = {}\n\n".format(tf.pywrap_tensorflow.IsMklEnabled()))
    #print("Keras API version: {}".format(K.__version__))

else:  # Don't print on workers > 0
    print_summary = 0
    verbose = 0
'''
#########################################################
global TR, Flag_CNN_Model, Flag_Simple_RESNET, img_resize, Nlabels
TOTAL_BATCH_SIZE = 128
TR = 0.72
Flag_CNN_Model = '3d'
Flag_Simple_RESNET = -1
learning_rate_decay = 0.9
l2_reg = 0.0001
dataselect_percent = 1.0
nb_class = 5
test_labels = []
##############################
'''
pathdata = Path('/mnt/lustrefs/mcgill/HCP/aws-s3_copy_041519/')
pathout = "/mnt/lustrefs/mcgill/HCP/temp_res_new/"

pathdata = Path('/data/HCP/aws-s3_copy_041519/')
pathout = "/data/HCP/temp_res_new/"
'''
pathdata = Path('/data/cisl/project/zenith_cpu/HCP/aws-s3_copy_041519/')
pathout = "/data/cisl/project/zenith_cpu/HCP/temp_res_new/"

###global steps_Train, steps_Val, steps_Test
steps_Train = 100000 #steps
steps_Val = 1000 #steps
steps_Test = 1000 #steps
##dataselect_percent = 0.01 #round(block_dura/Trial_Num

###################################################
def bulid_dict_task_modularity(modality):
    ##build the dict for different subtypes of events under different modalities
    motor_task_con = {"rf": "footR_mot",
                      "lf": "footL_mot",
                      "rh": "handR_mot",
                      "lh": "handL_mot",
                      "t": "tongue_mot"}
    '''
    lang_task_con =  {"present_math":  "math_lang",
                      "question_math": "math_lang",
                      "response_math": "math_lang",
                      "present_story":  "story_lang",
                      "question_story": "story_lang" ,
                      "response_story": "story_lang"}
    '''
    lang_task_con =  {"math":  "math_lang",
                      "story": "story_lang"}
    emotion_task_con={"fear": "fear_emo",
                      "neut": "non_emo"}
    gambl_task_con = {"win_event":  "win_gamb",
                      "loss_event": "loss_gamb",
                      "neut_event": "non_gamb"}
    reson_task_con = {"match":    "match_reson",
                      "relation": "relat_reson"}
    social_task_con ={"mental": "mental_soc",
                      "rnd":  "random_soc"}
    wm_task_con   =  {"2bk_body":   "body2b_wm",
                      "2bk_faces":  "face2b_wm",
                      "2bk_places": "place2b_wm",
                      "2bk_tools":  "tool2b_wm",
                      "0bk_body":   "body0b_wm",
                      "0bk_faces":  "face0b_wm",
                      "0bk_places": "place0b_wm",
                      "0bk_tools":  "tool0b_wm"}

    dicts = [motor_task_con, lang_task_con, emotion_task_con, gambl_task_con, reson_task_con, social_task_con, wm_task_con]
    from collections import defaultdict
    all_task_con = defaultdict(list)  # uses set to avoid duplicates
    for d in dicts:
        for k, v in d.items():
            all_task_con[k].append(v)  ## all_task_con[k]=v to remove the list []

    mod_chosen = modality[:3].lower().strip()
    mod_choices = {'mot': 'MOTOR',
                   'lan': 'LANGUAGE',
                   'emo': 'EMOTION',
                   'gam': 'GAMBLING',
                   'rel': 'RELATIONAL',
                   'soc': 'SOCIAL',
                   'wm': 'WM',
                   'all': 'ALLTasks'}
    task_choices = {'mot': motor_task_con,
                    'lan': lang_task_con,
                    'emo': emotion_task_con,
                    'gam': gambl_task_con,
                    'rel': reson_task_con,
                    'soc': social_task_con,
                    'wm': wm_task_con,
                    'all': all_task_con}

    modality = mod_choices.get(mod_chosen, 'default')
    task_contrasts = task_choices.get(mod_chosen, 'default')
    return task_contrasts, modality


def load_fmri_data(pathdata, modality=None, confound_name=None):
    ###fMRI decoding: using event signals instead of activation pattern from glm
    ##collect task-fMRI signals

    if not modality:
        modality = 'MOTOR'  # 'MOTOR'

    subjects = []
    fmri_files = []
    confound_files = []
    for fmri_file in sorted(pathdata.glob('*/FMRI/tfMRI_' + modality + '_??/*tfMRI_' + modality + '_??.nii.gz')):
        fmri_files.append(str(fmri_file))
        confound = os.path.join(os.path.dirname(fmri_file), 'Movement_Regressors.txt')
        if os.path.isfile(confound): confound_files.append(str(confound))

        subname_info = Path(os.path.dirname(fmri_file)).parts[-3]
        taskname_info = Path(os.path.dirname(fmri_file)).parts[-1].split('_')
        subjects.append('_'.join((subname_info, taskname_info[-2], taskname_info[-1])))

    '''
    for confound in sorted(pathdata.glob('*/FMRI/tfMRI_'+modality+'_??/*Movement_Regressors.txt')):
        confound_files.append(str(confound))
    '''
    print('%d subjects included in the dataset' % len(fmri_files))
    return fmri_files, confound_files, subjects


def load_event_files(fmri_files, confound_files, ev_filename=None):
    ###collect the event design files
    tc_matrix = nib.load(fmri_files[0])
    Subject_Num = len(fmri_files)
    Trial_Num = tc_matrix.shape[-1]
    ##print("Data samples including %d subjects with %d trials" % (Subject_Num, Trial_Num))

    EVS_files = []
    subj = 0
    ###adjust the code after changing to the new folder
    for ev in sorted(pathdata.glob('*/FMRI/tfMRI_' + modality + '_??/*combined_events_spm_' + modality + '.csv')):
        ###remove fmri files if the event design is missing
        while os.path.dirname(fmri_files[subj]) < os.path.dirname(str(ev)):
            fmri_files[subj] = []
            confound_files[subj] = []
            subj += 1
            if subj > Subject_Num: break
        if os.path.dirname(fmri_files[subj]) == os.path.dirname(str(ev)):
            EVS_files.append(str(ev))
            subj += 1

    fmri_files = list(filter(None, fmri_files))
    confound_files = list(filter(None, confound_files))
    if len(EVS_files) != len(fmri_files):
        print(len(fmri_files), len(confound_files), len(EVS_files))
        print('Miss-matching number of subjects between event:{} and fmri:{} files'.format(len(EVS_files), len(fmri_files)))

    ################################
    ###loading all event designs
    if not ev_filename:
        ev_filename = "_event_labels_1200R_LR_RL_new.txt"

    events_all_subjects_file = pathout + modality + ev_filename
    if os.path.isfile(events_all_subjects_file):
        trial_infos = pd.read_csv(EVS_files[0], sep="\t", encoding="utf8", header=None, names=['onset', 'duration', 'rep', 'task'])
        Duras = np.ceil((trial_infos.duration / TR)).astype(int)  # (trial_infos.duration/TR).astype(int)

        subjects_trial_labels = pd.read_csv(events_all_subjects_file, sep="\t", encoding="utf8")
        ###print(subjects_trial_labels.keys())

        try:
            label_matrix = subjects_trial_labels['label_data'].values
            # print(label_matrix[0],label_matrix[1])
            # xx = label_matrix[0].split(",")
            subjects_trial_label_matrix = []
            for subi in range(len(label_matrix)):
                xx = [x.replace("['", "").replace("']", "") for x in label_matrix[subi].split("', '")]
                subjects_trial_label_matrix.append(xx)
            subjects_trial_label_matrix = pd.DataFrame(data=(subjects_trial_label_matrix))
        except:
            print('only extracting {} trials from event design'.format(Trial_Num))
            subjects_trial_label_matrix = subjects_trial_labels.loc[:, 'trial1':'trial' + str(Trial_Num)]

        ##subjects_trial_label_matrix = subjects_trial_labels.values.tolist()
        trialID = subjects_trial_labels['trialID']
        sub_name = subjects_trial_labels['subject'].tolist()
        coding_direct = subjects_trial_labels['coding']
        trial_class_counts = Counter(list(subjects_trial_label_matrix.iloc[0, :]))
        print(trial_class_counts)
        print('Collecting trial info from file:', events_all_subjects_file)
        print(np.array(subjects_trial_label_matrix).shape, len(sub_name), len(np.unique(sub_name)), len(coding_direct))
    else:
        print('Loading trial info for each task-fmri file and save to csv file:', events_all_subjects_file)
        subjects_trial_label_matrix = []
        sub_name = []
        coding_direct = []
        for subj in np.arange(Subject_Num):
            pathsub = Path(os.path.dirname(EVS_files[subj]))
            # sub_name.append(pathsub.parts[-3])
            ###adjust the code after changing to the new folder
            sub_name.append(str(os.path.basename(EVS_files[subj]).split('_')[0]))
            coding_direct.append(pathsub.parts[-1].split('_')[-1])

            ##trial info in volume
            trial_infos = pd.read_csv(EVS_files[subj], sep="\t", encoding="utf8", header=None, names=['onset', 'duration', 'rep', 'task'])
            Onsets = np.ceil((trial_infos.onset / TR)).astype(int)  # (trial_infos.onset/TR).astype(int)
            Duras = np.ceil((trial_infos.duration / TR)).astype(int)  # (trial_infos.duration/TR).astype(int)
            Movetypes = trial_infos.task

            labels = ["rest"] * Trial_Num;
            trialID = [0] * Trial_Num;
            tid = 1
            for start, dur, move in zip(Onsets, Duras, Movetypes):
                for ti in range(start - 1, start + dur):
                    labels[ti] = task_contrasts[move]
                    trialID[ti] = tid
                tid += 1
            subjects_trial_label_matrix.append(labels)

        ##subjects_trial_label_matrix = np.array(subjects_trial_label_matrix)
        # print(np.array(subjects_trial_label_matrix[0]))
        subjects_trial_labels = pd.DataFrame(data=np.array(subjects_trial_label_matrix), columns=['trial' + str(i + 1) for i in range(Trial_Num)])
        subjects_trial_labels['trialID'] = tid
        subjects_trial_labels['subject'] = sub_name
        subjects_trial_labels['coding'] = coding_direct
        subjects_trial_labels.keys()
        subjects_trial_label_matrix = pd.DataFrame(data=np.array(subjects_trial_label_matrix),
                                                   columns=['trial' + str(i + 1) for i in range(Trial_Num)])

        ##save the labels
        subjects_trial_labels.to_csv(events_all_subjects_file, sep='\t', encoding='utf-8', index=False)

    block_dura = sum(Duras)
    return subjects_trial_label_matrix, sub_name, block_dura


def preclean_data_matrix_for_shape_match(fmri_files, confound_files, label_matrix, ev_sub_name, fmri_sub_name=None):
    ###"Pre-clean the fmri and event data to make sure the matching shapes between two arrays!"

    #####################sort list of files
    print('Sort both fmri and event files into the same order!')
    print(np.array(fmri_files).shape, np.array(label_matrix).shape)

    fmrifile_index, fmri_sub_name_sorted = zip(*sorted(enumerate(fmri_sub_name), key=itemgetter(1)))
    fmri_matrix_sorted = [fmri_files[ind] for ind in fmrifile_index]
    confound_matrix_sorted = [confound_files[ind] for ind in fmrifile_index]
    ev_index, ev_sub_name_sorted = zip(*sorted(enumerate(ev_sub_name), key=itemgetter(1)))
    ##subjects_trial_label_matrix_sorted = [subjects_trial_label_matrix.iloc[ind] for ind in ev_index]
    label_matrix_sorted = [list(filter(None, label_matrix.iloc[ind])) for ind in ev_index]
    fmri_sub_name_sorted = list(fmri_sub_name_sorted)
    ev_sub_name_sorted = list(ev_sub_name_sorted)

    ####check matching of filenames
    for subcount, ev in enumerate(ev_sub_name_sorted):
        evfile_mask = pd.Series(ev).isin(fmri_sub_name_sorted)
        if not evfile_mask[0]:
            label_matrix_sorted[subcount] = []
            ev_sub_name_sorted[subcount] = []
            print("Remove event file: {} from the list!! Remaining {} event files".format(ev, len(ev_sub_name_sorted)))
    label_matrix_sorted = list(filter(None, label_matrix_sorted))
    ev_sub_name_sorted = list(filter(None, ev_sub_name_sorted))
    print("Remaining {}/{} event files".format(len(ev_sub_name_sorted), np.array(label_matrix_sorted).shape))

    for subcount, fmri_file in enumerate(fmri_sub_name_sorted):
        fmrifile_mask = pd.Series(fmri_file).isin(ev_sub_name_sorted)
        if not fmrifile_mask[0]:
            del fmri_matrix_sorted[subcount]
            del confound_matrix_sorted[subcount]
            fmri_sub_name_sorted.remove(fmri_file)
            print("Remove fmri file: {} from the list!! Remaining {} fmri files".format(fmri_file, len(fmri_sub_name_sorted)))
        else:
            subcount += 1

    label_matrix_sorted = pd.DataFrame(data=(label_matrix_sorted))
    print('New shapes of fmri-data-matrix and trial-label-matrix after matching!')
    print(np.array(fmri_matrix_sorted).shape, np.array(label_matrix_sorted).shape)
    if len(fmri_sub_name_sorted) != len(ev_sub_name_sorted):
        print('Warning: Mis-matching subjects list between fmri-data-matrix and trial-label-matrix')
        print(np.array(fmri_matrix_sorted).shape, np.array(label_matrix_sorted).shape)

    for subj in range(min(len(fmri_sub_name_sorted), len(ev_sub_name_sorted))):
        if fmri_sub_name_sorted[subj] != ev_sub_name_sorted[subj]:
            print('miss-matching of subject order! {}/{}'.format(fmri_sub_name_sorted[subj], ev_sub_name_sorted[subj]))
    print('Done matching data shapes:', len(fmri_matrix_sorted), len(confound_matrix_sorted), np.array(label_matrix_sorted).shape)

    return fmri_matrix_sorted, confound_matrix_sorted, label_matrix_sorted, fmri_sub_name


from sklearn.base import TransformerMixin
class NDStandardScaler(TransformerMixin):
    def __init__(self, axis=0,**kwargs):
        self._scaler = preprocessing.StandardScaler(copy=True, **kwargs)
        self._orig_shape = None
        self._axis = axis

    def fit(self, X, **kwargs):
        X = np.array(X)
        # Save the original shape to reshape the flattened X later
        # back to its original shape
        if len(X.shape) > 1:
            self._orig_shape = X.shape[1:]
        X = self._flatten(X)
        X = self._scaler.fit(X, **kwargs)
        return self

    def transform(self, X, **kwargs):
        X = np.array(X)
        X = self._flatten(X)
        X = self._scaler.transform(X, **kwargs)
        X = self._reshape(X)
        return X

    def fit_transform(self, X, **kwargs):
        X = np.array(X)
        # Save the original shape to reshape the flattened X later
        # back to its original shape
        if len(X.shape) > 1:
            self._orig_shape = X.shape[1:]
        X = self._flatten(X)
        X = self._scaler.fit_transform(X, **kwargs)
        X = self._reshape(X)
        return X

    def _flatten(self, X):
        # Reshape X to <= 2 dimensions
        if len(X.shape) > 1:
            n_dims = np.prod(self._orig_shape)
            X = X.reshape(-1, n_dims)
        return X

    def _reshape(self, X):
        # Reshape X back to it's original shape
        if len(X.shape) > 1:
            X = X.reshape(-1, *self._orig_shape)
        return X


#############################
#######################################
####tensorpack: multithread
class gen_fmri_file(dataflow.DataFlow):
    """ Iterate through fmri filenames, confound filenames and labels
    """
    def __init__(self, fmri_files,confound_files, label_matrix,data_type='train'):
        assert (len(fmri_files) == len(confound_files))
        # self.data=zip(fmri_files,confound_files)
        self.fmri_files = fmri_files
        self.confound_files = confound_files
        self.label_matrix = label_matrix

        self.data_type=data_type


    def size(self):
        return len(self.fmri_files)
        #return int(1e12)
        #split_num=int(len(self.fmri_files)*0.8)
        #if self.data_type=='train':
        #    return split_num
        #else:
        #    return len(self.fmri_files)-split_num

    def get_data(self):
        assert self.data_type in ['train', 'val', 'test']

        split_num=int(len(self.fmri_files))
        while True:
            #rand_pos=np.random.choice(split_num,1,replace=False)[0]
            rand_pos = np.random.randint(0, split_num)
            yield self.fmri_files[rand_pos],self.confound_files[rand_pos],self.label_matrix.iloc[rand_pos]
        #for pos_ in range(split_num):
        #    yield self.fmri_files[pos_],self.confound_files[pos_],self.label_matrix.iloc[pos_]

class split_samples(dataflow.DataFlow):
    """ Iterate through fmri filenames, confound filenames and labels
    """
    def __init__(self, ds):
        self.ds=ds
        self.Subjects_Num=ds.size()

    def size(self):
        #return 91*284
        ds_data_shape = self.data_info()
        self.Trial_Num = ds_data_shape[0]
        self.Block_dura = ds_data_shape[-1]

        return self.Subjects_Num * self.Trial_Num

    def data_info(self):
        ##for data in self.ds.get_data():
        data = self.ds.get_data().__next__()
        print('fmri/label data shape:',data[0].shape,data[1].shape)
        return data[0].shape

    def get_data(self):
        for data in self.ds.get_data():
            for i in range(data[1].shape[0]):
                ####yield data[0][i],data[1][i]
                yield data[0][i].astype('float32',casting='same_kind'),data[1][i]


#################################################################
#########reshape fmri and label data into blocks
#######change: use blocks instead of trials as input

def map_load_fmri_image_3d(dp, target_name,hrf_delay=0):
    fmri_file = dp[0]
    confound_file = dp[1]
    label_trials = dp[2]
    '''
    ###remove confound effects
    confound = np.loadtxt(confound_file)
    ###using orthogonal matrix
    normed_matrix = preprocessing.normalize(confound, axis=0, norm='l2')
    confound, R = np.linalg.qr(normed_matrix)
    try:
        mask_img = masking.compute_epi_mask(fmri_file)
        fmri_data_clean = image.clean_img(fmri_file, detrend=False, standardize=True, confounds=confound, t_r=TR,ensure_finite=True,mask_img=mask_img)
    except:
        print('Error loading fmri file. Check fmri data first: %s' % fmri_file)
        mask_img = masking.compute_epi_mask(fmri_file)
        fmri_data_clean = image.clean_img(fmri_file, detrend=False, standardize=True, confounds=confound, t_r=TR,ensure_finite=True,mask_img=mask_img)
    '''
    fmri_data_clean = fmri_file
    try:
        ##resample image into a smaller size to save memory
        target_affine = np.diag((img_resize, img_resize, img_resize))
        fmri_data_clean_resample = image.resample_img(fmri_data_clean, target_affine=target_affine)
    except:
        fmri_data_clean_resample = fmri_data_clean

    ##pre-select task types
    if hrf_delay > 0:
        label_trials = np.roll(label_trials, hrf_delay)
    trial_mask = pd.Series(label_trials).isin(target_name)  ##['hand', 'foot','tongue']
    #fmri_data_cnn = image.index_img(fmri_data_clean_resample, np.where(trial_mask)[0]).get_data()
    fmri_data_cnn = image.index_img(fmri_data_clean_resample, np.where(trial_mask)[0]).get_data().astype('float32',casting='same_kind')
    img_rows, img_cols, img_deps = fmri_data_cnn.shape[:-1]


    ###use each slice along z-axis as one sample
    label_data_trial = np.array(label_trials.loc[trial_mask])
    le = preprocessing.LabelEncoder()
    le.fit(target_name)
    label_data_cnn = le.transform(label_data_trial)  ##np_utils.to_categorical(): convert label vector to matrix

    fmri_data_cnn_test = np.transpose(fmri_data_cnn, (3, 0, 1, 2))
    fmri_data_cnn_test = NDStandardScaler().fit_transform(fmri_data_cnn_test)
    label_data_cnn_test = label_data_cnn.flatten()
    ##print(fmri_file, fmri_data_cnn_test.shape, label_data_cnn_test.shape)
    fmri_data_cnn = None

    return fmri_data_cnn_test, label_data_cnn_test

def map_load_fmri_image_3d_block(dp, target_name,block_dura=1,hrf_delay=0):
    fmri_file = dp[0]
    confound_file = dp[1]
    label_trials = dp[2]
    '''
    ###remove confound effects
    confound = np.loadtxt(confound_file)
    ###using orthogonal matrix
    normed_matrix = preprocessing.normalize(confound, axis=0, norm='l2')
    confound, R = np.linalg.qr(normed_matrix)
    mask_img = masking.compute_epi_mask(fmri_file)
    fmri_data_clean = image.clean_img(fmri_file, detrend=False, standardize=True, confounds=confound, t_r=TR,ensure_finite=True,mask_img=mask_img)
    '''
    fmri_data_clean = fmri_file
    try:
        ##resample image into a smaller size to save memory
        target_affine = np.diag((img_resize, img_resize, img_resize))
        fmri_data_clean_resample = image.resample_img(fmri_data_clean, target_affine=target_affine)
    except:
        fmri_data_clean_resample = fmri_data_clean

    ##pre-select task types
    if hrf_delay > 0:
        label_trials = np.roll(label_trials, hrf_delay)
    trial_mask = pd.Series(label_trials).isin(target_name)  ##['hand', 'foot','tongue']
    #fmri_data_cnn = image.index_img(fmri_data_clean_resample, np.where(trial_mask)[0]).get_data()
    fmri_data_cnn = image.index_img(fmri_data_clean_resample, np.where(trial_mask)[0]).get_data().astype('float32',casting='same_kind')
    img_rows, img_cols, img_deps = fmri_data_cnn.shape[:-1]
    '''
    #min_max_scaler = preprocessing.MinMaxScaler()
    #fmri_data_cnn = min_max_scaler.fit_transform(fmri_data_cnn.reshape(np.prod(fmri_data_cnn.shape[:-1]),fmri_data_cnn.shape[-1]))
    fmri_data_cnn = fmri_data_cnn.reshape(np.prod(fmri_data_cnn.shape[:-1]),fmri_data_cnn.shape[-1])
    fmri_data_cnn = preprocessing.scale(fmri_data_cnn/np.max(fmri_data_cnn), axis=1).astype('float32',casting='same_kind')
    fmri_data_cnn[np.isnan(fmri_data_cnn)] = 0
    fmri_data_cnn[np.isinf(fmri_data_cnn)] = 0
    fmri_data_cnn = fmri_data_cnn.reshape((img_rows, img_cols, img_deps,fmri_data_cnn.shape[-1]))
    '''

    ###use each slice along z-axis as one sample
    label_data_trial = np.array(label_trials.loc[trial_mask])
    le = preprocessing.LabelEncoder()
    le.fit(target_name)
    label_data_int = le.transform(label_data_trial)

    ##cut the trials
    chunks = int(np.floor(len(label_data_trial) / block_dura))
    label_data_trial_block = np.array(np.split(label_data_trial, np.where(np.diff(label_data_int))[0] + 1))
    fmri_data_cnn_block = np.array_split(fmri_data_cnn, np.where(np.diff(label_data_int))[0] + 1, axis=3)
    trial_lengths = [label_data_trial_block[ii].shape[0] for ii in range(label_data_trial_block.shape[0])]

    # ulabel = [np.unique(x) for x in label_data_trial_block]
    # print("After cutting: unique values for each block of trials %s with %d blocks" % (np.array(ulabel), len(ulabel)))
    if label_data_trial_block.shape[0] != chunks or len(np.unique(trial_lengths)) > 1:
        # print("Wrong cutting of event data...")
        # print("Should have %d block-trials but only found %d cuts" % (chunks, label_data_trial_block.shape[0]))
        try:
            label_data_trial_block = np.array(np.split(label_data_trial, chunks))
            fmri_data_cnn_block = np.array_split(fmri_data_cnn, chunks, axis=3)
            fmri_data_cnn = None
        except:
            ##reshape based on both trial continous and block_dura
            fmri_data_cnn = None
            label_data_trial_block_new = []
            fmri_data_cnn_block_new = []
            chunks = 0
            blocks_num = label_data_trial_block.shape[0]
            for bi in range(blocks_num):
                trial_num_used = len(label_data_trial_block[bi]) // block_dura * block_dura
                chunks += trial_num_used // block_dura
                label_data_trial_block_new.append(np.array(np.split(label_data_trial_block[bi][:trial_num_used], trial_num_used // block_dura)))
                fmri_data_cnn_block_new.append(
                    np.array(np.split(fmri_data_cnn_block[bi][:, :, :, :trial_num_used], trial_num_used // block_dura, axis=-1)))
            label_data_trial_block = np.concatenate([label_data_trial_block_new[ii] for ii in range(blocks_num)], axis=0)
            fmri_data_cnn_block = np.concatenate([fmri_data_cnn_block_new[ii] for ii in range(blocks_num)], axis=0)
            fmri_data_cnn_block_new = None
        # ulabel = [np.unique(x) for x in label_data_trial_block]
        # print("Adjust the cutting: unique values for each block of trials %s with %d blocks" % (np.array(ulabel), len(ulabel)))

    label_data = np.array([label_data_trial_block[i][:block_dura] for i in range(chunks)])
    label_data_cnn_test = le.transform(label_data[:, 0]).flatten()
    fmri_data_cnn_test = np.array([fmri_data_cnn_block[i][:, :, :, :block_dura].astype('float32') for i in range(chunks)])
    ##fmri_data_cnn_test = NDStandardScaler().fit_transform(fmri_data_cnn_test)
    fmri_data_cnn_block = None
    ##print(fmri_file, fmri_data_cnn_test.shape, label_data_cnn_test.shape)

    return fmri_data_cnn_test, label_data_cnn_test

def data_pipe_3dcnn_block(fmri_files, confound_files, label_matrix, target_name=None, flag_cnn='3d', block_dura=1,
                          batch_size=32,data_type='train',nr_thread=8, buffer_size=10):
    assert data_type in ['train', 'val', 'test']
    assert flag_cnn in ['3d', '2d']
    assert fmri_files is not None
    isTrain = data_type == 'train'
    isVal = data_type == 'val'

    print('\n\nGenerating dataflow for %s datasets \n' % data_type)

    buffer_size = int(min(len(fmri_files), buffer_size))
    nr_thread = int(min(len(fmri_files), nr_thread))

    ds0 = gen_fmri_file(fmri_files, confound_files, label_matrix, data_type=data_type)
    print('dataflowSize is ' + str(ds0.size()))
    print('Loading data using %d threads with %d buffer_size ... \n' % (nr_thread, buffer_size))

    if target_name is None:
        target_name = np.unique(label_matrix)
    ##Subject_Num, Trial_Num = np.array(label_matrix).shape

    ####running the model
    start_time = time.clock()
    if flag_cnn == '2d':
        ds1 = dataflow.MultiThreadMapData(
            ds0, nr_thread=nr_thread,
            map_func=lambda dp: map_load_fmri_image_block(dp, target_name,block_dura=block_dura),
            buffer_size=buffer_size,
            strict=True)
    elif flag_cnn == '3d':
        ds1 = dataflow.MultiThreadMapData(
            ds0, nr_thread=nr_thread,
            map_func=lambda dp: map_load_fmri_image_3d_block(dp, target_name,block_dura=block_dura),
            buffer_size=buffer_size,
            strict=True)
    ##dataflow.TestDataSpeed(ds1, size=10000).start()

    ds1 = dataflow.PrefetchData(ds1, buffer_size, 1)

    ds1 = split_samples(ds1)
    print('prefetch dataflowSize is ' + str(ds1.size()))
    '''
    if isTrain:
        print('%d #Trials/Samples per subject with %d channels in tc' % (ds1.Trial_Num, ds1.Block_dura))
        Trial_Num = ds1.Trial_Num
        #ds1 = dataflow.LocallyShuffleData(ds1, buffer_size=ds1.size() * buffer_size)
        ds1 = dataflow.LocallyShuffleData(ds1, buffer_size=Trial_Num * buffer_size, shuffle_interval=Trial_Num * buffer_size) #//2
    '''
    ds1 = dataflow.BatchData(ds1, batch_size=batch_size, remainder=True)
    print('Time Usage of loading mini-batch {} of data using seconds: {} \n'.format(batch_size, time.clock() - start_time))

    ds1 = dataflow.PrefetchDataZMQ(ds1, nr_proc=nr_thread*2) ##1
    ##dataflow.TestDataSpeed(ds1, size=10000).start()
    #ds1._reset_once()
    ##ds1.reset_state()
    '''
    for df in ds1.get_data():
        if flag_cnn == '2d':
            yield (df[0].astype('float32'),to_categorical(df[1], len(target_name)))
        elif flag_cnn == '3d':
            yield (df[0].astype('float32'),to_categorical(df[1], len(target_name)))
    '''
    return ds1
###end of tensorpack: multithread
##############################################################

####################################################################
def bn(x, bn_axis=-1, zero_init=False):
    if K.image_data_format() == 'channels_first':
        bn_axis = 1
    elif K.image_data_format() == 'channels_last':
        bn_axis = -1
    return BatchNormalization(axis=bn_axis, fused=True,momentum=0.9, epsilon=1e-5,gamma_initializer='zeros' if zero_init else 'ones')(x)

def conv2d(x, filters, kernel, strides=1, name=None):
    return Conv2D(filters, kernel, strides=strides, use_bias=False, padding='same',
                  kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(l2_reg))(x)

def conv3d(x, filters, kernel, strides=1, name=None):
    return Conv3D(filters, kernel,strides=strides, use_bias=False, padding='same',
                  kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(l2_reg))(x)

def conv(x, filters, kernel, flag_2d3d='2d', strides=1):
    if flag_2d3d == '2d':
        return conv2d(x, filters, kernel, strides=strides)
    elif flag_2d3d == '3d':
        return conv3d(x, filters, kernel, strides=strides)

def identity_block(input_tensor, filters, kernel_size, flag_2d3d='2d'):
    filters1, filters2, filters3 = filters

    x = conv(input_tensor, filters1, 1, flag_2d3d)
    x = bn(x)
    x = Activation('relu')(x)

    x = conv(x, filters2, kernel_size, flag_2d3d)
    x = bn(x)
    x = Activation('relu')(x)

    x = conv(x, filters3, 1, flag_2d3d)
    x = bn(x, zero_init=True)

    x = Add()([x, input_tensor])
    x = Activation('relu')(x)
    return x

def conv_block(input_tensor, filters, kernel_size, strides=2,flag_2d3d='2d'):
    filters1, filters2, filters3 = filters

    x = conv(input_tensor, filters1, 1, flag_2d3d)
    x = bn(x)
    x = Activation('relu')(x)

    x = conv(x, filters2, kernel_size, flag_2d3d, strides=strides)
    x = bn(x)
    x = Activation('relu')(x)

    x = conv(x, filters3, 1, flag_2d3d)
    x = bn(x, zero_init=True)

    shortcut = conv(input_tensor,filters3, 1, flag_2d3d, strides=strides)
    shortcut = bn(shortcut)

    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x

def build_model_resnet50(image, Nlabels=nb_class, filters=16, convsize=3, convsize2=5, poolsize=2, hidden_size=256):
    #############build resnet50 of 2d and 3d conv

    global Flag_CNN_Model, Flag_Simple_RESNET

    input0 = Input(tensor=image)

    x = conv(input0, filters, convsize2, Flag_CNN_Model, strides=2)
    x = bn(x)
    x = Activation('relu')(x)
    try:
        img_resize   ## using either maxpooling or img_resampling
    except:
        if Flag_CNN_Model == '2d':
            x = MaxPooling2D(poolsize, padding='same')(x)
        elif Flag_CNN_Model == '3d':
            x = MaxPooling3D(poolsize, padding='same')(x)

    x = conv_block(x, [filters, filters, filters*2], convsize, strides=1, flag_2d3d=Flag_CNN_Model)
    x = identity_block(x, [filters, filters, filters*2], convsize, flag_2d3d=Flag_CNN_Model)
    x = identity_block(x, [filters, filters, filters*2], convsize, flag_2d3d=Flag_CNN_Model)

    filters *= 2
    x = conv_block(x, [filters, filters, filters*2], convsize, strides=2, flag_2d3d=Flag_CNN_Model)
    x = identity_block(x, [filters, filters, filters*2],convsize, flag_2d3d=Flag_CNN_Model)
    x = identity_block(x, [filters, filters, filters*2],convsize, flag_2d3d=Flag_CNN_Model)
    x = identity_block(x, [filters, filters, filters*2],convsize, flag_2d3d=Flag_CNN_Model)

    if not Flag_Simple_RESNET:
        filters *= 2
        x = conv_block(x, [filters, filters, filters*2], convsize, strides=2, flag_2d3d=Flag_CNN_Model)
        x = identity_block(x, [filters, filters, filters*2],convsize, flag_2d3d=Flag_CNN_Model)
        x = identity_block(x, [filters, filters, filters*2],convsize, flag_2d3d=Flag_CNN_Model)
        x = identity_block(x, [filters, filters, filters*2],convsize, flag_2d3d=Flag_CNN_Model)
        x = identity_block(x, [filters, filters, filters*2],convsize, flag_2d3d=Flag_CNN_Model)
        x = identity_block(x, [filters, filters, filters*2],convsize, flag_2d3d=Flag_CNN_Model)


    filters *= 2
    x = conv_block(x, [filters, filters, filters*2], convsize, strides=2, flag_2d3d=Flag_CNN_Model)
    x = identity_block(x, [filters, filters, filters*2],convsize, flag_2d3d=Flag_CNN_Model)
    x = identity_block(x, [filters, filters, filters*2],convsize, flag_2d3d=Flag_CNN_Model)

    if Flag_CNN_Model == '2d':
        x = GlobalAveragePooling2D(name='avg_pool')(x)
    elif Flag_CNN_Model == '3d':
        x = GlobalAveragePooling3D(name='avg_pool')(x)

    out = Dense(Nlabels, activation='softmax')(x)

    model = tf.keras.models.Model(input0, out)
    model.summary()
    return model

####change: reduce memory but increase parameters to train
def build_cnn_model_test(image, Nlabels=nb_class, filters=16, convsize=3, convsize2=5, poolsize=2, hidden_size=256, conv_layers=4,flag_print=True):
    #     import keras.backend as K
    #     if K.image_data_format() == 'channels_first':
    #         img_shape = (1,img_rows,img_cols)
    #     elif K.image_data_format() == 'channels_last':
    #         img_shape = (img_rows,img_cols,1)

    if not Nlabels:
        global target_name
        Nlabels = len(np.unique(target_name))

    global Flag_CNN_Model

    input_tensor = Input(tensor=image)   ##use tensor instead of shape
    ####quickly reducing image dimension first
    x = conv(input_tensor, filters, convsize2, Flag_CNN_Model, strides=2)
    x = bn(x)
    x = Activation('relu')(x)

    for li in range(conv_layers-1):
        x = conv(x, filters, convsize, Flag_CNN_Model)
        x = bn(x)
        x = Activation('relu')(x)

        x = conv(x, filters, convsize, Flag_CNN_Model)
        x = bn(x)
        x = Activation('relu')(x)

        x = conv(x, filters*2, convsize, Flag_CNN_Model, strides=2)
        x = bn(x)
        x = Activation('relu')(x)

        x = Dropout(0.25)(x)
        filters *= 2
        #if (li+1) % 2 == 0:
        #    filters *= 2

    if Flag_CNN_Model == '2d':
        x = GlobalAveragePooling2D(name='avg_pool')(x)
    elif Flag_CNN_Model == '3d':
        x = GlobalAveragePooling3D(name='avg_pool')(x)

    out = Dense(Nlabels, activation='softmax')(x)

    model = Model(inputs=input_tensor, outputs=out)
    if flag_print:
        model.summary()

    '''
    adam = Adam(lr=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    '''
    return model

# Reset Keras Session
def reset_keras():
    sess = K.get_session()
    K.clear_session()
    sess.close()
    sess = K.get_session()

    try:
        del train_gen
        del val_set
        del model_test_GPU_new # this is from global space - change this as you need
    except:
        pass

    print(gc.collect()) # if it's done something you should see a number being outputted

    # use the same config as you used to create the session
    session = tf.Session(config=config)
    K.set_session(session)

#####################
#####
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='The description of the parameters')

    parser.add_argument('--num_gpu', '-g', default=2, help="(required, int, default=1) The number of gpus to run model training",type=int)
    parser.add_argument('--task_modality', '-m', default='motor', help="(required, str, default='wm') Choosing which modality of fmri data for modeling", type=str)
    parser.add_argument('--flag_resnet', '-c', default=-1, help="(required, str, default=-1) Choosing to run 2d-cnn or 3d-cnn model", type=int)
    parser.add_argument('--flag_2d3d', '-n', default='3d', help="(required, str, default='2d') Choosing to run 2d-cnn or 3d-cnn model", type=str)
    parser.add_argument('--img_resize', '-i', default=0, help='(optional, int, default=0) The duration of fmri volumes in each data sample', type=int)
    parser.add_argument('--sub_num', default=0, help='(optional, int, default=0) The number of subjects to be used ', type=int)

    parser.add_argument('--learning_rate', '-r', default=0.001, help="(required, float, default=0.001) Choosing to run 2d-cnn or 3d-cnn model", type=float)
    parser.add_argument('--val_size', '-v', default=0.1, help='(optional, float, default=0.1) The potion of subjects used in validation set', type=float)
    parser.add_argument('--kfold', '-k', default=1, help='(optional, float, default=1) The potion of subjects used in validation set',type=int)
    parser.add_argument('--block_dura', '-b', default=17, help='(optional, int, default=1) The duration of fmri volumes in each data sample', type=int)

    parser.add_argument('--batch_size', '-a', default=2, help='(optional, int, default=16) The batch size for model training', type=int)
    parser.add_argument('--nsteps', '-s', default=1000, help='(optional, int, default=1000) The number of steps within each epoch for model training', type=int)
    parser.add_argument('--nepochs', '-e', default=100, help='(optional, int, default=100) The number of epochs for model training', type=int)
    parser.add_argument('--nr_thread', '-t', default=2, help='(optional, int, default=10) The number of threads for data loading/generator', type=int)
    parser.add_argument('--buffer_size', '-f', default=4, help='(optional, int, default=20) The buffer size for data loading/generator', type=int)
    args = parser.parse_args()

    if args.num_gpu > 0:
        USE_GPU_CPU = 1
        num_CPU = 6
        if args.num_gpu == 1:
            num_GPU = args.num_gpu
        else:
            num_GPU = get_num_gpu()  # len(used_GPU_avail)
        print('\nAvaliable GPUs for usage: %d \n' % num_GPU)

    block_dura = args.block_dura
    val_size = args.val_size

    buffer_size = args.buffer_size
    nr_thread = args.nr_thread

    batch_size = args.batch_size
    #steps = args.nsteps
    nepoch = args.nepochs
    learning_rate = args.learning_rate
    ######################
    if not args.img_resize:
        img_resize = None
    else:
        img_resize = args.img_resize
        print("Downsampling fMRI volumes to {}mm before model training".format(img_resize))
    #####################
    if args.flag_resnet < 0:
        Flag_Simple_RESNET = None
        print('Use CNN models with {} convolution'.format(args.flag_2d3d))
    elif args.flag_resnet == 0:
        Flag_Simple_RESNET = 0
        print('Use RESNET50 models with {} convolution'.format(args.flag_2d3d))
    else:
        Flag_Simple_RESNET = 1
        print('Use a simple implementation of RESNET50 models with {} convolution'.format(args.flag_2d3d))
    #################################################################
    ###change buffer_size to save time
    Flag_CNN_Model = args.flag_2d3d
    '''
    if Flag_CNN_Model == '2d':
        batch_size = 64
        buffer_size = 50
        steps = 1000
        val_size = 0.1
        learning_rate = 0.001  # 0.0005  # 0.001
        nepoch = 100
    elif Flag_CNN_Model == '3d':
        batch_size = 16
        buffer_size = 20 ##20
        steps = 1000 #500
        val_size = 0.01
        learning_rate = 0.001  # 0.001
        nepoch = 50
    '''
    #####################################################
    with tf.device("/cpu:0"):
        task_contrasts, modality = bulid_dict_task_modularity(args.task_modality)
        target_name = np.unique(list(task_contrasts.values()))
        print(target_name)

        fmri_files, confound_files, subjects = load_fmri_data(pathdata, modality)
        ev_filename = "_event_labels_1200R_test_Dec2018_ALL_newLR.h5"
        label_matrix, sub_name, Trial_dura = load_event_files(fmri_files, confound_files, ev_filename=ev_filename)
        ###match fmri-files with event-design matrix
        fmri_files, confound_files, label_matrix, fmri_sub_name = preclean_data_matrix_for_shape_match(fmri_files, confound_files, label_matrix,
                                                                                                       sub_name, fmri_sub_name=subjects)

        print('each trial contains %d volumes/TRs for task %s' % (Trial_dura,modality))
        print('Collecting event design files for subjects and saved into matrix ...' , np.array(label_matrix).shape)

        nb_class = len(target_name)
        img_shape = []
        try:
            fmri_data_resample = image.resample_img(fmri_files[0], target_affine=np.diag((img_resize, img_resize, img_resize)))
        except:
            fmri_data_resample = nib.load(fmri_files[0])
        if Flag_CNN_Model == '2d':
            #tc_matrix = nib.load(fmri_files[0])
            img_rows, img_cols, img_deps = fmri_data_resample.shape[:-1]
            if K.image_data_format() == 'channels_first':
                img_shape = (block_dura, img_rows, img_cols)
            elif K.image_data_format() == 'channels_last':
                img_shape = (img_rows, img_cols, block_dura)

            steps = int(Trial_dura * img_deps / block_dura)
        elif Flag_CNN_Model == '3d':
            #fmri_data_resample = nib.load(fmri_files[0])
            img_rows, img_cols, img_deps = fmri_data_resample.get_data().shape[:-1]
            if K.image_data_format() == 'channels_first':
                img_shape = (block_dura, img_rows, img_cols, img_deps)
            elif K.image_data_format() == 'channels_last':
                img_shape = (img_rows, img_cols, img_deps, block_dura)

            steps = int(Trial_dura / block_dura)
        print("fmri data in shape: ", img_shape)


    #########################################
    '''
    ##test whether dataflow from tensorpack works
    test_sub_num = 4
    tst = data_pipe_3dcnn_block(fmri_files[:test_sub_num], confound_files[:test_sub_num], label_matrix.iloc[:test_sub_num],
                          target_name=target_name, flag_cnn=Flag_CNN_Model, block_dura=block_dura,
                          batch_size=4, data_type='train', nr_thread=2, buffer_size=4)

    out = next(tst)
    print(out[0].shape)
    print(out[1].shape)
    '''
    #########################################
    ###change:  k-fold cross-validation: split data into train, validation,test
    ##test_sub_num = int(len(fmri_files))
    if args.sub_num > 0 and args.sub_num < len(fmri_files):
        test_sub_num = args.sub_num
    else:
        test_sub_num = len(fmri_files)

    START_LR = 0.01
    BASE_LR = 0.001
    lr = tf.get_variable('learning_rate', initializer=min(START_LR, BASE_LR), trainable=False)
    tf.summary.scalar('learning_rate-summary', lr)
    opt_use = tf.train.AdamOptimizer(learning_rate)  #tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True)  # tf.train.AdamOptimizer(lr)

    ############################
    model_str = 'CNN' if Flag_Simple_RESNET is None else 'RESNET50'
    model_str += '_simple' if Flag_Simple_RESNET else ''
    logg_dir = "tensorpack_{}_conv{}_batch{}_step{}_{}".format(model_str, Flag_CNN_Model, steps, batch_size,datetime.datetime.now().strftime("%m-%d-%H"))
    shutil.rmtree(os.path.join("train_log", logg_dir), ignore_errors=True)
    logger.set_logger_dir(os.path.join("train_log", logg_dir))

    checkpoint_dir = "checkpoints/{}_model_{}_conv{}_lr{}_{}/".format(modality, model_str, Flag_CNN_Model, learning_rate,datetime.datetime.now().strftime("%m-%d-%H"))
    shutil.rmtree(checkpoint_dir, ignore_errors=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    print("Start training a {} model with {}conv, batch-size={}, step={}".format(model_str, Flag_CNN_Model, batch_size,steps))

    rs = np.random.RandomState(1234)
    for cvi in range(args.kfold):
        print('\nFold #%d:...' % (cvi+1))
        reset_keras()

        ########spliting into train,val and testing
        train_sid, val_sid = train_test_split(range(test_sub_num), test_size=val_size, random_state=rs, shuffle=True)
        if len(train_sid) < 2 or len(val_sid) < 2:
            print("Only %d subjects avaliable. Use all subjects for training and testing" % (test_sub_num))
            train_sid = range(test_sub_num)
            val_sid = range(test_sub_num)
        print('training on %d subjects, validating on %d subjects using mini-batch %d' % (len(train_sid), len(val_sid), batch_size))

        fmri_data_train = np.array([fmri_files[i] for i in train_sid])
        confounds_train = np.array([confound_files[i] for i in train_sid])
        label_train = pd.DataFrame(np.array([label_matrix.iloc[i] for i in train_sid]))

        fmri_data_val = np.array([fmri_files[i] for i in val_sid])
        confounds_val = np.array([confound_files[i] for i in val_sid])
        label_val = pd.DataFrame(np.array([label_matrix.iloc[i] for i in val_sid]))

        if args.nsteps >0 :
            steps_Train = args.nsteps
        else:
            steps_Train = int(steps * len(fmri_data_train)) // (batch_size*num_GPU)

        #########################################
        train_gen = data_pipe_3dcnn_block(fmri_data_train, confounds_train,label_train,
                                          target_name=target_name, flag_cnn=Flag_CNN_Model, block_dura=block_dura,
                                          batch_size=batch_size, data_type='train', nr_thread=nr_thread, buffer_size=buffer_size)
        val_set = data_pipe_3dcnn_block(fmri_data_val, confounds_val,label_val,
                                        target_name=target_name, flag_cnn=Flag_CNN_Model, block_dura=block_dura,
                                        batch_size=batch_size, data_type='val', nr_thread=nr_thread, buffer_size=buffer_size)

        #########################################
        def one_hot(label):
            return np.eye(len(target_name))[label]

        train_gen = dataflow.MapDataComponent(train_gen, one_hot, 1)
        val_set = dataflow.MapDataComponent(val_set, one_hot, 1)

        ###start building models
        model_test = build_cnn_model_test if Flag_Simple_RESNET is None else build_model_resnet50
        model_test_GPU_new = KerasModel(model_test,
                                        inputs_desc=[InputDesc(tf.float32, (None,) + img_shape, 'images')],
                                        targets_desc=[InputDesc(tf.float32, (None, len(target_name)), 'labels')],
                                        input=train_gen, trainer=SyncMultiGPUTrainerParameterServer(num_GPU)
                                        )

        model_test_GPU_new.compile(optimizer=opt_use, loss='categorical_crossentropy', metrics='categorical_accuracy')
        callbacks = [ModelSaver(checkpoint_dir=checkpoint_dir),
                     # MinSaver('val-error-top1'),   ##backup the model with best validation error
                     GPUUtilizationTracker(),  ## record GPU utilizations during training
                     ##ScheduledHyperParamSetter('learning_rate', [(0, min(START_LR, BASE_LR)), (5, BASE_LR * 1e-1), (50, BASE_LR * 1e-2),(80, BASE_LR * 1e-3)]),
                     ##DataParallelInferenceRunner(val_set, ScalarStats(['categorical_accuracy','categorical_crossentropy']), num_GPU)  ##comuting classification error and log to monitors
                     ]
        if args.val_size > 0:
            callbacks.append(DataParallelInferenceRunner(val_set, ScalarStats(['categorical_accuracy','categorical_crossentropy']), num_GPU))

        ######start training the model
        print('\nTraining the model on %d subjects using %d steps and validating on %d \n' % (len(train_sid),steps_Train, len(val_sid)))
        sys.stdout.flush()

        start_time = time.time()
        ##model_test_GPU_new.fit(validation_data=val_set,steps_per_epoch=steps, max_epoch=nepoch, callbacks=[ModelSaver(checkpoint_dir='checkpoints/')] )
        model_test_GPU_new.fit(steps_per_epoch=steps_Train, max_epoch=nepoch, callbacks=callbacks)
        print("Finish model traning in {} s".format(time.time() - start_time))
        reset_keras()

    sys.exit(0)

