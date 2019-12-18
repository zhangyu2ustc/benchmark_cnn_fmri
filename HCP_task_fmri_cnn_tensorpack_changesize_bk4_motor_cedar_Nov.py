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
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
###%matplotlib inline

from collections import Counter
from operator import itemgetter
from nilearn import signal
from nilearn import image,masking
from sklearn import preprocessing

import tensorflow as tf
from tensorpack import dataflow
from tensorpack.utils.serialize import dumps, loads

from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, LearningRateScheduler, Callback
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras.layers import Input, Activation, Dense, Flatten, Conv2D, MaxPooling2D, Dropout, AveragePooling2D, GlobalAveragePooling2D
from tensorflow.keras.layers import Conv3D, MaxPooling3D, BatchNormalization, AveragePooling3D, GlobalAveragePooling3D, Add
import tensorflow.keras as keras

from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K


#####global variable settings
'''
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
'''
'''
USE_GPU_CPU = 1
num_cores = 4

if not USE_GPU_CPU :
    num_CPU = num_cores
    num_GPU = 0
else:
    num_CPU = 1
    num_GPU = num_cores

config = tf.ConfigProto(intra_op_parallelism_threads=24,\
        inter_op_parallelism_threads=num_cores, allow_soft_placement=True,\
        device_count = {'CPU' : num_CPU, 'GPU' : num_GPU})
'''
USE_GPU_CPU =1
num_GPU =2
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = "0,1"

session = tf.Session(config=config)
K.set_session(session)

###check the actual No of GPUs for usage
from tensorflow.python.client import device_lib
used_GPU_avail = [x.name for x in device_lib.list_local_devices() if x.device_type == 'GPU']
num_GPU = len(used_GPU_avail)
print('\nAvaliable GPUs for usage: %s \n' % used_GPU_avail)
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os.system("lscpu")

#########################################################
'''
pathdata = Path('/home/yuzhang/projects/rrg-pbellec/DATA/HCP/aws-s3_copy_022718/')
pathout = '/home/yuzhang/projects/rrg-pbellec/yuzhang/HCP/temp_res_new/'
'''
pathdata = Path('/data/cisl/project/zenith_cpu/HCP/aws-s3_copy_041519/')
pathout = "/data/cisl/project/zenith_cpu/HCP/temp_res_new/"

####################################
TR = 0.72
nr_thread=4 #4
buffer_size=4 #16
#batch_size=8

'''
#img_resize = 3
Flag_CNN_Model = '3d'
Flag_Simple_RESNET = -1
Flag_Block_Trial = 1 #0
block_dura = 17 #1
steps = 170 // block_dura
'''

learning_rate = 0.001 #0.001
learning_rate_decay = 0.9 #0.9
l2_reg = 0.0001
randseed = 1234
val_size = 0.1
test_size = 0.2
########################

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
        if os.path.isfile(confound):
            confound_files.append(str(confound))
        else:
            print("confound files missing for subject:",fmri_file)

        subname_info = Path(os.path.dirname(fmri_file)).parts[-3]
        taskname_info = Path(os.path.dirname(fmri_file)).parts[-1].split('_')
        subjects.append('_'.join((subname_info, taskname_info[-2], taskname_info[-1])))

    '''
    for confound in sorted(pathdata.glob('*/FMRI/tfMRI_'+modality+'_??/*Movement_Regressors.txt')):
        confound_files.append(str(confound))
    '''
    print('%d subjects included in the dataset' % len(fmri_files))
    return fmri_files, confound_files, subjects


def load_event_files(fmri_files,confound_files,ev_filename=None):
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

    events_all_subjects_file = pathout+modality+ev_filename
    if os.path.isfile(events_all_subjects_file):
        trial_infos = pd.read_csv(EVS_files[0],sep="\t",encoding="utf8",header = None,names=['onset','duration','rep','task'])
        Duras = np.ceil((trial_infos.duration/TR)).astype(int) #(trial_infos.duration/TR).astype(int)

        subjects_trial_labels = pd.read_csv(events_all_subjects_file,sep="\t",encoding="utf8")
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
        
        trial_class_counts = Counter(list(subjects_trial_label_matrix.iloc[0,:]))
        print(trial_class_counts)
        print('Collecting trial info from file:', events_all_subjects_file)
        print(np.array(subjects_trial_label_matrix).shape,len(sub_name),len(np.unique(sub_name)),len(coding_direct))
    else:
        print('Loading trial info for each task-fmri file and save to csv file:', events_all_subjects_file)
        subjects_trial_label_matrix = []
        sub_name = []
        coding_direct = []
        for subj in np.arange(Subject_Num):
            pathsub = Path(os.path.dirname(EVS_files[subj]))
            #sub_name.append(pathsub.parts[-3])
            ###adjust the code after changing to the new folder
            sub_name.append(str(os.path.basename(EVS_files[subj]).split('_')[0]))
            coding_direct.append(pathsub.parts[-1].split('_')[-1])

            ##trial info in volume
            trial_infos = pd.read_csv(EVS_files[subj],sep="\t",encoding="utf8",header = None,names=['onset','duration','rep','task'])
            Onsets = np.ceil((trial_infos.onset/TR)).astype(int) #(trial_infos.onset/TR).astype(int)
            Duras = np.ceil((trial_infos.duration/TR)).astype(int) #(trial_infos.duration/TR).astype(int)
            Movetypes = trial_infos.task

            labels = ["rest"]*Trial_Num;
            trialID = [0] * Trial_Num;
            tid = 1
            for start,dur,move in zip(Onsets,Duras,Movetypes):
                for ti in range(start-1,start+dur):
                    labels[ti]= task_contrasts[move]
                    trialID[ti] = tid
                tid += 1
            subjects_trial_label_matrix.append(labels)

        ##subjects_trial_label_matrix = np.array(subjects_trial_label_matrix)
        #print(np.array(subjects_trial_label_matrix[0]))
        subjects_trial_labels = pd.DataFrame(data=np.array(subjects_trial_label_matrix),columns=['trial'+str(i+1) for i in range(Trial_Num)])
        subjects_trial_labels['trialID'] = tid
        subjects_trial_labels['subject'] = sub_name
        subjects_trial_labels['coding'] = coding_direct
        subjects_trial_labels.keys()
        subjects_trial_label_matrix = pd.DataFrame(data=np.array(subjects_trial_label_matrix),columns=['trial'+str(i+1) for i in range(Trial_Num)])

        ##save the labels
        subjects_trial_labels.to_csv(events_all_subjects_file,sep='\t', encoding='utf-8',index=False)

    block_dura = sum(Duras)
    return subjects_trial_label_matrix, sub_name, block_dura

def preclean_data_matrix_for_shape_match(fmri_files, confound_files, label_matrix, ev_sub_name, fmri_sub_name=None):
    ###"Pre-clean the fmri and event data to make sure the matching shapes between two arrays!"

    #####################sort list of files
    print('\nSort both fmri and event files into the same order!')
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
    for subcount,ev in enumerate(ev_sub_name_sorted):
        evfile_mask = pd.Series(ev).isin(fmri_sub_name_sorted)
        if not evfile_mask[0]:
            label_matrix_sorted[subcount] = []
            ev_sub_name_sorted[subcount] = []
            print("Remove event file: {} from the list!! Remaining {} event files".format(ev, len(ev_sub_name_sorted)))
    label_matrix_sorted = list(filter(None, label_matrix_sorted))
    ev_sub_name_sorted = list(filter(None, ev_sub_name_sorted))
    print("Remaining {}/{} event files".format(len(ev_sub_name_sorted),np.array(label_matrix_sorted).shape))

    for subcount,fmri_file in enumerate(fmri_sub_name_sorted):
        fmrifile_mask = pd.Series(fmri_file).isin(ev_sub_name_sorted)
        if not fmrifile_mask[0]:
            del fmri_matrix_sorted[subcount]
            del confound_matrix_sorted[subcount]
            fmri_sub_name_sorted.remove(fmri_file)
            print("Remove fmri file: {} from the list!! Remaining {} fmri files".format(fmri_file, len(fmri_sub_name_sorted)))
        else:
            subcount += 1

    label_matrix_sorted = pd.DataFrame(data=(label_matrix_sorted))
    print('\nNew shapes of fmri-data-matrix and trial-label-matrix after matching!')
    print(np.array(fmri_matrix_sorted).shape, np.array(label_matrix_sorted).shape)

    if len(fmri_sub_name_sorted) != len(ev_sub_name_sorted):
        print('Warning: Mis-matching subjects list between fmri-data-matrix and trial-label-matrix')
        print(np.array(fmri_matrix_sorted).shape, np.array(label_matrix_sorted).shape)

    for subj in range(min(len(fmri_sub_name_sorted), len(ev_sub_name_sorted))):
        if fmri_sub_name_sorted[subj] != ev_sub_name_sorted[subj]:
            print('miss-matching of subject order! {}/{}'.format(fmri_sub_name_sorted[subj],ev_sub_name_sorted[subj]))

    print('Done matching data shapes:', len(fmri_matrix_sorted), len(confound_matrix_sorted), np.array(label_matrix_sorted).shape)
    return fmri_matrix_sorted,confound_matrix_sorted, label_matrix_sorted, fmri_sub_name


############################
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

#######################################
####tensorpack: multithread
class gen_fmri_file(dataflow.DataFlow):
    """ Iterate through fmri filenames, confound filenames and labels
    """
    def __init__(self, fmri_files,confound_files, label_matrix,data_type='train',train_percent=0.8):
        assert (len(fmri_files) == len(confound_files))
        # self.data=zip(fmri_files,confound_files)
        self.fmri_files = fmri_files
        self.confound_files = confound_files
        self.label_matrix = label_matrix

        self.data_type=data_type
        self.train_percent=train_percent

    def size(self):
        return int(1e8)
        #split_num=int(len(self.fmri_files)*0.8)
        #if self.data_type=='train':
        #    return split_num
        #else:
        #    return len(self.fmri_files)-split_num

    def get_data(self):
        split_num=int(len(self.fmri_files) * self.train_percent)
        if self.data_type=='train':
            while True:
                rand_pos=np.random.choice(split_num,1)[0]
                yield self.fmri_files[rand_pos],self.confound_files[rand_pos],self.label_matrix.iloc[rand_pos]
        else:
            for pos_ in range(split_num,len(self.fmri_files)):
                yield self.fmri_files[pos_],self.confound_files[pos_],self.label_matrix.iloc[pos_]


class split_samples(dataflow.DataFlow):
    """ Iterate through fmri filenames, confound filenames and labels
    """
    def __init__(self, ds, subject_num=1200, batch_size=16, dataselect_percent=1.0):
        self.ds = ds
        self.Subjects_Num = ds.size()
        self.batch_size = batch_size
        self.dataselect_percent = dataselect_percent
        self.labels = []

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(self.size())

    def size(self):
        #return 91*284
        #return 91*284
        ds_data_shape = self.data_info()
        self.Trial_Num = ds_data_shape[0]
        self.Block_dura = ds_data_shape[-1]
        self.Samplesize = self.Subjects_Num * self.Trial_Num

        #return self.Samplesize
        return int(self.Samplesize / self.batch_size * self.dataselect_percent)

    def data_info(self):
        data = self.ds.get_data().__next__()
        print('fmri/label data shape:',data[0].shape,data[1].shape)
        return data[0].shape
   
    def get_data(self):
        for data in self.ds.get_data():
            for i in range(data[1].shape[0]):
                ####yield data[0][i],data[1][i]
                yield data[0][i].astype('float32',casting='same_kind'),data[1][i]


def map_load_fmri_image(dp,target_name):
    fmri_file=dp[0]
    confound_file=dp[1]
    label_trials=dp[2]
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
    ##pre-select task types
    trial_mask = pd.Series(label_trials).isin(target_name)  ##['hand', 'foot','tongue']
    fmri_data_cnn = image.index_img(fmri_data_clean, np.where(trial_mask)[0]).get_data().astype('float32',casting='same_kind')
    img_rows, img_cols, img_deps = fmri_data_cnn.shape[:-1]
    #min_max_scaler = preprocessing.MinMaxScaler()
    #fmri_data_cnn = min_max_scaler.fit_transform(fmri_data_cnn.reshape(np.prod(fmri_data_cnn.shape[:-1]),fmri_data_cnn.shape[-1]))
    fmri_data_cnn = fmri_data_cnn.reshape(np.prod(fmri_data_cnn.shape[:-1]),fmri_data_cnn.shape[-1])
    fmri_data_cnn = preprocessing.scale(fmri_data_cnn, axis=1).astype('float32',casting='same_kind')
    fmri_data_cnn[np.isnan(fmri_data_cnn)] = 0
    fmri_data_cnn[np.isinf(fmri_data_cnn)] = 0
    fmri_data_cnn = fmri_data_cnn.reshape((img_rows, img_cols, img_deps,fmri_data_cnn.shape[-1]))

    ###use each slice along z-axis as one sample
    label_data_trial = np.array(label_trials.loc[trial_mask])
    le = preprocessing.LabelEncoder()
    le.fit(target_name)
    label_data_cnn = le.transform(label_data_trial) ##np_utils.to_categorical(): convert label vector to matrix

    img_rows, img_cols, img_deps = fmri_data_cnn.shape[:-1]
    fmri_data_cnn_test = np.transpose(fmri_data_cnn.reshape(img_rows, img_cols, np.prod(fmri_data_cnn.shape[2:])), (2, 0, 1))
    label_data_cnn_test = np.repeat(label_data_cnn, img_deps, axis=0).flatten()
    ##print(fmri_file, fmri_data_cnn_test.shape,label_data_cnn_test.shape)
    fmri_data_cnn = None

    return fmri_data_cnn_test, label_data_cnn_test

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

def data_pipe(fmri_files,confound_files,label_matrix,target_name=None,batch_size=32,data_type='train',
              train_percent=0.8,nr_thread=nr_thread,buffer_size=buffer_size):
    assert data_type in ['train', 'val', 'test']
    assert fmri_files is not None

    print('\n\nGenerating dataflow for %s datasets \n' % data_type)

    buffer_size = min(len(fmri_files),buffer_size)
    nr_thread = min(len(fmri_files),nr_thread)

    ds0 = gen_fmri_file(fmri_files,confound_files, label_matrix,data_type=data_type,train_percent=train_percent)
    print('dataflowSize is ' + str(ds0.size()))
    print('Loading data using %d threads with %d buffer_size ... \n' % (nr_thread, buffer_size))

    if target_name is None:
        target_name = np.unique(label_matrix)

    ####running the model
    start_time = time.clock()
    ds1 = dataflow.MultiThreadMapData(
        ds0, nr_thread=nr_thread,
        map_func=lambda dp: map_load_fmri_image(dp,target_name),
        buffer_size=buffer_size,
        strict=True)

    ds1 = dataflow.PrefetchData(ds1, buffer_size,1)

    ds1 = split_samples(ds1, subject_num=len(fmri_files), batch_size=batch_size)
    #ds1 = split_samples(ds1)
    print('prefetch dataflowSize is ' + str(ds1.size()))
    Trial_Num = ds1.data_info()[0]
    print('%d #Trials/Samples per subject' % Trial_Num)

    ##ds1 = dataflow.LocallyShuffleData(ds1,buffer_size=ds1.size()*buffer_size)
    ds1 = dataflow.LocallyShuffleData(ds1,buffer_size=Trial_Num * buffer_size)

    ds1 = dataflow.BatchData(ds1,batch_size=batch_size)
    print('Time Usage of loading data in seconds: {} \n'.format(time.clock() - start_time))

    ds1 = dataflow.PrefetchDataZMQ(ds1, nr_proc=1)
    ##ds1._reset_once()
    ds1.reset_state()

    #return ds1.get_data()
    for df in ds1.get_data():
        ##print(np.expand_dims(df[0].astype('float32'),axis=3).shape)
        yield (np.expand_dims(df[0].astype('float32'),axis=3),to_categorical(df[1].astype('int32'),len(target_name)))


def data_pipe_3dcnn(fmri_files, confound_files, label_matrix, target_name=None, flag_cnn='3d', batch_size=32,
                    data_type='train',train_percent=0.8, nr_thread=nr_thread, buffer_size=buffer_size):
    assert data_type in ['train', 'val', 'test']
    assert flag_cnn in ['3d', '2d']
    assert fmri_files is not None
    isTrain = data_type == 'train'
    isVal = data_type == 'val'
    isTest = data_type == 'test'

    print('\n\nGenerating dataflow for %s datasets \n' % data_type)

    buffer_size = min(len(fmri_files), buffer_size)
    nr_thread = min(len(fmri_files), nr_thread)

    ds0 = gen_fmri_file(fmri_files, confound_files, label_matrix, data_type=data_type, train_percent=train_percent)
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
            map_func=lambda dp: map_load_fmri_image(dp, target_name),
            buffer_size=buffer_size,
            strict=True)
    elif flag_cnn == '3d':
        ds1 = dataflow.MultiThreadMapData(
            ds0, nr_thread=nr_thread,
            map_func=lambda dp: map_load_fmri_image_3d(dp, target_name),
            buffer_size=buffer_size,
            strict=True)

    ds1 = dataflow.PrefetchData(ds1, buffer_size, 1)

    ds1 = split_samples(ds1, subject_num=len(fmri_files), batch_size=batch_size)
    dataflowSize = ds1.size()
    print('prefetch dataflowSize is ' + str(dataflowSize))

    if isTrain:
        Trial_Num = ds1.Trial_Num
        ds1 = dataflow.LocallyShuffleData(ds1, buffer_size=Trial_Num * buffer_size, shuffle_interval=Trial_Num * buffer_size//2) #//2

    ds1 = dataflow.BatchData(ds1, batch_size=batch_size)
    print('Time Usage of loading data in seconds: {} \n'.format(time.clock() - start_time))

    ds1 = dataflow.PrefetchDataZMQ(ds1, nr_proc=1)
    ##ds1._reset_once()
    ds1.reset_state()

    ##return ds1.get_data()

    for df in ds1.get_data():
        if flag_cnn == '2d':
            yield (np.expand_dims(df[0].astype('float32'), axis=3),to_categorical(df[1].astype('int32'), len(target_name)))
        elif flag_cnn == '3d':
            yield (np.expand_dims(df[0].astype('float32'), axis=4),to_categorical(df[1].astype('int32'), len(target_name)))


#################################################################
#########reshape fmri and label data into blocks
#######change: use blocks instead of trials as input
def map_load_fmri_image_block(dp,target_name,block_dura=1,hrf_delay=0):
    ###extract time-series within each block in terms of trial numbers
    fmri_file=dp[0]
    confound_file=dp[1]
    label_trials=dp[2]
    '''
    ###remove confound effects
    confound = np.loadtxt(confound_file)
    ##using orthogonal matrix instead of original matrix
    normed_matrix = preprocessing.normalize(confound, axis=0, norm='l2')
    confound, R = np.linalg.qr(normed_matrix)
    mask_img = masking.compute_epi_mask(fmri_file)
    fmri_data_clean = image.clean_img(fmri_file, detrend=False, standardize=True, confounds=confound, t_r=TR, ensure_finite=True, mask_img=mask_img)
    '''
    fmri_data_clean = fmri_file
    ##pre-select task types
    if hrf_delay > 0:
        label_trials = np.roll(label_trials, hrf_delay)
    trial_mask = pd.Series(label_trials).isin(target_name)  ##['hand', 'foot','tongue']
    fmri_data_cnn = image.index_img(fmri_data_clean, np.where(trial_mask)[0]).get_data().astype('float32',casting='same_kind')
    img_rows, img_cols, img_deps = fmri_data_cnn.shape[:-1]

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
            blocks_num = label_data_trial_block.shape[0]
            chunks = 0
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
    label_data_cnn_test = le.transform(np.repeat(label_data[:, 0], img_deps, axis=0)).flatten()
    ##fmri_data_cnn_block = np.array_split(fmri_data_cnn, np.where(np.diff(label_data_int))[0] + 1, axis=3)
    fmri_data_cnn_test = np.array([fmri_data_cnn_block[i][:, :, :, :block_dura] for i in range(chunks)])
    fmri_data_cnn = None
    fmri_data_cnn_block = None
    ###reshape data to fit the model
    fmri_data_cnn_test = np.transpose(fmri_data_cnn_test, (3, 0, 1, 2, 4))
    fmri_data_cnn_test = fmri_data_cnn_test.reshape(np.prod(fmri_data_cnn_test.shape[:2]), img_rows, img_cols,block_dura)
    #fmri_data_cnn_test = NDStandardScaler().fit_transform(fmri_data_cnn_test)
    ##print(fmri_data_clean, fmri_data_cnn_test.shape, label_data_cnn_test.shape)

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
    ##fmri_data_cnn_block = np.array_split(fmri_data_cnn, np.where(np.diff(label_data_int))[0] + 1, axis=3)
    fmri_data_cnn = None
    fmri_data_cnn_test = np.array([fmri_data_cnn_block[i][:, :, :, :block_dura] for i in range(chunks)])
    fmri_data_cnn_test = NDStandardScaler().fit_transform(fmri_data_cnn_test)
    fmri_data_cnn_block = None
    ##print(fmri_file, fmri_data_cnn_test.shape, label_data_cnn_test.shape)

    return fmri_data_cnn_test, label_data_cnn_test

def data_pipe_3dcnn_block(fmri_files, confound_files, label_matrix, target_name=None, flag_cnn='3d', block_dura=1,
                    batch_size=32,data_type='train',train_percent=0.8, nr_thread=nr_thread, buffer_size=buffer_size):
    assert data_type in ['train', 'val', 'test']
    assert flag_cnn in ['3d', '2d']
    assert fmri_files is not None
    isTrain = data_type == 'train'
    isVal = data_type == 'val'
    isTest = data_type == 'test'


    print('\n\nGenerating dataflow for %s datasets \n' % data_type)

    buffer_size = min(len(fmri_files), buffer_size)
    nr_thread = min(len(fmri_files), nr_thread)

    ds0 = gen_fmri_file(fmri_files, confound_files, label_matrix, data_type=data_type, train_percent=train_percent)
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

    ds1 = dataflow.PrefetchData(ds1, buffer_size, 1)

    ds1 = split_samples(ds1, subject_num=len(fmri_files), batch_size=batch_size)
    dataflowSize = ds1.size()
    print('prefetch dataflowSize is ' + str(dataflowSize))

    if isTrain:
        Trial_Num = ds1.Trial_Num
        ds1 = dataflow.LocallyShuffleData(ds1, buffer_size=Trial_Num * buffer_size, shuffle_interval=Trial_Num * buffer_size//2) #//2

    ds1 = dataflow.BatchData(ds1, batch_size=batch_size)
    print('Time Usage of loading data in seconds: {} \n'.format(time.clock() - start_time))

    ds1 = dataflow.PrefetchDataZMQ(ds1, nr_proc=1)
    ##ds1._reset_once()
    ds1.reset_state()

    for df in ds1.get_data():
        if flag_cnn == '2d':
            yield (df[0].astype('float32'),to_categorical(df[1].astype('int32'), len(target_name)))
        elif flag_cnn == '3d':
            yield (df[0].astype('float32'),to_categorical(df[1].astype('int32'), len(target_name)))

###end of tensorpack: multithread
##############################################################

def plot_history(model_history):
    plt.figure()
    plt.subplot(121)
    plt.plot(model_history.history['acc'], color='r')
    plt.plot(model_history.history['val_acc'], color='b')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(['Training', 'Validation'])

    plt.subplot(122)
    plt.plot(model_history.history['loss'], color='r')
    plt.plot(model_history.history['val_loss'], color='b')
    plt.xlabel('Epochs')
    plt.ylabel('Loss Function')
    plt.legend(['Training', 'Validation'])
    return None

#####build different neural networks
####################################################################
def bn(x, bn_axis=-1, zero_init=False):
    if K.image_data_format() == 'channels_first':
        bn_axis = 1
    elif K.image_data_format() == 'channels_last':
        bn_axis = -1
    return BatchNormalization(axis=bn_axis, fused=True,momentum=0.9, epsilon=1e-5,gamma_initializer='zeros' if zero_init else 'ones')(x)

def conv2d(x, filters, kernel, strides=1, name=None):
    return Conv2D(filters, kernel, strides=strides, use_bias=False, padding='same',
                  kernel_initializer="he_normal", kernel_regularizer=keras.regularizers.l2(l2_reg))(x)

def conv3d(x, filters, kernel, strides=1, name=None):
    return Conv3D(filters, kernel,strides=strides, use_bias=False, padding='same',
                  kernel_initializer="he_normal", kernel_regularizer=keras.regularizers.l2(l2_reg))(x)

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

def build_model_resnet50(input_shape, Nlabels=6, filters=16, convsize=3, convsize2=5, poolsize=2, hidden_size=256,flag_print=True):
    #############build resnet50 of 2d and 3d conv
    if not Nlabels:
        global target_name
        Nlabels = len(np.unique(target_name))+1

    global Flag_CNN_Model, Flag_Simple_RESNET

    input0 = Input(shape=input_shape)

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
    if flag_print:
        model.summary()

    return model

####change: reduce memory but increase parameters to train
def build_cnn_model_test(input_shape, Nlabels=6, filters=16, convsize=3, convsize2=5, poolsize=2, hidden_size=256, conv_layers=4,flag_print=True):
    #     import keras.backend as K
    #     if K.image_data_format() == 'channels_first':
    #         img_shape = (1,img_rows,img_cols)
    #     elif K.image_data_format() == 'channels_last':
    #         img_shape = (img_rows,img_cols,1)

    if not Nlabels:
        global target_name
        Nlabels = len(np.unique(target_name))

    global Flag_CNN_Model

    input_tensor = Input(shape=input_shape)    ##use tensor instead of shape
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

####change: reduce memory but increase parameters to train
def build_cnn2d_model_test(input_shape, Nlabels, filters=16, convsize=3, convsize2=5, poolsize=2, hidden_size=128, conv_layers=3):
    #     import keras.backend as K
    #     if K.image_data_format() == 'channels_first':
    #         img_shape = (1,img_rows,img_cols)
    #     elif K.image_data_format() == 'channels_last':
    #         img_shape = (img_rows,img_cols,1)

    input0 = Input(shape=input_shape)
    drop1 = input0
    ####quickly reducing image dimension first
    for li in range(2):
        conv1 = Conv2D(filters, (convsize, convsize), strides=2, padding='same',activation='relu',
                    kernel_initializer='he_normal',kernel_regularizer=regularizers.l2(l2_reg))(drop1)
        conv1 = BatchNormalization()(conv1)
        #pool1 = MaxPooling2D((poolsize, poolsize))(conv1)
        #drop1 = Dropout(0.25)(pool1)
        drop1 = Dropout(0.25)(conv1)
        filters *= 2
    for li in range(conv_layers-1):
        conv1 = Conv2D(filters, (convsize2, convsize2), padding='same',activation='relu',
                    kernel_initializer='he_normal',kernel_regularizer=regularizers.l2(l2_reg))(drop1)
        conv1 = BatchNormalization()(conv1)
        conv1 = Conv2D(filters, (convsize, convsize), padding='same',activation='relu',
                    kernel_initializer='he_normal',kernel_regularizer=regularizers.l2(l2_reg))(conv1)
        conv1 = BatchNormalization()(conv1)
        conv1 = Conv2D(filters, (convsize, convsize), strides=2, padding='same',activation='relu',
                    kernel_initializer='he_normal',kernel_regularizer=regularizers.l2(l2_reg))(conv1)
        conv1 = BatchNormalization()(conv1)
        #pool1 = MaxPooling2D((poolsize, poolsize))(conv1)
        #drop1 = Dropout(0.25)(pool1)
        drop1 = Dropout(0.25)(conv1)
        if (li+1) % 2 == 0:
            filters *= 2

    drop2 = drop1
    avg1 = AveragePooling2D(pool_size=(5, 5))(drop2)
    flat = Flatten()(avg1)
    hidden = Dense(hidden_size, kernel_initializer='he_normal',kernel_regularizer=regularizers.l2(l2_reg), activation='relu')(flat)
    drop3 = Dropout(0.4)(hidden)
    # hidden = Dense((hidden_size/4).astype(int), activation='relu')(drop3)
    # drop4 = Dropout(0.5)(hidden)

    out = Dense(Nlabels, activation='softmax')(drop3)

    model = Model(inputs=input0, outputs=out)
    model.summary()
    '''
    adam = Adam(lr=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    '''
    return model

def build_cnn3d_model_test(input_shape, Nlabels, filters=8, convsize=3, convsize2=5, poolsize=2, hidden_size=128, conv_layers=3):
    #     import keras.backend as K
    #     if K.image_data_format() == 'channels_first':
    #         img_shape = (1,img_rows,img_cols,img_deps)
    #     elif K.image_data_format() == 'channels_last':
    #         img_shape = (img_rows,img_cols, img_deps,1)

    input0 = Input(shape=input_shape)
    drop1 = input0
    ####quickly reducing image dimension first
    for li in range(2):
        conv1 = Conv3D(filters, (convsize, convsize, convsize), strides=2, padding='same',activation='relu',
                    kernel_initializer='he_normal',kernel_regularizer=regularizers.l2(l2_reg))(drop1)
        conv1 = BatchNormalization()(conv1)
        #pool1 = MaxPooling3D((poolsize, poolsize, poolsize))(conv1)
        #drop1 = Dropout(0.25)(pool1)
        drop1 = conv1 #Dropout(0.25)(conv1)
        filters *= 2
    for li in range(conv_layers-1):
        conv1 = Conv3D(filters, (convsize, convsize, convsize), padding='same',activation='relu',
                    kernel_initializer='he_normal',kernel_regularizer=regularizers.l2(l2_reg))(drop1)
        conv1 = BatchNormalization()(conv1)
        conv1 = Conv3D(filters, (convsize, convsize, convsize), padding='same',activation='relu',
                    kernel_initializer='he_normal',kernel_regularizer=regularizers.l2(l2_reg))(conv1)
        conv1 = BatchNormalization()(conv1)
        conv1 = Conv3D(filters, (convsize, convsize, convsize),strides=2, padding='same',activation='relu',
                    kernel_initializer='he_normal',kernel_regularizer=regularizers.l2(l2_reg))(conv1)
        conv1 = BatchNormalization()(conv1)
        #pool1 = MaxPooling3D((poolsize, poolsize, poolsize))(conv1)
        #drop1 = Dropout(0.25)(pool1)
        drop1 = conv1 #Dropout(0.25)(conv1)
        if (li+1) % 2 == 0:
            filters *= 2

    drop2 = drop1
    avg1 = AveragePooling3D(pool_size=(5, 5, 5))(drop2)
    flat = Flatten()(avg1)
    hidden = Dense(hidden_size, kernel_initializer='he_normal',kernel_regularizer=regularizers.l2(l2_reg), activation='relu')(flat)
    drop3 = Dropout(0.4)(hidden)

    out = Dense(Nlabels, activation='softmax')(drop3)

    model = Model(inputs=input0, outputs=out)
    model.summary()
    '''
    ###change optimizer to SGD with changing learning rate
    adam = Adam(lr=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    '''
    return model

class TimeHistory(Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)

#####################
#####
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='The description of the parameters')

    parser.add_argument('--task_modality', '-m', default='MOTOR', help="(required, str, default='wm') Choosing which modality of fmri data for modeling", type=str)
    parser.add_argument('--flag_resnet', '-c', default=-1, help="(required, str, default=-1) Choosing to run 2d-cnn or 3d-cnn model", type=int)
    parser.add_argument('--flag_2d3d', '-n', default='3d', help="(required, str, default='2d') Choosing to run 2d-cnn or 3d-cnn model", type=str)
    parser.add_argument('--nepochs', '-e', default=100, help='(optional, int, default=100) The number of epochs for model training', type=int)

    parser.add_argument('--block_dura', '-b', default=17, help='(optional, int, default=1) The duration of fmri volumes in each data sample', type=int)
    parser.add_argument('--batch_size', '-a', default=4, help='(optional, int, default=16) The batch size for model training', type=int)

    args = parser.parse_args()
    batch_size = args.batch_size
    block_dura = args.block_dura
    Flag_Block_Trial = 0 if block_dura == 1 else 1
    Flag_CNN_Model = args.flag_2d3d

    with tf.device("/cpu:0"):
        task_contrasts, modality = bulid_dict_task_modularity(args.task_modality)
        target_name = np.unique(list(task_contrasts.values()))
        Nlabels = len(target_name) ###+ 1
        print(target_name)

        fmri_files, confound_files, subjects = load_fmri_data(pathdata,modality)
        print('including %d fmri files and %d confounds files \n\n' % (len(fmri_files), len(confound_files)))

        ev_filename = "_event_labels_1200R_test_Dec2018_ALL_newLR.h5"
        label_matrix, sub_name, Trial_dura = load_event_files(fmri_files, confound_files, ev_filename=ev_filename)
        #label_matrix, sub_name, Block_dura = load_event_files(fmri_files,confound_files)
        if Flag_Block_Trial == 0:
            block_dura = 1
            print('\n use each trial as one sample during model training ')
        print('Collecting event design files for subjects and saved into matrix ...' , np.array(label_matrix).shape)
        print("batch_size:", batch_size)
        print("duration of each trial:", Trial_dura)

        ###match fmri-files with event-design matrix
        fmri_files, confound_files, label_matrix, fmri_sub_name = preclean_data_matrix_for_shape_match(fmri_files, confound_files, label_matrix,
                                                                                                       sub_name, fmri_sub_name=subjects)

        img_shape = []
        if Flag_CNN_Model == '2d':
            tc_matrix = nib.load(fmri_files[0])
            img_rows, img_cols, img_deps = tc_matrix.shape[:-1]
            if K.image_data_format() == 'channels_first':
                img_shape = (block_dura, img_rows, img_cols)
            elif K.image_data_format() == 'channels_last':
                img_shape = (img_rows, img_cols, block_dura)
            steps = int(Trial_dura * img_deps / block_dura)

        elif Flag_CNN_Model == '3d':
            ##fmri_data_resample = image.resample_img(fmri_files[0], target_affine=np.diag((img_resize, img_resize, img_resize)))
            fmri_data_resample = nib.load(fmri_files[0])
            img_rows, img_cols, img_deps = fmri_data_resample.get_data().shape[:-1]
            if K.image_data_format() == 'channels_first':
                img_shape = (block_dura, img_rows, img_cols, img_deps)
            elif K.image_data_format() == 'channels_last':
                img_shape = (img_rows, img_cols, img_deps, block_dura)
            steps = int(Trial_dura / block_dura)

        print("\nfmri data in shape: {} with {} steps per functional scan".format(img_shape,steps))

        #####################
        if args.flag_resnet < 0:
            Flag_Simple_RESNET = None
            model_str = Flag_CNN_Model + 'CNN'
            print('Use CNN models with {} convolution'.format(Flag_CNN_Model))
        elif args.flag_resnet == 0:
            Flag_Simple_RESNET = 0
            model_str = 'RESNET50_' + Flag_CNN_Model + 'CNN'
            print('Use RESNET50 models with {} convolution'.format(Flag_CNN_Model))
        else:
            Flag_Simple_RESNET = 1
            model_str = 'RESNETsimp_' + Flag_CNN_Model + 'CNN'
            print('Use a simple implementation of RESNET50 models with {} convolution'.format(Flag_CNN_Model))

    #########################################
        ##first reorder the subjectlist then split to train, validation and test
        np.random.seed(randseed)
        subjectList = np.random.permutation(range(len(fmri_files)))
        subjectTest = int(test_size * len(fmri_files))
        subjectVal = int(val_size * len(fmri_files))

        train_sid = subjectList[:-subjectVal-subjectTest]
        val_sid = subjectList[-subjectVal-subjectTest:-subjectTest]
        test_sid = subjectList[-subjectTest:]

        ##train
        fmri_data_train = np.array([fmri_files[i] for i in train_sid])
        confounds_train = np.array([confound_files[i] for i in train_sid])
        label_train = pd.DataFrame(np.array([label_matrix.iloc[i] for i in train_sid]))

        ##val
        fmri_data_val = np.array([fmri_files[i] for i in val_sid])
        confounds_val = np.array([confound_files[i] for i in val_sid])
        label_val = pd.DataFrame(np.array([label_matrix.iloc[i] for i in val_sid]))

        ##test
        fmri_data_test = np.array([fmri_files[i] for i in test_sid])
        confounds_test = np.array([confound_files[i] for i in test_sid])
        label_test = pd.DataFrame(np.array([label_matrix.iloc[i] for i in test_sid]))

        print('\nTraining the model on %d subjects, validating on %d and testing on %d \n' % (len(train_sid),len(val_sid),len(test_sid)))
        steps_Train = steps * len(train_sid) // batch_size
        steps_Val = steps * len(val_sid) // batch_size
        steps_Test = steps * len(test_sid) // batch_size

        if Flag_Block_Trial == 0:
            ###change:  k-fold cross-validation: split data into train, validation,test
            ######start cnn model
            train_gen = data_pipe_3dcnn(fmri_data_train, confounds_train, label_train,
                                        target_name=target_name, flag_cnn=Flag_CNN_Model, train_percent=1.0,
                                        batch_size=batch_size, data_type='train', nr_thread=nr_thread, buffer_size=buffer_size)
            val_set = data_pipe_3dcnn(fmri_data_val, confounds_val, label_val,
                                      target_name=target_name, flag_cnn=Flag_CNN_Model, train_percent=1.0,
                                      batch_size=batch_size, data_type='val', nr_thread=nr_thread, buffer_size=buffer_size)
            test_set = data_pipe_3dcnn(fmri_data_test, confounds_test, label_test,
                                      target_name=target_name, flag_cnn=Flag_CNN_Model, train_percent=1.0,
                                      batch_size=batch_size, data_type='test', nr_thread=nr_thread, buffer_size=buffer_size)

        else:
            #########################################
            train_gen = data_pipe_3dcnn_block(fmri_data_train, confounds_train, label_train,
                                            target_name=target_name, flag_cnn=Flag_CNN_Model, train_percent=1.0, block_dura=block_dura,
                                            batch_size=batch_size, data_type='train', nr_thread=nr_thread, buffer_size=buffer_size)
            val_set = data_pipe_3dcnn_block(fmri_data_val, confounds_val, label_val,
                                          target_name=target_name, flag_cnn=Flag_CNN_Model, train_percent=1.0, block_dura=block_dura,
                                          batch_size=batch_size, data_type='val', nr_thread=nr_thread, buffer_size=buffer_size)
            test_set = data_pipe_3dcnn_block(fmri_data_test, confounds_test, label_test,
                                          target_name=target_name, flag_cnn=Flag_CNN_Model, train_percent=1.0, block_dura=block_dura,
                                          batch_size=batch_size, data_type='test', nr_thread=nr_thread, buffer_size=buffer_size)

    #########################################
    ###build the model
    '''
    if Flag_CNN_Model == '2d':
        print('\nTraining the model using 2d-CNN with learning-rate: %s \n' % str(learning_rate))
        model_test = build_cnn2d_model_test(img_shape, Nlabels)
    elif Flag_CNN_Model == '3d':
        print('\nTraining the model using 3d-CNN with learning-rate: %s \n' % str(learning_rate))
        model_test = build_cnn3d_model_test(img_shape, Nlabels)
    '''
    model_test0 = build_cnn_model_test if Flag_Simple_RESNET is None else build_model_resnet50
    model_test = model_test0(img_shape, Nlabels)

    if USE_GPU_CPU and num_GPU > 1:
        # make the model parallel
    	model_test_GPU = multi_gpu_model(model_test, gpus=num_GPU)
    else:
        model_test_GPU = model_test

    adam = Adam(lr=learning_rate,beta_1=learning_rate_decay)
    model_test_GPU.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    #########################################
    ######start training the model
    ##model_name = 'checkpoints/' + 'model_test_' + Flag_CNN_Model + 'cnn_' + modality
    checkpoint_dir = "checkpoints/{}/{}_win{}/".format(modality,model_str,block_dura)
    model_name = checkpoint_dir + '{}_win{}_batch{}_{}_test1.h5'.format(model_str,block_dura,batch_size,datetime.datetime.now().strftime("%m-%d-%H"))
    ##model_name = "checkpoints/{}_model_test_{}cnn_lr{}_{}".format(modality,Flag_CNN_Model,learning_rate,datetime.datetime.now().strftime("%m-%d-%H"))


    tensorboard = TensorBoard(log_dir="logs/{}_{}cnn_win{}_lr{}_{}".format(modality,Flag_CNN_Model,block_dura,learning_rate,datetime.datetime.now().strftime("%m-%d-%Y-%H:%M")))
    early_stopping_callback = EarlyStopping(monitor='val_acc', patience=20, mode='max')
    checkpoint_callback = ModelCheckpoint(model_name, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    time_callback = TimeHistory()
    model_test_history2 = model_test_GPU.fit_generator(train_gen, epochs=args.nepochs, steps_per_epoch=steps_Train,
                                                        validation_data=val_set,validation_steps=steps_Val, verbose=1,shuffle=True,
                                                        callbacks=[checkpoint_callback, time_callback] ##tensorboard,early_stopping_callback,
                                                        ) ##workers=1, use_multiprocessing=False)
    print(model_test_history2.history)
    print(time_callback.times)
    ## visualized with TensorBoad launched at the command line:
    ## tensorboard --logdir=logs/

    scores = model_test_GPU.evaluate_generator(test_set, steps=steps_Test, verbose=1)
    print(scores)

    import pickle
    logfilename = pathout+'train_val_scores_' + Flag_CNN_Model + 'cnn_dump2.txt'
    if os.path.isfile(logfilename):
        logfilename = logfilename.split('.')[0] + '2.txt'
    print("Saving model results to logfile: %s" % logfilename)
    file = open(logfilename, 'wb')
    pickle.dump(model_test_history2.history, file)
    pickle.dump(time_callback.times, file)
    pickle.dump(scores, file)
    file.close()

    sys.exit(0)
