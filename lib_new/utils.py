#!/home/yuzhang/tensorflow-py3.6/bin/python3.6

# Author: Yu Zhang
# License: simplified BSD
# coding: utf-8

###fMRI decoding: using event signals instead of activation pattern from glm

from pathlib import Path
import glob
import itertools
##import lmdb
##import h5py
import os
import sys
import time
import datetime
import numpy as np
import pandas as pd
import nibabel as nib
from collections import Counter
from operator import itemgetter

from nilearn import signal
from nilearn import image,masking
from sklearn import preprocessing, metrics
from sklearn.model_selection import cross_val_score, train_test_split,ShuffleSplit

##from keras.utils import np_utils
import tensorflow as tf
from tensorpack import dataflow
##from tensorpack.utils.serialize import dumps, loads

from tensorflow.keras.optimizers import SGD, Adam, Nadam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, LearningRateScheduler, Callback
from tensorflow.keras.utils import to_categorical
##from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras.layers import Input, Activation, Dense, Flatten, Conv2D, MaxPooling2D, Dropout, AveragePooling2D, GlobalAveragePooling2D
from tensorflow.keras.layers import Conv3D, MaxPooling3D, BatchNormalization, AveragePooling3D, GlobalAveragePooling3D, Add
from tensorflow.keras.models import Model
import tensorflow.keras as keras

import tensorflow.keras.backend as K

global img_resize, Flag_CNN_Model, Flag_Simple_RESNET, Nlabels
#from datapipe import data_pipe_3dcnn_block
#from model import build_cnn_model_test, build_model_resnet50
from configure_fmri import *

print('Finish Loading packages!')


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
    lang_task_con = {"math": "math_lang",
                     "story": "story_lang"}
    emotion_task_con = {"fear": "fear_emo",
                        "neut": "non_emo"}
    gambl_task_con = {"win_event": "win_gamb",
                      "loss_event": "loss_gamb",
                      "neut_event": "non_gamb"}
    reson_task_con = {"match": "match_reson",
                      "relation": "relat_reson"}
    social_task_con = {"mental": "mental_soc",
                       "rnd": "random_soc"}
    wm_task_con = {"2bk_body": "body2b_wm",
                   "2bk_faces": "face2b_wm",
                   "2bk_places": "place2b_wm",
                   "2bk_tools": "tool2b_wm",
                   "0bk_body": "body0b_wm",
                   "0bk_faces": "face0b_wm",
                   "0bk_places": "place0b_wm",
                   "0bk_tools": "tool0b_wm"}

    dicts = [motor_task_con, lang_task_con, emotion_task_con, gambl_task_con, reson_task_con, social_task_con,
             wm_task_con]
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

    if isinstance(pathdata, str):
        pathdata = Path(pathdata)

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

    tc_matrix = nib.load(fmri_files[0])
    Trial_Num = tc_matrix.shape[-1]
    return fmri_files, confound_files, subjects, Trial_Num


def load_event_files(pathdata, modality, fmri_files, confound_files):
    ###collect the event design files
    if isinstance(pathdata, str):
        pathdata = Path(pathdata)

    Subject_Num = len(fmri_files)
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

    return fmri_files, confound_files, EVS_files

def load_event_from_h5(EVS_files, events_all_subjects_file, task_contrasts, Trial_Num=284, TR=0.72,verbose=0):
    ################################
    ###loading all event designs from h5

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
            subjects_trial_label_matrix = subjects_trial_labels.loc[:, 'trial1':'trial' + str(Trial_Num)]

        ##subjects_trial_label_matrix = subjects_trial_labels.values.tolist()
        trialID = subjects_trial_labels['trialID']
        sub_name = subjects_trial_labels['subject'].tolist()
        coding_direct = subjects_trial_labels['coding']
        if verbose:
            trial_class_counts = Counter(list(subjects_trial_label_matrix.iloc[0, :]))
            print(trial_class_counts)
            print('Collecting trial info from file:', events_all_subjects_file)
            print(np.array(subjects_trial_label_matrix).shape, len(sub_name), len(np.unique(sub_name)),
                  len(coding_direct))
    else:
        if verbose:
            print('Loading trial info for each task-fmri file and save to csv file:', events_all_subjects_file)
        subjects_trial_label_matrix = []
        sub_name = []
        coding_direct = []
        for subj in np.arange(len(EVS_files)):
            pathsub = Path(os.path.dirname(EVS_files[subj]))
            # sub_name.append(pathsub.parts[-3])
            ###adjust the code after changing to the new folder
            sub_name.append(str(os.path.basename(EVS_files[subj]).split('_')[0]))
            coding_direct.append(pathsub.parts[-1].split('_')[-1])

            ##trial info in volume
            trial_infos = pd.read_csv(EVS_files[subj], sep="\t", encoding="utf8", header=None,
                                      names=['onset', 'duration', 'rep', 'task'])
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
        subjects_trial_labels = pd.DataFrame(data=np.array(subjects_trial_label_matrix),
                                             columns=['trial' + str(i + 1) for i in range(Trial_Num)])
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


def preclean_data_matrix_for_shape_match(fmri_files, confound_files, label_matrix, ev_sub_name, fmri_sub_name=None,verbose=0):
    ###"Pre-clean the fmri and event data to make sure the matching shapes between two arrays!"

    #####################sort list of files
    if verbose:
        # print('fmri-files:',fmri_sub_name)
        # print('event-files:', ev_sub_name)
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
            if verbose:
                print("Remove event file: {} from the list!! Remaining {} event files".format(ev,
                                                                                              len(ev_sub_name_sorted)))
    label_matrix_sorted = list(filter(None, label_matrix_sorted))
    ev_sub_name_sorted = list(filter(None, ev_sub_name_sorted))
    if verbose:
        print("Remaining {}/{} event files".format(len(ev_sub_name_sorted), np.array(label_matrix_sorted).shape))

    for subcount, fmri_file in enumerate(fmri_sub_name_sorted):
        fmrifile_mask = pd.Series(fmri_file).isin(ev_sub_name_sorted)
        if not fmrifile_mask[0]:
            del fmri_matrix_sorted[subcount]
            del confound_matrix_sorted[subcount]
            fmri_sub_name_sorted.remove(fmri_file)
            if verbose:
                print("Remove fmri file: {} from the list!! Remaining {} fmri files".format(fmri_file,
                                                                                            len(fmri_sub_name_sorted)))
        else:
            subcount += 1

    label_matrix_sorted = pd.DataFrame(data=(label_matrix_sorted))
    if verbose:
        print('New shapes of fmri-data-matrix and trial-label-matrix after matching!')
        print(np.array(fmri_matrix_sorted).shape, np.array(label_matrix_sorted).shape)

        if len(fmri_sub_name_sorted) != len(ev_sub_name_sorted):
            print('Warning: Mis-matching subjects list between fmri-data-matrix and trial-label-matrix')
            print(np.array(fmri_matrix_sorted).shape, np.array(label_matrix_sorted).shape)

    for subj in range(min(len(fmri_sub_name_sorted), len(ev_sub_name_sorted))):
        if fmri_sub_name_sorted[subj] != ev_sub_name_sorted[subj]:
            if verbose:
                print('miss-matching of subject order! {}/{}'.format(fmri_sub_name_sorted[subj],
                                                                     ev_sub_name_sorted[subj]))
    if verbose:
        print('Done matching data shapes:', len(fmri_matrix_sorted), len(confound_matrix_sorted),
              np.array(label_matrix_sorted).shape)

    return fmri_matrix_sorted, confound_matrix_sorted, label_matrix_sorted, fmri_sub_name


def prepare_fmri_data(pathdata, task_modality, pathout, test_size=0.2,val_size=0.1, test_sub_num=0,randseed=1234,verbose=0):
    task_contrasts, modality = bulid_dict_task_modularity(task_modality)
    target_name = np.unique(list(task_contrasts.values()))
    if verbose:
        print(target_name)

    fmri_files, confound_files, fmri_sub_name, Trial_Num = load_fmri_data(pathdata, modality)
    if verbose:
        print("\nStep1: collect all fMRI files")
        print('%d subjects included in the dataset, each with %d fMRI volumes' % (len(fmri_files), Trial_Num))


    fmri_files, confound_files, EVS_files = load_event_files(pathdata, modality, fmri_files, confound_files)
    if verbose:
        print("\nStep2: check avaliable event design files")
        if len(EVS_files) != len(fmri_files):
            print(len(fmri_files), len(confound_files), len(EVS_files))
            print('Miss-matching number of subjects between event:{} and fmri:{} files'.format(len(EVS_files), len(fmri_files)))

    ev_filename = "_event_labels_1200R_test_Dec2018_ALL_newLR.h5"
    events_all_subjects_file = pathout + modality + ev_filename
    label_matrix, ev_sub_name, Trial_dura = load_event_from_h5(EVS_files, events_all_subjects_file,task_contrasts)
    steps = int(Trial_dura / block_dura)

    if verbose:
        print("\nStep3: Loading all event designs")
        print("duration of task trial in each fMRI data:", Trial_dura)
        print("generating {} samples per fMRI scan".format(steps))

    ###match fmri-files with event-design matrix
    fmri_files, confound_files, label_matrix, fmri_sub_name = preclean_data_matrix_for_shape_match(fmri_files, confound_files, label_matrix, ev_sub_name,fmri_sub_name=fmri_sub_name)
    if verbose>0:
        print("\nStep4: Matching {} fMRI data with {} event files for all subjects".format(len(fmri_files),len(label_matrix)))
        print(fmri_sub_name)


    return fmri_files, confound_files, label_matrix, modality, target_name, fmri_sub_name, steps

'''
if __name__ == "__main__":

    fmri_files, confound_files, label_matrix, modality, target_name, fmri_sub_name = prepare_fmri_data(pathdata, task_modality, pathout, verbose=1)
    Nlabels = len(target_name) + 1

    tc_matrix = nib.load(fmri_files[0])
    img_rows, img_cols, img_deps = tc_matrix.shape[:-1]
    img_shape = (img_rows, img_cols, img_deps, block_dura)


    if not test_sub_num:
        subjects = [sub.split("_")[0] for sub in fmri_sub_name]
        print(subjects)
        subjects_unique = np.unique(subjects)
        test_sub_num = len(subjects_unique)

    np.random.seed(randseed)
    subjectList = np.random.permutation(range(test_sub_num))
    subjectTest = int(test_size*test_sub_num)
    train_sid_tmp = subjectList[:test_sub_num-subjectTest]
    test_sid = subjectList[-subjectTest:]

    ##train
    ##convert from subject index to file index
    ##test
    test_file_sid = [si for si,sub in enumerate(subjects) if sub in subjects_unique[test_sid]]
    fmri_data_test = np.array([fmri_files[i] for i in test_file_sid])
    confounds_test = np.array([confound_files[i] for i in test_file_sid])
    label_test = pd.DataFrame(np.array([label_matrix.iloc[i] for i in test_file_sid]))

    subjectList = np.random.permutation(train_sid_tmp)
    subjectVal = int(val_size * test_sub_num)
    train_sid = subjectList[: test_sub_num-subjectVal-subjectTest]
    val_sid = subjectList[-subjectVal:]

    ##train
    train_file_sid = [si for si, sub in enumerate(subjects) if sub in subjects_unique[train_sid]]
    fmri_data_train = np.array([fmri_files[i] for i in train_file_sid])
    confounds_train = np.array([confound_files[i] for i in train_file_sid])
    label_train = pd.DataFrame(np.array([label_matrix.iloc[i] for i in train_file_sid]))
    ##val
    val_file_sid = [si for si, sub in enumerate(subjects) if sub in subjects_unique[val_sid]]
    fmri_data_val = np.array([fmri_files[i] for i in val_file_sid])
    confounds_val = np.array([confound_files[i] for i in val_file_sid])
    label_val = pd.DataFrame(np.array([label_matrix.iloc[i] for i in val_file_sid]))


    ###
    test_set = data_pipe_3dcnn_block(fmri_data_test, confounds_test, label_test,
                                     target_name=target_name, block_dura=block_dura,
                                     batch_size=1, nr_thread=1, buffer_size=1,)

    start_time = datetime.datetime.now()
    print("Started script on {}".format(start_time))
    print('\nTraining the model on {} subjects with {} fmri scans and validated on {} subjects with {} fmri scans\n'
          .format(len(train_sid), len(train_file_sid),len(val_sid),len(val_file_sid)))

    model_test = build_cnn_model_test if Flag_Simple_RESNET<=0 else build_model_resnet50
    model_test_GPU = model_test(img_shape, Nlabels, flag_print=1)
'''