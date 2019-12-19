import os
import sys
import time
import datetime
import numpy as np
import pandas as pd
import nibabel as nib

from tensorflow.keras.optimizers import SGD, Adam, Nadam

global img_resize, Flag_CNN_Model, Flag_Simple_RESNET, Nlabels

from datapipe import data_pipe_3dcnn_block
from model import build_cnn_model_test, build_model_resnet50
from configure_fmri import *
from utils import *

'''
import horovod.tensorflow.keras as hvd

hvd.init()
print('Hello from worker #{} of {}'.format(hvd.rank(), hvd.size()))
'''

if __name__ == "__main__":

    fmri_files, confound_files, label_matrix, modality, target_name, fmri_sub_name, steps = prepare_fmri_data(pathdata, task_modality, pathout, verbose=1)
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


    steps_Test = steps * len(test_sid) // batch_size
    steps_Train = int(steps * len(train_sid) // batch_size)
    steps_Val = 2 * steps * len(val_sid) // batch_size  ##oversampling of validation set helps
    ###data generator
    test_set = data_pipe_3dcnn_block(fmri_data_test, confounds_test, label_test,
                                     target_name=target_name, block_dura=block_dura,
                                     batch_size=1, nr_thread=1, buffer_size=1,)

    train_set = data_pipe_3dcnn_block(fmri_data_train, confounds_train, label_train,
                                     target_name=target_name, block_dura=block_dura,
                                     batch_size=1, nr_thread=1, buffer_size=1,)

    val_set = data_pipe_3dcnn_block(fmri_data_val, confounds_val, label_val,
                                     target_name=target_name, block_dura=block_dura,
                                     batch_size=1, nr_thread=1, buffer_size=1,)


    start_time = datetime.datetime.now()
    print("Started script on {}".format(start_time))
    print('\nTraining the model on {} subjects with {} fmri scans and validated on {} subjects with {} fmri scans\n'
          .format(len(train_sid), len(train_file_sid),len(val_sid),len(val_file_sid)))

    ###build the model
    model_test = build_cnn_model_test if Flag_Simple_RESNET<=0 else build_model_resnet50
    model_test_GPU = model_test(img_shape, Nlabels, flag_print=1)

    opt = Adam(lr=learning_rate,beta_1=learning_rate_decay)
    model_test_GPU.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    ######start training the model
    model_test_history2 = model_test_GPU.fit_generator(train_set, epochs=10, steps_per_epoch=steps_Train,
                                                       validation_data=val_set, validation_steps=steps_Val,
                                                       verbose=1, shuffle=True, use_multiprocessing=False,
                                                       ##callbacks=[tensorboard,checkpoint_callback,early_stopping_callback],
                                                       )  ###workers=1, use_multiprocessing=False)