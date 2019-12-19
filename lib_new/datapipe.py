#!/home/yuzhang/tensorflow-py3.6/bin/python3.6

# Author: Yu Zhang
# License: simplified BSD
# coding: utf-8

import time
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn import image,masking

from sklearn import preprocessing, metrics, base



from tensorpack import dataflow


#######################################
def one_hot(label, Nlabels=6):
    return np.eye(Nlabels)[label]


class NDStandardScaler(base.TransformerMixin):
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

####tensorpack: multithread
class gen_fmri_file(dataflow.DataFlow):
    """ Iterate through fmri filenames, confound filenames and labels
    """
    def __init__(self, fmri_files,confound_files, label_matrix,data_type='train',seed=814):
        assert (len(fmri_files) == len(label_matrix))
        assert data_type in ['train', 'val', 'test']

        # self.data=zip(fmri_files,confound_files)
        self.fmri_files = fmri_files
        self.confound_files = confound_files
        self.label_matrix = label_matrix
        self.seed = seed
        np.random.seed(self.seed)

        self.data_type=data_type

    def size(self):
        #return min(len(self.fmri_files),len(self.label_matrix))
        return int(1e8)
        #split_num=int(len(self.fmri_files)*0.8)
        #if self.data_type=='train':
        #    return split_num
        #else:
        #    return len(self.fmri_files)-split_num

    def get_data(self):

        split_num=min(len(self.fmri_files),len(self.label_matrix))
        if self.data_type.lower() == 'train':
            while True:
                #rand_pos=np.random.choice(split_num,1,replace=False)[0]
                rand_pos = np.random.randint(0, split_num)
                ##print('rank #{}, fmri_file: {}'.format(self.seed,self.fmri_files[rand_pos]))
                yield self.fmri_files[rand_pos],self.confound_files[rand_pos],self.label_matrix.iloc[rand_pos]
        else:
            for pos_ in range(split_num):
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
        ds_data_shape = self.data_info()
        self.Trial_Num = ds_data_shape[0]
        self.Block_dura = ds_data_shape[-1]
        self.Samplesize = self.Subjects_Num * self.Trial_Num

        #return self.Samplesize
        return int(self.Samplesize / self.batch_size * self.dataselect_percent)

    def data_info(self):
        ##for data in self.ds.get_data():
        data = self.ds.get_data().__next__()
        print('fmri/label data shape:',data[0].shape,data[1].shape)
        return data[0].shape

    def get_data(self):
        for data in self.ds.get_data():
            for i in range(data[1].shape[0]):
                ##self.labels.append(data[1][i])
                ####yield data[0][i],data[1][i]
                yield data[0][i].astype('float32',casting='same_kind'),data[1][i]

########################################
##loading functions
def map_load_fmri_image(dp,target_name,hrf_delay=0):
    fmri_file=dp[0]
    confound_file=dp[1]
    label_trials=dp[2]

    fmri_data_clean = fmri_file
    ##pre-select task types
    if hrf_delay > 0:
        label_trials = np.roll(label_trials, hrf_delay)
    trial_mask = pd.Series(label_trials).isin(target_name)  ##['hand', 'foot','tongue']
    fmri_data_cnn = image.index_img(fmri_data_clean, np.where(trial_mask)[0]).get_data().astype('float32',casting='same_kind')
    img_rows, img_cols, img_deps = fmri_data_cnn.shape[:-1]
    #min_max_scaler = preprocessing.MinMaxScaler()
    #fmri_data_cnn = min_max_scaler.fit_transform(fmri_data_cnn.reshape(np.prod(fmri_data_cnn.shape[:-1]),fmri_data_cnn.shape[-1]))
    fmri_data_cnn = fmri_data_cnn.reshape(np.prod(fmri_data_cnn.shape[:-1]),fmri_data_cnn.shape[-1])
    fmri_data_cnn = preprocessing.scale(fmri_data_cnn/np.max(fmri_data_cnn), axis=1).astype('float32',casting='same_kind')
    fmri_data_cnn[np.isnan(fmri_data_cnn)] = 0
    fmri_data_cnn[np.isinf(fmri_data_cnn)] = 0
    fmri_data_cnn = fmri_data_cnn.reshape((img_rows, img_cols, img_deps,fmri_data_cnn.shape[-1]))

    ###use each slice along z-axis as one sample
    label_data_trial = np.array(label_trials.loc[trial_mask])
    le = preprocessing.LabelEncoder()
    le.fit(target_name)
    label_data_cnn = le.transform(label_data_trial) ##np_utils.one_hot(): convert label vector to matrix

    img_rows, img_cols, img_deps = fmri_data_cnn.shape[:-1]
    fmri_data_cnn_test = np.transpose(fmri_data_cnn.reshape(img_rows, img_cols, np.prod(fmri_data_cnn.shape[2:])), (2, 0, 1))
    #fmri_data_cnn_test = NDStandardScaler().fit_transform(fmri_data_cnn_test)
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
    label_data_cnn = le.transform(label_data_trial)  ##np_utils.one_hot(): convert label vector to matrix

    fmri_data_cnn_test = np.transpose(fmri_data_cnn, (3, 0, 1, 2))
    fmri_data_cnn_test = NDStandardScaler().fit_transform(fmri_data_cnn_test)
    label_data_cnn_test = label_data_cnn.flatten()
    ##print(fmri_file, fmri_data_cnn_test.shape, label_data_cnn_test.shape)
    fmri_data_cnn = None

    return fmri_data_cnn_test, label_data_cnn_test


def data_pipe_3dcnn(fmri_files, confound_files, label_matrix, target_name=None, flag_cnn='3d', batch_size=32, hrf_delay=0,
                    data_type='train',nr_thread=4, buffer_size=10, dataselect_percent=1.0,seed=814, verbose=0):
    assert data_type in ['train', 'val', 'test']
    assert flag_cnn in ['3d', '2d']
    assert fmri_files is not None
    isTrain = data_type == 'train'
    isVal = data_type == 'val'
    isTest = data_type == 'test'

    buffer_size = min(len(fmri_files), buffer_size)
    nr_thread = min(len(fmri_files), nr_thread)

    ds0 = gen_fmri_file(fmri_files, confound_files, label_matrix, data_type=data_type,seed=seed)


    if target_name is None:
        target_name = np.unique(label_matrix)
    #else:
    #    if verbose:
    #        print(target_name, np.unique(label_matrix))
    ##Subject_Num, Trial_Num = np.array(label_matrix).shape

    ####running the model
    start_time = time.clock()
    if flag_cnn == '2d':
        ds1 = dataflow.MultiThreadMapData(
            ds0, nr_thread=nr_thread,
            map_func=lambda dp: map_load_fmri_image(dp, target_name,hrf_delay=hrf_delay),
            buffer_size=buffer_size,
            strict=True)
    elif flag_cnn == '3d':
        ds1 = dataflow.MultiThreadMapData(
            ds0, nr_thread=nr_thread,
            map_func=lambda dp: map_load_fmri_image_3d(dp, target_name,hrf_delay=hrf_delay),
            buffer_size=buffer_size,
            strict=True)

    ds1 = dataflow.PrefetchData(ds1, buffer_size, 1) ##1

    ds1 = split_samples(ds1, subject_num=len(fmri_files), batch_size=batch_size, dataselect_percent=dataselect_percent)
    dataflowSize = ds1.size()

    if isTrain:
        Trial_Num = ds1.Trial_Num
        ds1 = dataflow.LocallyShuffleData(ds1, buffer_size=Trial_Num * buffer_size, shuffle_interval=Trial_Num * buffer_size//2) #//2

    ds1 = dataflow.BatchData(ds1, batch_size=batch_size)

    if verbose:
        print('\n\nGenerating dataflow for %s datasets \n' % data_type)
        print('dataflowSize is ' + str(ds0.size()))
        print('Loading data using %d threads with %d buffer_size ... \n' % (nr_thread, buffer_size))
        print('prefetch dataflowSize is ' + str(dataflowSize))
        print('Time Usage of loading data in seconds: {} \n'.format(time.clock() - start_time))

    if isTrain:
        ds1 = dataflow.PrefetchDataZMQ(ds1, nr_proc=nr_thread) ##1
    else:
        ds1 = dataflow.PrefetchDataZMQ(ds1, nr_proc=1)  ##1
    ##ds1._reset_once()
    ds1.reset_state()

    ##return ds1.get_data()
    for df in ds1.get_data():
        if flag_cnn == '2d':
            yield (np.expand_dims(df[0].astype('float32'), axis=3), one_hot(df[1], len(target_name)+1).astype('uint8'))
        elif flag_cnn == '3d':
            yield (np.expand_dims(df[0].astype('float32'), axis=4), one_hot(df[1], len(target_name)+1).astype('uint8'))


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

def data_pipe_3dcnn_block(fmri_files, confound_files, label_matrix, target_name=None, flag_cnn='3d', block_dura=1,hrf_delay=0,
                    batch_size=32,data_type='train', nr_thread=4, buffer_size=10, dataselect_percent=1.0, seed=814, verbose=0):
    assert data_type in ['train', 'val', 'test']
    assert flag_cnn in ['3d', '2d']
    assert fmri_files is not None
    isTrain = data_type == 'train'
    isVal = data_type == 'val'
    isTest = data_type == 'test'

    buffer_size = int(min(len(fmri_files), buffer_size))
    nr_thread = int(min(len(fmri_files), nr_thread))

    ds0 = gen_fmri_file(fmri_files, confound_files, label_matrix, data_type=data_type,seed=seed)

    if target_name is None:
        target_name = np.unique(label_matrix)
    ##Subject_Num, Trial_Num = np.array(label_matrix).shape

    ####running the model
    start_time = time.clock()
    if flag_cnn == '2d':
        ds1 = dataflow.MultiThreadMapData(
            ds0, nr_thread=nr_thread,
            map_func=lambda dp: map_load_fmri_image_block(dp, target_name,block_dura=block_dura,hrf_delay=hrf_delay),
            buffer_size=buffer_size,
            strict=True)
    elif flag_cnn == '3d':
        ds1 = dataflow.MultiThreadMapData(
            ds0, nr_thread=nr_thread,
            map_func=lambda dp: map_load_fmri_image_3d_block(dp, target_name,block_dura=block_dura,hrf_delay=hrf_delay),
            buffer_size=buffer_size,
            strict=True)

    ds1 = dataflow.PrefetchData(ds1, buffer_size, 1) ##1

    ds1 = split_samples(ds1, subject_num=len(fmri_files), batch_size=batch_size, dataselect_percent=dataselect_percent)
    dataflowSize = ds1.size()

    if isTrain:
        if verbose:
            print('%d #Trials/Samples per subject with %d channels in tc' % (ds1.Trial_Num, ds1.Block_dura))
        Trial_Num = ds1.Trial_Num
        ds1 = dataflow.LocallyShuffleData(ds1, buffer_size=Trial_Num * buffer_size, shuffle_interval=Trial_Num * buffer_size//2) #//2

    ds1 = dataflow.BatchData(ds1, batch_size=batch_size)

    if verbose:
        print('\n\nGenerating dataflow for %s datasets \n' % data_type)
        print('dataflowSize is ' + str(ds0.size()))
        print('Loading data using %d threads with %d buffer_size ... \n' % (nr_thread, buffer_size))
        print('prefetch dataflowSize is ' + str(dataflowSize))

        print('Time Usage of loading data in seconds: {} \n'.format(time.clock() - start_time))


    if isTrain:
        ds1 = dataflow.PrefetchDataZMQ(ds1, nr_proc=nr_thread) ##1
    else:
        ds1 = dataflow.PrefetchDataZMQ(ds1, nr_proc=1)  ##1
    ##ds1._reset_once()
    ds1.reset_state()

    for df in ds1.get_data():
        yield (df[0].astype('float32'), one_hot(df[1], len(target_name)+1).astype('uint8'))

###end of tensorpack: multithread
##############################################################