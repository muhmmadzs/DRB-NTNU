import numpy as np
import pandas as pd
import os
import matplotlib as mpl
import matplotlib.pylab as plt
from sklearn.preprocessing import MinMaxScaler,StandardScaler


mpl.style.use('seaborn-paper')

from utils.constants import TRAIN_FILES, TEST_FILES, MAX_NB_VARIABLES, NB_CLASSES_LIST
def load_dataset_at_mlp(index, fold_index=None, normalize_timeseries=False, verbose=True) -> (np.array, np.array):
    if verbose: print("Loading train / test dataset : ", TRAIN_FILES[index], TEST_FILES[index])

    if fold_index is None:
        x_train_path = TRAIN_FILES[index] + "X_train.npy"
        y_train_path = TRAIN_FILES[index] + "y_train.npy"
        x_test_path = TEST_FILES[index] + "X_test.npy"
        y_test_path = TEST_FILES[index] + "y_test.npy"
        m_train_path = TRAIN_FILES[index] + "m_train.npy"
        m_test_path = TEST_FILES[index] + "m_test.npy"
    else:
        x_train_path = TRAIN_FILES[index] + "X_train_%d.npy" % fold_index
        y_train_path = TRAIN_FILES[index] + "y_train_%d.npy" % fold_index
        x_test_path = TEST_FILES[index] + "X_test_%d.npy" % fold_index
        y_test_path = TEST_FILES[index] + "y_test_%d.npy" % fold_index
        m_train_path = TRAIN_FILES[index] + "m_train_%d.npy" % fold_index
        m_test_path = TEST_FILES[index] + "m_test_%d.npy" % fold_index
        
    if os.path.exists(x_train_path):
        X_train = np.load(x_train_path)
        m_train = np.load(m_train_path)
        y_train = np.load(y_train_path)
        X_test = np.load(x_test_path)
        m_test = np.load(m_test_path)
        y_test = np.load(y_test_path)
    elif os.path.exists(x_train_path[1:]):
        X_train = np.load(x_train_path[1:])
        m_train = np.load(m_train_path[1:])
        y_train = np.load(y_train_path[1:])
        X_test = np.load(x_test_path[1:])
        m_test = np.load(m_test_path[1:])
        y_test = np.load(y_test_path[1:])
    else:
        raise FileNotFoundError('File %s not found!' % (TRAIN_FILES[index]))

    is_timeseries = True

    # extract labels Y and normalize to [0 - (MAX - 1)] range
    nb_classes = len(np.unique(y_train))
    y_train = (y_train - y_train.min()) / (y_train.max() - y_train.min()) * (nb_classes - 1)


    if is_timeseries:
        # scale the values
        if normalize_timeseries:
            X_train_mean = X_train.mean()
            X_train_std = X_train.std()
            # X_train_mean =  1.781220470760043e-14
            # X_train_std = 0.16372038656702945
            # X_train = np.reshape(X_train, [X_train.shape[0], X_train.shape[1], X_train.shape[2]])
            # X_train_flatten = np.reshape(X_train, [X_train.shape[0],X_train.shape[1]*X_train.shape[2]])
            # #scaler_X = MinMaxScaler(feature_range=(-1, 1))
            # scaler_X = StandardScaler()
            # scaler_X.fit(X_train_flatten)
            # X_train_flatten_map = scaler_X.transform(X_train_flatten )
            # X_train = np.reshape(X_train_flatten_map, [X_train.shape[0], X_train.shape[1], X_train.shape[2]])
            sc = StandardScaler()
            X_inverse=sc.fit(m_train)
            m_train= X_inverse.transform(m_train)
           # X_train = (X_train - X_train_mean) / (X_train_std+ 1e-8 )
           
            X_train = (X_train - X_train_mean) / (X_train_std ) # Traffice
            # X_train =  (X_train - X_train.min()) / (X_train.max() - X_train.min())
    if verbose: print("Finished processing train dataset..")

    # extract labels Y and normalize to [0 - (MAX - 1)] range
    nb_classes = len(np.unique(y_test))
    y_test = (y_test - y_test.min()) / (y_test.max() - y_test.min()) * (nb_classes - 1)

    if is_timeseries:
        # scale the values
        if normalize_timeseries:
            # X_test = np.reshape(X_test, [X_test.shape[0], X_test.shape[1], X_test.shape[2]])
            # X_test_flatten = np.reshape(X_test, [X_test.shape[0],X_test.shape[1]*X_test.shape[2]])
            # #scaler_y = MinMaxScaler(feature_range=(-1, 1))
            # # scaler_y = StandardScaler()
            # scaler_X.fit(X_test_flatten)
            # X_test_flatten_map = scaler_X.transform(X_test_flatten )
            # X_test = np.reshape(X_test_flatten_map, [X_test.shape[0], X_test.shape[1], X_test.shape[2]])            
            
            m_test= X_inverse.transform(m_test)
           # X_test = (X_test - X_train_mean) / (X_train_std + 1e-8 )
            X_test = (X_test - X_train_mean) / (X_train_std )  # Traffice

            
            # X_test = (X_test - X_test.mean()) / (X_test.std()+ 1e-8 )

            #X_test = (X_test - 9.433329682468139e-19) / (-0.6758118256611586 + 1e-8)

            #X_test =  (X_test - X_test.min()) / (X_test.max() - X_test.min())
    if verbose:
        print("Finished loading test dataset..")
        print()
        print("Number of train samples : ", X_train.shape[0], "Number of test samples : ", X_test.shape[0])
        print("Number of classes : ", nb_classes)
        print("Sequence length : ", X_train.shape[-1])

    return X_train, y_train, X_test, y_test, is_timeseries, m_train, m_test

def load_dataset_at(index, fold_index=None, normalize_timeseries=False, verbose=True) -> (np.array, np.array):
    if verbose: print("Loading train / test dataset : ", TRAIN_FILES[index], TEST_FILES[index])

    if fold_index is None:
        x_train_path = TRAIN_FILES[index] + "X_train.npy"
        y_train_path = TRAIN_FILES[index] + "y_train.npy"
        x_test_path = TEST_FILES[index] + "X_test.npy"
        y_test_path = TEST_FILES[index] + "y_test.npy"
        x_test_Vali_path = TEST_FILES[index] + "X_test_Vali.npy"
        y_test_Vali_path = TEST_FILES[index] + "y_test_Vali.npy"
        x_test_Reh_path = TEST_FILES[index] + "X_test_Reh.npy"
        y_test_Reh_path = TEST_FILES[index] + "y_test_Reh.npy"
        x_test_Reh2_path = TEST_FILES[index] + "X_test_Reh2.npy"
        y_test_Reh2_path = TEST_FILES[index] + "y_test_Reh2.npy"
    else:
        x_train_path = TRAIN_FILES[index] + "X_train_%d.npy" % fold_index
        y_train_path = TRAIN_FILES[index] + "y_train_%d.npy" % fold_index
        x_test_path = TEST_FILES[index] + "X_test_%d.npy" % fold_index
        y_test_path = TEST_FILES[index] + "y_test_%d.npy" % fold_index
        x_test_Reh_path = TEST_FILES[index] + "X_test_Reh_%d.npy" % fold_index
        y_test_Reh_path = TEST_FILES[index] + "y_test_Reh_%d.npy" % fold_index
        x_test_Reh2_path = TEST_FILES[index] + "X_test_Reh2_%d.npy" % fold_index
        y_test_Reh2_path = TEST_FILES[index] + "y_test_Reh2_%d.npy" % fold_index

    if os.path.exists(x_train_path):
        X_train = np.load(x_train_path)
        y_train = np.load(y_train_path)
        X_test = np.load(x_test_path)
        y_test = np.load(y_test_path)
        X_test_Vali = np.load(x_test_Vali_path)
        y_test_Vali = np.load(y_test_Vali_path)
        X_test_Reh = np.load(x_test_Reh_path)
        y_test_Reh = np.load(y_test_Reh_path)
        X_test_Reh2 = np.load(x_test_Reh2_path)
        y_test_Reh2 = np.load(y_test_Reh2_path)
    elif os.path.exists(x_train_path[1:]):
        X_train = np.load(x_train_path[1:])
        y_train = np.load(y_train_path[1:])
        X_test = np.load(x_test_path[1:])
        y_test = np.load(y_test_path[1:])
        X_test_Vali = np.load(x_test_Vali_path[1:])
        y_test_Vali = np.load(y_test_Vali_path[1:])
        X_test_Reh = np.load(x_test_Reh_path[1:])
        y_test_Reh = np.load(y_test_Reh_path[1:])
        X_test_Reh2 = np.load(x_test_Reh2_path[1:])
        y_test_Reh2 = np.load(y_test_Reh2_path[1:])
    else:
        raise FileNotFoundError('File %s not found!' % (TRAIN_FILES[index]))
    
    nb_classes = len(np.unique(y_train))
    print(np.shape(X_train))
    y_train = (y_train - y_train.min()) / (y_train.max() - y_train.min()) * (nb_classes - 1)
    X_train=np.transpose(X_train, (0, 2, 1))
    X_test=np.transpose(X_test, (0, 2, 1))
    X_test_Vali=np.transpose(X_test_Vali, (0, 2, 1))
    X_test_Reh=np.transpose(X_test_Reh, (0, 2, 1))   
    X_test_Reh2=np.transpose(X_test_Reh2, (0, 2, 1))    
    X_train_new=X_train
    X_test_new=X_test
    X_test_Vali_new=X_test_Vali
    X_test_Reh_new=X_test_Reh
    X_test_Reh2_new=X_test_Reh2

    is_timeseries = True

    # extract labels Y and normalize to [0 - (MAX - 1)] range
 
    if is_timeseries:
        # scale the values
        if normalize_timeseries:
            # X_train_mean = X_train.mean()
            # X_train_std = X_train.std()
            #X_train=np.transpose(X_train, (0, 2, 1))
            print(np.shape(X_train))
  
            X_train = np.reshape(X_train, [X_train.shape[0], X_train.shape[1], X_train.shape[2]])
            X_train_flatten = np.reshape(X_train, [X_train.shape[0]*X_train.shape[1], X_train.shape[2]])
            scaler_X = MinMaxScaler(feature_range=(0, 1))
            #scaler_X = StandardScaler()
            scaler_X.fit(X_train_flatten)
            X_train_flatten_map = scaler_X.transform(X_train_flatten)
            X_train_new = np.reshape(X_train_flatten_map, [X_train.shape[0], X_train.shape[1], X_train.shape[2]])
            print(np.shape(X_train_new))
            
            
            # X_train_new =  (X_train - X_train_mean) / (X_train_std)
            # X_train = (X_train - X_train_mean) / (X_train_std)
            # arr=np.arange(550,560)
            # X_train = np.delete(X_train, arr,axis=0)
            # X_train_new =  (X_train - X_train.min()) / (X_train.max() - X_train.min())
    if verbose: print("Finished processing train dataset..")

    # extract labels Y and normalize to [0 - (MAX - 1)] range
    nb_classes = len(np.unique(y_test))
    y_test = (y_test - y_test.min()) / (y_test.max() - y_test.min()) * (nb_classes - 1)

    if is_timeseries:
        # scale the values
        if normalize_timeseries:

            # X_test=np.transpose(X_test, (0, 2, 1))
            # X_test_Vali=np.transpose(X_test_Vali, (0, 2, 1))
            # X_test_Reh=np.transpose(X_test_Reh, (0, 2, 1))
            
            
            # Scaling for DC0
            # X_test = np.reshape(X_test, [X_test.shape[0], X_test.shape[1], X_test.shape[2]])
            X_test_flatten = np.reshape(X_test, [X_test.shape[0]*X_test.shape[1], X_test.shape[2]])
            X_test_flatten_map = scaler_X.transform(X_test_flatten)
            X_test_new = np.reshape(X_test_flatten_map, [X_test.shape[0], X_test.shape[1], X_test.shape[2]])

            # Scaling for DC0
            # X_test_Vali = np.reshape(X_test_Vali, [X_test_Vali.shape[0], X_test_Vali.shape[1], X_test_Vali.shape[2]])
            X_test_Vali_flatten = np.reshape(X_test_Vali, [X_test_Vali.shape[0]*X_test_Vali.shape[1], X_test_Vali.shape[2]])
            X_test_Vali_flatten_map = scaler_X.transform(X_test_Vali_flatten)
            X_test_Vali_new = np.reshape(X_test_Vali_flatten_map, [X_test_Vali.shape[0], X_test_Vali.shape[1], X_test_Vali.shape[2]])
            
            # Scaling for DC0
            # X_test_Reh = np.reshape(X_test_Reh, [X_test_Reh.shape[0], X_test_Reh.shape[1], X_test_Reh.shape[2]])
            X_test_Reh_flatten = np.reshape(X_test_Reh, [X_test_Reh.shape[0]*X_test_Reh.shape[1], X_test_Reh.shape[2]])
            X_test_Reh_flatten_map = scaler_X.transform(X_test_Reh_flatten)
            X_test_Reh_new = np.reshape(X_test_Reh_flatten_map, [X_test_Reh.shape[0], X_test_Reh.shape[1], X_test_Reh.shape[2]])
            
            
            X_test_Reh2_flatten = np.reshape(X_test_Reh2, [X_test_Reh2.shape[0]*X_test_Reh2.shape[1], X_test_Reh2.shape[2]])
            X_test_Reh2_flatten_map = scaler_X.transform(X_test_Reh2_flatten)
            X_test_Reh2_new = np.reshape(X_test_Reh2_flatten_map, [X_test_Reh2.shape[0], X_test_Reh2.shape[1], X_test_Reh2.shape[2]])

            # X_test_new =  (X_test -  X_train_mean) / (X_train_std)
            # X_test_Vali_new =  (X_test_Vali -  X_train_mean) / (X_train_std)
            # X_test_Reh_new =  (X_test_Reh -  X_train_mean) / (X_train_std)
            
            
            # X_test_new =  (X_test - X_train.min()) / (X_train.max() - X_train.min())
            # X_test_Vali_new =  (X_test_Vali - X_train.min()) / (X_train.max() - X_train.min())
            # X_test_Reh_new =  (X_test_Reh - X_train.min()) / (X_train.max() - X_train.min())
    
    if verbose:
        print("Finished loading test dataset..")
        print()
        print("Number of train samples : ", X_train.shape[0], "Number of test samples : ", X_test.shape[0])
        print("Number of classes : ", nb_classes)
        print("Sequence length : ", X_train.shape[-1])

    return X_train_new, y_train, X_test_new, y_test, X_test_Vali_new, y_test_Vali, X_test_Reh_new, y_test_Reh,  X_test_Reh2_new, y_test_Reh2,is_timeseries


def calculate_dataset_metrics(X_train):
    max_nb_variables = X_train.shape[-1]
    max_timesteps = X_train.shape[1]

    return max_timesteps, max_nb_variables


def cutoff_choice(dataset_id, sequence_length):
    print("Original sequence length was :", sequence_length, "New sequence Length will be : ",
          MAX_NB_VARIABLES[dataset_id])
    choice = input('Options : \n'
                   '`pre` - cut the sequence from the beginning\n'
                   '`post`- cut the sequence from the end\n'
                   '`anything else` - stop execution\n'
                   'To automate choice: add flag `cutoff` = choice as above\n'
                   'Choice = ')

    choice = str(choice).lower()
    return choice


def cutoff_sequence(X_train, X_test, choice, dataset_id, sequence_length):
    assert MAX_NB_VARIABLES[dataset_id] < sequence_length, "If sequence is to be cut, max sequence" \
                                                                   "length must be less than original sequence length."
    cutoff = sequence_length - MAX_NB_VARIABLES[dataset_id]
    if choice == 'pre':
        if X_train is not None:
            X_train = X_train[:, :, cutoff:]
        if X_test is not None:
            X_test = X_test[:, :, cutoff:]
    else:
        if X_train is not None:
            X_train = X_train[:, :, :-cutoff]
        if X_test is not None:
            X_test = X_test[:, :, :-cutoff]
    print("New sequence length :", MAX_NB_VARIABLES[dataset_id])
    return X_train, X_test


if __name__ == "__main__":
    pass