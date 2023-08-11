import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow as tf
import datetime
from utils.constants import MAX_NB_VARIABLES, NB_CLASSES_LIST, MAX_TIMESTEPS_LIST
mpl.style.use('seaborn-paper')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import figure  # pylint: disable=g-import-not-at-top
from matplotlib.backends import backend_agg
from absl import flags
from sklearn.preprocessing import LabelEncoder
import tensorflow
import warnings
warnings.simplefilter('ignore', category=DeprecationWarning)

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Permute
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras import backend as K
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from utils.generic_utils import load_dataset_at, calculate_dataset_metrics, cutoff_choice, \
                                cutoff_sequence,load_dataset_at_mlp
from utils.constants import MAX_NB_VARIABLES, MAX_TIMESTEPS_LIST
try:
  import seaborn as sns  # pylint: disable=g-import-not-at-top
  HAS_SEABORN = True
except ImportError:
  HAS_SEABORN = False

NB_CLASS = NB_CLASSES_LIST[0]
def plot_weight_posteriors(names, qm_vals, qs_vals, fname):
  """Save a PNG plot with histograms of weight means and stddevs.
  Args:
    names: A Python `iterable` of `str` variable names.
      qm_vals: A Python `iterable`, the same length as `names`,
      whose elements are Numpy `array`s, of any shape, containing
      posterior means of weight varibles.
    qs_vals: A Python `iterable`, the same length as `names`,
      whose elements are Numpy `array`s, of any shape, containing
      posterior standard deviations of weight varibles.
    fname: Python `str` filename to save the plot to.
  """
  fig = figure.Figure(figsize=(6, 3))
  canvas = backend_agg.FigureCanvasAgg(fig)

  ax = fig.add_subplot(1, 2, 1)
  for n, qm in zip(names, qm_vals):
    sns.distplot(qm.reshape([-1]), ax=ax, label=n)
  ax.set_title('weight means')
  ax.set_xlim([-1.25, 1.25])
  ax.legend()

  ax = fig.add_subplot(1, 2, 2)
  for n, qs in zip(names, qs_vals):
    sns.distplot(qs.reshape([-1]), ax=ax)
  ax.set_title('weight stddevs')
  ax.set_xlim([0, 3.])

  fig.tight_layout()
  canvas.print_figure(fname, format='png',dpi=1200)
  print('saved {}'.format(fname))
  
def plot_heldout_prediction(input_vals, probs,
                            fname, n=4, title=''):
  """Save a PNG plot visualizing posterior uncertainty on heldout data.
  Args:
    input_vals: A `float`-like Numpy `array` of shape
      `[num_heldout] + IMAGE_SHAPE`, containing heldout input images.
    probs: A `float`-like Numpy array of shape `[num_monte_carlo,
      num_heldout, num_classes]` containing Monte Carlo samples of
      class probabilities for each heldout sample.
    fname: Python `str` filename to save the plot to.
    n: Python `int` number of datapoints to vizualize.
    title: Python `str` title for the plot.
  """
  fig = figure.Figure(figsize=(9, 3*n))
  canvas = backend_agg.FigureCanvasAgg(fig)
  for i in range(n):
    ax = fig.add_subplot(n, 3, 3*i + 1)
    # ax.imshow(input_vals[i, :].reshape(IMAGE_SHAPE[:-1]), interpolation='None')
    sns.barplot(np.arange(NB_CLASS), input_vals[i, :], ax=ax)
    ax.set_title('Input sample')
    ax = fig.add_subplot(n, 3, 3*i + 2)
    for prob_sample in probs:
      sns.barplot(np.arange(NB_CLASS), prob_sample[i, :], alpha=0.1, ax=ax)
      ax.set_ylim([0, 1])
    ax.set_title('Posterior samples')

    ax = fig.add_subplot(n, 3, 3*i + 3)
    sns.barplot(np.arange(NB_CLASS), np.mean(probs[:, i, :], axis=0), ax=ax)
    ax.set_ylim([0, 1])
    ax.set_title('Predictive sample')
  fig.suptitle(title)
  fig.tight_layout()

  canvas.print_figure(fname, format='png',dpi=1200)
  print('saved {}'.format(fname))
  
  
def multi_label_log_loss(y_pred, y_true):
    return K.sum(K.binary_crossentropy(y_pred, y_true), axis=-1)


def _average_gradient_norm(model, X_train, y_train, batch_size):
    # just checking if the model was already compiled
    if not hasattr(model, "train_function"):
        raise RuntimeError("You must compile your model before using it.")

    weights = model.trainable_weights  # weight tensors

    get_gradients = model.optimizer.get_gradients(model.total_loss, weights)  # gradient tensors

    input_tensors = [
        # input data
        model.inputs[0],
        # how much to weight each sample by
        model.sample_weights[0],
        # labels
        model.targets[0],
        # train or test mode
        K.learning_phase()
    ]

    grad_fct = K.function(inputs=input_tensors, outputs=get_gradients)

    steps = 0
    total_norm = 0
    s_w = None

    nb_steps = X_train.shape[0] // batch_size

    if X_train.shape[0] % batch_size == 0:
        pad_last = False
    else:
        pad_last = True

    def generator(X_train, y_train, pad_last):
        for i in range(nb_steps):
            X = X_train[i * batch_size: (i + 1) * batch_size, ...]
            y = y_train[i * batch_size: (i + 1) * batch_size, ...]

            yield (X, y)

        if pad_last:
            X = X_train[nb_steps * batch_size:, ...]
            y = y_train[nb_steps * batch_size:, ...]

            yield (X, y)

    datagen = generator(X_train, y_train, pad_last)

    while steps < nb_steps:
        X, y = next(datagen)
        # set sample weights to one
        # for every input
        if s_w is None:
            s_w = np.ones(X.shape[0])

        gradients = grad_fct([X, s_w, y, 0])
        total_norm += np.sqrt(np.sum([np.sum(np.square(g)) for g in gradients]))
        steps += 1

    if pad_last:
        X, y = next(datagen)
        # set sample weights to one
        # for every input
        if s_w is None:
            s_w = np.ones(X.shape[0])

        gradients = grad_fct([X, s_w, y, 0])
        total_norm += np.sqrt(np.sum([np.sum(np.square(g)) for g in gradients]))
        steps += 1

    return total_norm / float(steps)

def train_model_mlp(model:Model, dataset_id, dataset_prefix, dataset_fold_id=None, epochs=50, batch_size=128, val_subset=None,
                cutoff=None, normalize_timeseries=False, learning_rate=1e-4, monitor='val_loss', optimization_mode='auto', compile_model=True):
    X_train, y_train, X_test, y_test, is_timeseries,m_train,m_test = load_dataset_at_mlp(dataset_id,
                                                                      fold_index=dataset_fold_id,
                                                                      normalize_timeseries=normalize_timeseries)
    
    max_timesteps, max_nb_variables = calculate_dataset_metrics(X_train)

    if max_nb_variables != MAX_NB_VARIABLES[dataset_id]:
        if cutoff is None:
            choice = cutoff_choice(dataset_id, max_nb_variables)
        else:
            assert cutoff in ['pre', 'post'], 'Cutoff parameter value must be either "pre" or "post"'
            choice = cutoff

        if choice not in ['pre', 'post']:
            return
        else:
            X_train, X_test = cutoff_sequence(X_train, X_test, choice, dataset_id, max_nb_variables)

    classes = np.unique(y_train)
    le = LabelEncoder()
    y_ind = le.fit_transform(y_train.ravel())
    recip_freq = len(y_train) / (len(le.classes_) *
                           np.bincount(y_ind).astype(np.float64))
    class_weight = recip_freq[le.transform(classes)]

    print("Class weights : ", class_weight)

    y_train = to_categorical(y_train, len(np.unique(y_train)))
    y_test = to_categorical(y_test, len(np.unique(y_test)))

    if is_timeseries:
        factor = 1. / np.cbrt(2)
    else:
        factor = 1. / np.sqrt(2)

    if dataset_fold_id is None:
        weight_fn = "./weights/%s/weights_mlp.h5" % dataset_prefix
    else:
        weight_fn = "./weights/%s/%s_fold_%d_weights_Final_Paper_Long_Bridge_Traffic_vehicles_mlp.h5" % (dataset_prefix, dataset_prefix, dataset_fold_id)

    model_checkpoint = ModelCheckpoint(weight_fn, verbose=1, mode=optimization_mode,
                                       monitor=monitor, save_best_only=True,save_weights_only=True)
    reduce_lr = ReduceLROnPlateau(monitor=monitor, patience=10, mode=optimization_mode,
                                  factor=factor, cooldown=0, min_lr=1e-8, verbose=2)
    callback_list = [model_checkpoint, reduce_lr]

    optm = Adam(lr=learning_rate)

    if compile_model:
        model.compile(optimizer=optm, loss='categorical_crossentropy', metrics=['accuracy'])

    if val_subset is not None:
        X_test = X_test[:val_subset]
        y_test = y_test[:val_subset]
    
    X_train, y_train, m_train = shuffle(X_train, y_train,m_train, random_state=138) #138 standard 100
    X_test, y_test ,m_test  = shuffle(X_test, y_test,m_test, random_state=138)
    #x_train, x_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.33, random_state=5)
    
    for epochs_try in range(9):
        history=model.fit([X_train, m_train], y_train, batch_size=batch_size, epochs=epochs, callbacks=callback_list,
               verbose=2,shuffle=True, validation_data=([X_test, m_test], y_test))
        print(' ... Running monte carlo inference')
        probs = tf.stack([model.predict([X_test, m_test], verbose=1)
                            for _ in range(20)], axis=0)
        #mean_probs = tf.reduce_mean(probs, axis=0)
        #heldout_log_prob = tf.reduce_mean(tf.math.log(mean_probs))
        #print(' ... Held-out nats: {:.3f}'.format(heldout_log_prob))

        if HAS_SEABORN:
            names = [layer.name for layer in model.layers
                    if 'flipout' in layer.name]
            qm_vals = [layer.kernel_posterior.mean().numpy()
                      for layer in model.layers
                      if 'flipout' in layer.name]
            qs_vals = [layer.kernel_posterior.stddev().numpy()
                      for layer in model.layers
                      if 'flipout' in layer.name]
            plot_weight_posteriors(names, qm_vals, qs_vals,
                                  fname='D:/Zohaib_Phd Folder/Paper 2/Matlab Model/GVW_Paper/Python Folder/Bayesian ML/MLSTM-FCN-master/output/%s/epoch_weights_mlp{}.png'  .format(epochs_try) % dataset_prefix) 
            plot_heldout_prediction(y_test, probs.numpy(),
                                    fname='D:/Zohaib_Phd Folder/Paper 2/Matlab Model/GVW_Paper/Python Folder/Bayesian ML/MLSTM-FCN-master/output/%s/epoch_predictions_mlp{}.png' .format(epochs_try) % dataset_prefix
                                     )  
        
        
    return history

def evaluate_model_mlp(model:Model, dataset_id, dataset_prefix, dataset_fold_id=None, batch_size=128, test_data_subset=None,
                   cutoff=None, normalize_timeseries=False):
    _, _, X_test, y_test, is_timeseries,m_train,m_test = load_dataset_at_mlp(dataset_id,
                                                          fold_index=dataset_fold_id,
                                                          normalize_timeseries=normalize_timeseries)
    max_timesteps, max_nb_variables = calculate_dataset_metrics(X_test)

    if max_nb_variables != MAX_NB_VARIABLES[dataset_id]:
        if cutoff is None:
            choice = cutoff_choice(dataset_id, max_nb_variables)
        else:
            assert cutoff in ['pre', 'post'], 'Cutoff parameter value must be either "pre" or "post"'
            choice = cutoff

        if choice not in ['pre', 'post']:
            return
        else:
            _, X_test = cutoff_sequence(None, X_test, choice, dataset_id, max_nb_variables)

    if not is_timeseries:
        X_test = pad_sequences(X_test, maxlen=MAX_NB_VARIABLES[dataset_id], padding='post', truncating='post')
    y_test = to_categorical(y_test, len(np.unique(y_test)))

    optm = Adam(lr=1e-3)
    model.compile(optimizer=optm, loss='categorical_crossentropy', metrics=['accuracy'])

    if dataset_fold_id is None:
      weight_fn = "./weights/%s/weights_mlp.h5" % dataset_prefix
    else:
      weight_fn = "./weights/%s/%s_fold_%d_weights_Final_Paper_Long_Bridge_Traffic_vehicles_mlp.h5" % (dataset_prefix, dataset_prefix, dataset_fold_id)
    model.load_weights(weight_fn)

    if test_data_subset is not None:
        X_test = X_test[:test_data_subset]
        y_test = y_test[:test_data_subset]

    print("\nEvaluating : ")
    loss, accuracy = model.evaluate([X_test, m_test], y_test, batch_size=batch_size)
    print()
    print("Final Accuracy : ", accuracy)
    out=model.predict([X_test, m_test], batch_size=128)
    print("CLASS : ", out)
    print("original : ", y_test)
    
    return accuracy, loss, out, X_test, m_test, y_test

def train_model(model:Model, dataset_id, dataset_prefix, dataset_fold_id=None, epochs=50, batch_size=128, val_subset=None,cutoff=None, normalize_timeseries=False,
                learning_rate=1e-4, monitor='val_loss', optimization_mode='auto', compile_model=True):
    X_train, y_train, X_test, y_test, is_timeseries = load_dataset_at(dataset_id,
                                                                      fold_index=dataset_fold_id,
                                                                      normalize_timeseries=normalize_timeseries)
    
    max_timesteps, max_nb_variables = calculate_dataset_metrics(X_train)

    if max_nb_variables != MAX_NB_VARIABLES[dataset_id]:
        if cutoff is None:
            choice = cutoff_choice(dataset_id, max_nb_variables)
        else:
            assert cutoff in ['pre', 'post'], 'Cutoff parameter value must be either "pre" or "post"'
            choice = cutoff

        if choice not in ['pre', 'post']:
            return
        else:
            X_train, X_test = cutoff_sequence(X_train, X_test, choice, dataset_id, max_nb_variables)

    classes = np.unique(y_train)
    le = LabelEncoder()
    y_ind = le.fit_transform(y_train.ravel())
    recip_freq = len(y_train) / (len(le.classes_) *
                           np.bincount(y_ind).astype(np.float64))
    class_weight2 = recip_freq[le.transform(classes)]
    class_weight = {0: class_weight2[0],
                1: class_weight2[1]
                }
    print("Class weights : ", class_weight)

    y_train = to_categorical(y_train, len(np.unique(y_train)))
    y_test = to_categorical(y_test, len(np.unique(y_test)))

    if is_timeseries:
        factor = 1. / np.cbrt(2)
    else:
        factor = 1. / np.sqrt(2)

    if dataset_fold_id is None:
        weight_fn = "./weights/%s/weights.h5" % dataset_prefix
    else:
        weight_fn = "./weights/%s/%s_fold_%d_weights_Final_Paper_Long_Bridge_Traffic_vehicles.h5" % (dataset_prefix, dataset_prefix, dataset_fold_id)

    model_checkpoint = ModelCheckpoint(weight_fn, verbose=1, mode=optimization_mode,
                                       monitor=monitor, save_best_only=True,save_weights_only=True)
    reduce_lr = ReduceLROnPlateau(monitor=monitor, patience=10, mode=optimization_mode,
                                  factor=factor, cooldown=0, min_lr=1e-8, verbose=2)
    callback_list = [model_checkpoint, reduce_lr]

    optm = Adam(lr=learning_rate)

    if compile_model:
        model.compile(optimizer=optm, loss='binary_crossentropy', metrics=['accuracy'])

    if val_subset is not None:
        X_test = X_test[:val_subset]
        y_test = y_test[:val_subset]
    
    X_train, y_train = shuffle(X_train, y_train, random_state=138)
    X_test, y_test   = shuffle(X_test, y_test, random_state=138)
    #x_train, x_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.33, random_state=5)
    
    for epochs_try in range(9):
        history=model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, callbacks=callback_list,
               class_weight=class_weight,verbose=2,shuffle=True, validation_data=(X_test, y_test))
        print(' ... Running monte carlo inference')
        probs = tf.stack([model.predict(X_test, verbose=1)
                            for _ in range(20)], axis=0)
        #mean_probs = tf.reduce_mean(probs, axis=0)
        #heldout_log_prob = tf.reduce_mean(tf.math.log(mean_probs))
        #print(' ... Held-out nats: {:.3f}'.format(heldout_log_prob))

        if HAS_SEABORN:
            names = [layer.name for layer in model.layers
                    if 'flipout' in layer.name]
            qm_vals = [layer.kernel_posterior.mean().numpy()
                      for layer in model.layers
                      if 'flipout' in layer.name]
            qs_vals = [layer.kernel_posterior.stddev().numpy()
                      for layer in model.layers
                      if 'flipout' in layer.name]
            plot_weight_posteriors(names, qm_vals, qs_vals,
                                  fname='D:/Zohaib_Phd Folder/Paper 3/Experimental Data/MLSTM-FCN-master/output/%s/epoch_weights{}.png'  .format(epochs_try) % dataset_prefix) 
            plot_heldout_prediction(y_test, probs.numpy(),
                                    fname='D:/Zohaib_Phd Folder/Paper 3/Experimental Data/MLSTM-FCN-master/output//%s/epoch_predictions{}.png' .format(epochs_try) % dataset_prefix
                                     )  
        
        
    return history
def evaluate_model(model:Model, dataset_id, dataset_prefix, dataset_fold_id=None, batch_size=128, test_data_subset=None,
                   cutoff=None, normalize_timeseries=False):
    _, _, X_test, y_test, is_timeseries = load_dataset_at(dataset_id,
                                                          fold_index=dataset_fold_id,
                                                          normalize_timeseries=normalize_timeseries)
    max_timesteps, max_nb_variables = calculate_dataset_metrics(X_test)

    if max_nb_variables != MAX_NB_VARIABLES[dataset_id]:
        if cutoff is None:
            choice = cutoff_choice(dataset_id, max_nb_variables)
        else:
            assert cutoff in ['pre', 'post'], 'Cutoff parameter value must be either "pre" or "post"'
            choice = cutoff

        if choice not in ['pre', 'post']:
            return
        else:
            _, X_test = cutoff_sequence(None, X_test, choice, dataset_id, max_nb_variables)

    if not is_timeseries:
        X_test = pad_sequences(X_test, maxlen=MAX_NB_VARIABLES[dataset_id], padding='post', truncating='post')
    y_test = to_categorical(y_test, len(np.unique(y_test)))

    optm = Adam(lr=1e-3)
    model.compile(optimizer=optm, loss='binary_crossentropy', metrics=['accuracy'])

    if dataset_fold_id is None:
      weight_fn = "./weights/%s/weights.h5" % dataset_prefix
    else:
      weight_fn = "./weights/%s/%s_fold_%d_weights_Final_Paper_Long_Bridge_Traffic_vehicles.h5" % (dataset_prefix, dataset_prefix, dataset_fold_id)
    model.load_weights(weight_fn)

    if test_data_subset is not None:
        X_test = X_test[:test_data_subset]
        y_test = y_test[:test_data_subset]

    print("\nEvaluating : ")
    loss, accuracy = model.evaluate(X_test, y_test, batch_size=batch_size)
    print()
    print("Final Accuracy : ", accuracy)
    out=model.predict(X_test, batch_size=128)
    print("CLASS : ", out)
    print("original : ", y_test)
    
    return accuracy, loss, out, X_test,y_test

def set_trainable(layer, value):
   layer.trainable = value

   # case: container
   if hasattr(layer, 'layers'):
       for l in layer.layers:
           set_trainable(l, value)

   # case: wrapper (which is a case not covered by the PR)
   if hasattr(layer, 'layer'):
        set_trainable(layer.layer, value)


def compute_average_gradient_norm(model:Model, dataset_id, dataset_fold_id=None, batch_size=128,
                cutoff=None, normalize_timeseries=False, learning_rate=1e-3):
    X_train, y_train, X_test, y_test, is_timeseries = load_dataset_at(dataset_id,
                                                                      fold_index=dataset_fold_id,
                                                                      normalize_timeseries=normalize_timeseries)
    max_timesteps, sequence_length = calculate_dataset_metrics(X_train)

    if sequence_length != MAX_NB_VARIABLES[dataset_id]:
        if cutoff is None:
            choice = cutoff_choice(dataset_id, sequence_length)
        else:
            assert cutoff in ['pre', 'post'], 'Cutoff parameter value must be either "pre" or "post"'
            choice = cutoff

        if choice not in ['pre', 'post']:
            return
        else:
            X_train, X_test = cutoff_sequence(X_train, X_test, choice, dataset_id, sequence_length)

    y_train = to_categorical(y_train, len(np.unique(y_train)))

    optm = Adam(lr=learning_rate)
    model.compile(optimizer=optm, loss='categorical_crossentropy', metrics=['accuracy'])

    average_gradient = _average_gradient_norm(model, X_train, y_train, batch_size)
    print("Average gradient norm : ", average_gradient)


class MaskablePermute(Permute):

    def __init__(self, dims, **kwargs):
        super(MaskablePermute, self).__init__(dims, **kwargs)
        self.supports_masking = True


def f1_score(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)

    return 2 * ((precision * recall) / (precision + recall))

def train_model_mlp_nor(model:Model, dataset_id, dataset_prefix, dataset_fold_id=None, epochs=50, batch_size=128, val_subset=None,
                cutoff=None, normalize_timeseries=False, learning_rate=1e-4, monitor='val_loss', optimization_mode='auto', compile_model=True):
    X_train, y_train, X_test, y_test, is_timeseries,m_train,m_test = load_dataset_at_mlp(dataset_id,
                                                                      fold_index=dataset_fold_id,
                                                                      normalize_timeseries=normalize_timeseries)
    
    max_timesteps, max_nb_variables = calculate_dataset_metrics(X_train)

    if max_nb_variables != MAX_NB_VARIABLES[dataset_id]:
        if cutoff is None:
            choice = cutoff_choice(dataset_id, max_nb_variables)
        else:
            assert cutoff in ['pre', 'post'], 'Cutoff parameter value must be either "pre" or "post"'
            choice = cutoff

        if choice not in ['pre', 'post']:
            return
        else:
            X_train, X_test = cutoff_sequence(X_train, X_test, choice, dataset_id, max_nb_variables)

    classes = np.unique(y_train)
    le = LabelEncoder()
    y_ind = le.fit_transform(y_train.ravel())
    recip_freq = len(y_train) / (len(le.classes_) *
                           np.bincount(y_ind).astype(np.float64))
    class_weight = recip_freq[le.transform(classes)]

    print("Class weights : ", class_weight)

    y_train = to_categorical(y_train, len(np.unique(y_train)))
    y_test = to_categorical(y_test, len(np.unique(y_test)))

    if is_timeseries:
        factor = 1. / np.cbrt(2)
    else:
        factor = 1. / np.sqrt(2)

    if dataset_fold_id is None:
        weight_fn = "./weights/%s/weights_mlp_nor.h5" % dataset_prefix
    else:
        weight_fn = "./weights/%s/%s_fold_%d_weights_Final_Paper_Long_Bridge_Traffic_vehicles_mlp_nor.h5" % (dataset_prefix, dataset_prefix, dataset_fold_id)

    model_checkpoint = ModelCheckpoint(weight_fn, verbose=1, mode=optimization_mode,
                                       monitor=monitor, save_best_only=True,save_weights_only=True)
    reduce_lr = ReduceLROnPlateau(monitor=monitor, patience=5, mode=optimization_mode,
                                  factor=factor, cooldown=0, min_lr=1e-8, verbose=2)
    callback_list = [model_checkpoint, reduce_lr]

    optm = Adam(lr=learning_rate)

    if compile_model:
        model.compile(optimizer=optm, loss='categorical_crossentropy', metrics=['accuracy'])

    if val_subset is not None:
        X_test = X_test[:val_subset]
        y_test = y_test[:val_subset]
    
    X_train, y_train, m_train = shuffle(X_train, y_train,m_train, random_state=138)
    X_test, y_test ,m_test  = shuffle(X_test, y_test,m_test, random_state=138)
    #x_train, x_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.33, random_state=5)
    
    for epochs_try in range(5):
        history=model.fit([X_train, m_train], y_train, batch_size=batch_size, epochs=epochs, callbacks=callback_list,
               verbose=2,shuffle=True, validation_data=([X_test, m_test], y_test))
    
    return history

def evaluate_model_mlp_nor(model:Model, dataset_id, dataset_prefix, dataset_fold_id=None, batch_size=128, test_data_subset=None,
                   cutoff=None, normalize_timeseries=False):
    _, _, X_test, y_test, is_timeseries,m_train,m_test = load_dataset_at_mlp(dataset_id,
                                                          fold_index=dataset_fold_id,
                                                          normalize_timeseries=normalize_timeseries)
    max_timesteps, max_nb_variables = calculate_dataset_metrics(X_test)

    if max_nb_variables != MAX_NB_VARIABLES[dataset_id]:
        if cutoff is None:
            choice = cutoff_choice(dataset_id, max_nb_variables)
        else:
            assert cutoff in ['pre', 'post'], 'Cutoff parameter value must be either "pre" or "post"'
            choice = cutoff

        if choice not in ['pre', 'post']:
            return
        else:
            _, X_test = cutoff_sequence(None, X_test, choice, dataset_id, max_nb_variables)

    if not is_timeseries:
        X_test = pad_sequences(X_test, maxlen=MAX_NB_VARIABLES[dataset_id], padding='post', truncating='post')
    y_test = to_categorical(y_test, len(np.unique(y_test)))

    optm = Adam(lr=1e-3)
    model.compile(optimizer=optm, loss='categorical_crossentropy', metrics=['accuracy'])

    if dataset_fold_id is None:
      weight_fn = "./weights/%s/weights_mlp_nor.h5" % dataset_prefix
    else:
      weight_fn = "./weights/%s/%s_fold_%d_weights_Final_Paper_Long_Bridge_Traffic_vehicles_mlp_nor.h5" % (dataset_prefix, dataset_prefix, dataset_fold_id)
    model.load_weights(weight_fn)

    if test_data_subset is not None:
        X_test = X_test[:test_data_subset]
        y_test = y_test[:test_data_subset]

    print("\nEvaluating : ")
    loss, accuracy = model.evaluate([X_test, m_test], y_test, batch_size=batch_size)
    print()
    print("Final Accuracy : ", accuracy)
    out=model.predict([X_test, m_test], batch_size=128)
    print("CLASS : ", out)
    print("original : ", y_test)
    
    return accuracy, loss, out, X_test, m_test, y_test
def train_model_autoencoder(model:Model, dataset_id, dataset_prefix, dataset_fold_id=None, epochs=50, batch_size=128, val_subset=None,cutoff=None, normalize_timeseries=False,
                learning_rate=1e-3, monitor='val_loss', optimization_mode='auto', compile_model=True):
    X_train, y_train, X_test, y_test,_,_,_,_, is_timeseries = load_dataset_at(dataset_id,
                                                                      fold_index=dataset_fold_id,
                                                                      normalize_timeseries=normalize_timeseries)
    
    max_timesteps, max_nb_variables = calculate_dataset_metrics(X_train)

    if max_nb_variables != MAX_NB_VARIABLES[dataset_id]:
        if cutoff is None:
            choice = cutoff_choice(dataset_id, max_nb_variables)
        else:
            assert cutoff in ['pre', 'post'], 'Cutoff parameter value must be either "pre" or "post"'
            choice = cutoff

        if choice not in ['pre', 'post']:
            return
        else:
            X_train, X_test = cutoff_sequence(X_train, X_test, choice, dataset_id, max_nb_variables)


    if is_timeseries:
        factor = 1. / np.cbrt(2)
    else:
        factor = 1. / np.sqrt(2)

    if dataset_fold_id is None:
        weight_fn = "./weights/%s/weights_new_05.h5" % dataset_prefix
    else:
        weight_fn = "./weights/%s/%s_fold_%d_weights_Final_Paper_Long_Bridge_Traffic_vehicles.h5" % (dataset_prefix, dataset_prefix, dataset_fold_id)

    model_checkpoint = ModelCheckpoint(weight_fn, verbose=1, mode=optimization_mode,
                                       monitor=monitor, save_best_only=True,save_weights_only=True)
    reduce_lr = ReduceLROnPlateau(monitor=monitor, patience=200, mode=optimization_mode,
                                  factor=factor, cooldown=0, min_lr=1e-7, verbose=2)
    callback_list = [model_checkpoint, reduce_lr]

    optm = Adam(lr=learning_rate)

    if compile_model:
        model.compile(optimizer=optm, loss='mse', metrics=['mse'])

    if val_subset is not None:
        X_test = X_test[:val_subset]
        y_test = y_test[:val_subset]
    
    # X_train, y_train = shuffle(X_train, y_train, random_state=400)
    # X_test, y_test   = shuffle(X_test, y_test, random_state=400)
    #x_train, x_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.33, random_state=5)
    
    # X_train=np.transpose(X_train, (0, 2, 1))
    # X_test=np.transpose(X_test, (0, 2, 1))
    
    
   

    #save the model history in a list after fitting so that we can plot later

    print("======="*12, end="\n\n\n")
    history=model.fit(X_train, X_train, batch_size=batch_size, epochs=epochs, callbacks=callback_list,
                          verbose=2,shuffle=True, validation_data=(X_test, X_test))
          
    return history
def train_model_autoencoder_Prob(model:Model, dataset_id, dataset_prefix, dataset_fold_id=None, epochs=50, batch_size=128, val_subset=None,cutoff=None, normalize_timeseries=False,
                learning_rate=1e-3, monitor='val_loss', optimization_mode='auto', compile_model=True):
    X_train, y_train, X_test, y_test,_,_,_,_, is_timeseries = load_dataset_at(dataset_id,
                                                                      fold_index=dataset_fold_id,
                                                                      normalize_timeseries=normalize_timeseries)
    
    max_timesteps, max_nb_variables = calculate_dataset_metrics(X_train)

    if max_nb_variables != MAX_NB_VARIABLES[dataset_id]:
        if cutoff is None:
            choice = cutoff_choice(dataset_id, max_nb_variables)
        else:
            assert cutoff in ['pre', 'post'], 'Cutoff parameter value must be either "pre" or "post"'
            choice = cutoff

        if choice not in ['pre', 'post']:
            return
        else:
            X_train, X_test = cutoff_sequence(X_train, X_test, choice, dataset_id, max_nb_variables)


    if is_timeseries:
        factor = 1. / np.cbrt(2)
    else:
        factor = 1. / np.sqrt(2)

    if dataset_fold_id is None:
        weight_fn = "./weights/%s/weights_new_05.h5" % dataset_prefix
    else:
        weight_fn = "./weights/%s/%s_fold_%d_weights_Final_Paper_Long_Bridge_Traffic_vehicles.h5" % (dataset_prefix, dataset_prefix, dataset_fold_id)

    model_checkpoint = ModelCheckpoint(weight_fn, verbose=1, mode=optimization_mode,
                                       monitor=monitor, save_best_only=True,save_weights_only=True)
    reduce_lr = ReduceLROnPlateau(monitor=monitor, patience=40, mode=optimization_mode,
                                  factor=factor, cooldown=0, min_lr=1e-7, verbose=2)
    callback_list = [model_checkpoint, reduce_lr]

    optm = Adam(lr=learning_rate)

    if compile_model:
        model.compile(optimizer=optm, loss='mse', metrics=['mse'])

    if val_subset is not None:
        X_test = X_test[:val_subset]
        y_test = y_test[:val_subset]
    
    # X_train, y_train = shuffle(X_train, y_train, random_state=400)
    # X_test, y_test   = shuffle(X_test, y_test, random_state=400)
    #x_train, x_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.33, random_state=5)
    
    # X_train=np.transpose(X_train, (0, 2, 1))
    # X_test=np.transpose(X_test, (0, 2, 1))
    
    
   

    #save the model history in a list after fitting so that we can plot later

    for epochs_try in range(9):
        history=model.fit(X_train, X_train, batch_size=batch_size, epochs=epochs, callbacks=callback_list,
                          verbose=2,shuffle=True, validation_data=(X_test, X_test))
        print(' ... Running monte carlo inference')
        probs = tf.stack([model.predict(X_test, verbose=1)
                            for _ in range(10)], axis=0)
        #mean_probs = tf.reduce_mean(probs, axis=0)
        #heldout_log_prob = tf.reduce_mean(tf.math.log(mean_probs))
        #print(' ... Held-out nats: {:.3f}'.format(heldout_log_prob))

        if HAS_SEABORN:
            names = [layer.name for layer in model.layers
                    if 'flipout' in layer.name]
            qm_vals = [layer.kernel_posterior.mean().numpy()
                      for layer in model.layers
                      if 'flipout' in layer.name]
            qs_vals = [layer.kernel_posterior.stddev().numpy()
                      for layer in model.layers
                      if 'flipout' in layer.name]
            plot_weight_posteriors(names, qm_vals, qs_vals,
                                  fname='D:/Zohaib_Phd Folder/Paper 3/Experimental Data/MLSTM-FCN-master/output/%s/epoch_weights{}.png'  .format(epochs_try) % dataset_prefix) 
            # plot_heldout_prediction(y_test, probs.numpy(),
            #                         fname='D:/Zohaib_Phd Folder/Paper 3/Experimental Data/MLSTM-FCN-master/output//%s/epoch_predictions{}.png' .format(epochs_try) % dataset_prefix
            #                          )  
        
        
    return history,probs
def evaluate_model_autoencoder(model:Model, dataset_id, dataset_prefix, dataset_fold_id=None, batch_size=128, test_data_subset=None,
                   cutoff=None, normalize_timeseries=False):
    X_train, y_train, X_test, y_test,X_test_Vali,y_test_Vali,X_test_Reh,y_test_Reh, is_timeseries = load_dataset_at(dataset_id,
                                                          fold_index=dataset_fold_id,
                                                          normalize_timeseries=normalize_timeseries)
    max_timesteps, max_nb_variables = calculate_dataset_metrics(X_test)

    if max_nb_variables != MAX_NB_VARIABLES[dataset_id]:
        if cutoff is None:
            choice = cutoff_choice(dataset_id, max_nb_variables)
        else:
            assert cutoff in ['pre', 'post'], 'Cutoff parameter value must be either "pre" or "post"'
            choice = cutoff

        if choice not in ['pre', 'post']:
            return
        else:
            _, X_test = cutoff_sequence(None, X_test, choice, dataset_id, max_nb_variables)

    if not is_timeseries:
        X_test = pad_sequences(X_test, maxlen=MAX_NB_VARIABLES[dataset_id], padding='post', truncating='post')
       # y_test = to_categorical(y_test, len(np.unique(y_test)))

    optm = Adam(lr=1e-3)
    model.compile(optimizer=optm, loss='mse', metrics=['mse'])

    if dataset_fold_id is None:
      weight_fn = "./weights/%s/weights_new_05.h5" % dataset_prefix
    else:
      weight_fn = "./weights/%s/%s_fold_%d_weights_Final_Paper_Long_Bridge_Traffic_vehicles.h5" % (dataset_prefix, dataset_prefix, dataset_fold_id)
    model.load_weights(weight_fn)

    if test_data_subset is not None:
        X_test = X_test[:test_data_subset]
        y_test = y_test[:test_data_subset]

    print("\nEvaluating : ")
    # X_train=np.transpose(X_train, (0, 2, 1))
    # X_test=np.transpose(X_test, (0, 2, 1)) 
    # X_test_Vali=np.transpose(X_test_Vali, (0, 2, 1))  
    # X_test_Reh=np.transpose(X_test_Reh, (0, 2, 1))  

    
    
    X_predict= model.predict(X_test, batch_size=batch_size)
    X_predict_train= model.predict(X_train, batch_size=batch_size)
    X_predict_Vali= model.predict(X_test_Vali, batch_size=batch_size)
    X_predict_Reh= model.predict(X_test_Reh, batch_size=batch_size)

    return X_test,y_test,X_predict,X_train,X_predict_train,X_test_Vali,X_predict_Vali,X_test_Reh,X_predict_Reh


def train_model_Varational(model:Model,encoder, decoder, encoder_var,encoder_mu, encoder_log_variance, dataset_id, dataset_prefix, dataset_fold_id=None, epochs=50, batch_size=128, val_subset=None,cutoff=None, normalize_timeseries=False,
                learning_rate=1e-4, monitor='val_loss', optimization_mode='auto', compile_model=True):
    X_train, y_train, X_test, y_test,_,_,_,_, is_timeseries = load_dataset_at(dataset_id,
                                                                      fold_index=dataset_fold_id,
                                                                      normalize_timeseries=normalize_timeseries)
    
    max_timesteps, max_nb_variables = calculate_dataset_metrics(X_train)

    if max_nb_variables != MAX_NB_VARIABLES[dataset_id]:
        if cutoff is None:
            choice = cutoff_choice(dataset_id, max_nb_variables)
        else:
            assert cutoff in ['pre', 'post'], 'Cutoff parameter value must be either "pre" or "post"'
            choice = cutoff

        if choice not in ['pre', 'post']:
            return
        else:
            X_train, X_test = cutoff_sequence(X_train, X_test, choice, dataset_id, max_nb_variables)


    if is_timeseries:
        factor = 1. / np.cbrt(2)
    else:
        factor = 1. / np.sqrt(2)

    if dataset_fold_id is None:
        weight_fn = "./weights/%s/weights_Varational.h5" % dataset_prefix
    else:
        weight_fn = "./weights/%s/%s_fold_%d_weights_Final_Paper_Long_Bridge_Traffic_vehicles.h5" % (dataset_prefix, dataset_prefix, dataset_fold_id)

    model_checkpoint = ModelCheckpoint(weight_fn, verbose=1, mode=optimization_mode,
                                       monitor=monitor, save_best_only=True,save_weights_only=True)
    reduce_lr = ReduceLROnPlateau(monitor=monitor, patience=40, mode=optimization_mode,
                                  factor=factor, cooldown=0, min_lr=1e-7, verbose=2)
    callback_list = [model_checkpoint, reduce_lr]

    optm = Adam(lr=learning_rate)
    
    def loss_func(encoder_mu, encoder_log_variance):
       
        def vae_reconstruction_loss(y_true, y_predict):
            reconstruction_loss_factor = 1
            reconstruction_loss = tensorflow.keras.backend.mean(tensorflow.keras.backend.square(y_true-y_predict),axis=-1)
            return reconstruction_loss_factor * reconstruction_loss

        def vae_kl_loss(encoder_mu, encoder_log_variance):
            kl_loss = -0.5 * tensorflow.keras.backend.sum(1.0 + encoder_log_variance - tensorflow.keras.backend.square(encoder_mu) - tensorflow.keras.backend.exp(encoder_log_variance), axis=-1)
            return kl_loss

        def vae_kl_loss_metric(y_true, y_predict):
            kl_loss = -0.5 * tensorflow.keras.backend.sum(1.0 + encoder_log_variance - tensorflow.keras.backend.square(encoder_mu) - tensorflow.keras.backend.exp(encoder_log_variance), axis=-1)
            return kl_loss

        def vae_loss(y_true, y_predict):
            reconstruction_loss = vae_reconstruction_loss(y_true, y_predict)
            kl_loss = vae_kl_loss(y_true, y_predict)

            loss = reconstruction_loss + kl_loss
            return loss

            return vae_loss


    if compile_model:
        model.compile(optimizer=optm, loss='mse')

    if val_subset is not None:
        X_test = X_test[:val_subset]
        y_test = y_test[:val_subset]
    
    # X_train, y_train = shuffle(X_train, y_train, random_state=160)
    # X_test, y_test   = shuffle(X_test, y_test, random_state=160)
    #x_train, x_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.33, random_state=5)
    
    X_train=np.transpose(X_train, (0, 2, 1))
    X_test=np.transpose(X_test, (0, 2, 1))
    
    
   

    #save the model history in a list after fitting so that we can plot later

    print("======="*12, end="\n\n\n")
    history=model.fit(X_train, X_train, batch_size=batch_size, epochs=epochs, callbacks=callback_list,
                          verbose=2,shuffle=True, validation_data=(X_test, X_test))
          
    return history
def evaluate_model_autoencoder_prob(model:Model, dataset_id, dataset_prefix, dataset_fold_id=None, batch_size=128, test_data_subset=None,
                   cutoff=None, normalize_timeseries=False):
    X_train, y_train, X_test, y_test,X_test_Vali,y_test_Vali,X_test_Reh,y_test_Reh, is_timeseries = load_dataset_at(dataset_id,
                                                          fold_index=dataset_fold_id,
                                                          normalize_timeseries=normalize_timeseries)
    max_timesteps, max_nb_variables = calculate_dataset_metrics(X_test)

    if max_nb_variables != MAX_NB_VARIABLES[dataset_id]:
        if cutoff is None:
            choice = cutoff_choice(dataset_id, max_nb_variables)
        else:
            assert cutoff in ['pre', 'post'], 'Cutoff parameter value must be either "pre" or "post"'
            choice = cutoff

        if choice not in ['pre', 'post']:
            return
        else:
            _, X_test = cutoff_sequence(None, X_test, choice, dataset_id, max_nb_variables)

    if not is_timeseries:
        X_test = pad_sequences(X_test, maxlen=MAX_NB_VARIABLES[dataset_id], padding='post', truncating='post')
       # y_test = to_categorical(y_test, len(np.unique(y_test)))

    optm = Adam(lr=1e-3)
    model.compile(optimizer=optm, loss='mse', metrics=['mse'])

    if dataset_fold_id is None:
      weight_fn = "./weights/%s/weights_new_05.h5" % dataset_prefix
    else:
      weight_fn = "./weights/%s/%s_fold_%d_weights_Final_Paper_Long_Bridge_Traffic_vehicles.h5" % (dataset_prefix, dataset_prefix, dataset_fold_id)
    model.load_weights(weight_fn)

    if test_data_subset is not None:
        X_test = X_test[:test_data_subset]
        y_test = y_test[:test_data_subset]

    print("\nEvaluating : ")
    # X_train=np.transpose(X_train, (0, 2, 1))
    # X_test=np.transpose(X_test, (0, 2, 1)) 
    # X_test_Vali=np.transpose(X_test_Vali, (0, 2, 1))  
    # X_test_Reh=np.transpose(X_test_Reh, (0, 2, 1))  


    X_predict= tf.stack([model.predict(X_test, verbose=1)
                        for _ in range(50)], axis=0)
    X_predict_train= tf.stack([model.predict(X_train, verbose=1)
                        for _ in range(50)], axis=0)
    
    X_predict_Vali= tf.stack([model.predict(X_test_Vali, verbose=1)
                        for _ in range(50)], axis=0)
    X_predict_Reh=  tf.stack([model.predict(X_test_Reh, verbose=1)
                        for _ in range(50)], axis=0)
    return X_test,y_test,X_predict,X_train,X_predict_train,X_test_Vali,X_predict_Vali,X_test_Reh,X_predict_Reh