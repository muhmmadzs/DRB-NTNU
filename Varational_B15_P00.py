import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras import backend as K
from utils.constants import MAX_NB_VARIABLES, NB_CLASSES_LIST, MAX_TIMESTEPS_LIST
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.models import Model
from utils.generic_utils import load_dataset_at, calculate_dataset_metrics

import seaborn as sns

try:
  import seaborn as sns  # pylint: disable=g-import-not-at-top
  HAS_SEABORN = True
except ImportError:
  HAS_SEABORN = False
dataset_prefix='FEM Bridge/B15/PA00'
  
DATASET_INDEX = 8
import matplotlib
#matplotlib.use("pgf")
size=12
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
    'ytick.labelsize' : size,
    'xtick.labelsize' : size,
    'legend.fontsize' : 12,
    'legend.title_fontsize':12,
    'font.size':size,
    'axes.labelsize': size,
})

MAX_TIMESTEPS = MAX_TIMESTEPS_LIST[DATASET_INDEX]
MAX_NB_VARIABLES = MAX_NB_VARIABLES[DATASET_INDEX]
NB_CLASS = NB_CLASSES_LIST[DATASET_INDEX]
# Generate a simple time series dataset
dataset_id=DATASET_INDEX
dataset_fold_id=None

X_train, y_train, X_test, y_test,X_test_Vali,y_test_Vali,X_test_Reh,y_test_Reh, X_test_Reh2,y_test_Reh2,is_timeseries = load_dataset_at(dataset_id,  fold_index=dataset_fold_id,normalize_timeseries=True)

latent_space=16
batch_size=64
epochs=1000
# Build the  model

##
input_sig = keras.layers.Input(shape=(MAX_TIMESTEPS, MAX_NB_VARIABLES))


a_x = keras.layers.Conv1D(512,8,activation='relu', padding='same')(input_sig)    
x3 = keras.layers.MaxPooling1D(3)(a_x) 



b_x = keras.layers.Conv1D(256,5,activation='relu', padding='same')(x3)   
x3 = keras.layers.MaxPooling1D(2)(b_x) 

c_x = keras.layers.Conv1D(64,3, activation='relu', padding='same')(x3)    
x3  = keras.layers.MaxPooling1D(2)(c_x) 


d_x = keras.layers.Conv1D(32,3, activation='relu', padding='same')(x3)   
x3  = keras.layers.MaxPooling1D(1)(d_x) 

###uncomment it to use in the model 
# x3 =keras.layers.LSTM(32,return_sequences=True)(x3)
# x3 =keras.layers.Dropout(0.1)(x3)
# x3 =keras.layers.LSTM(16,return_sequences=True)(x3)

encoded = keras.layers.Flatten()(x3) 
# encoded = Dense(2000,activation = 'relu')(flat) 

encoded =keras.layers. Dense(16,activation = 'linear',kernel_regularizer=tf.keras.regularizers.l2(1e-3))(encoded) 

##
print("shape of encoded {}".format(K.int_shape(encoded))) 

dec_1 = keras.layers.Dense(4000,activation = 'relu')(encoded) 

dec_1= keras.layers.Reshape((125,32))(dec_1)

dec_1 = keras.layers.UpSampling1D(2)(dec_1) 
dec_1 = keras.layers.Conv1D(64, 3,activation='relu', padding='same')(dec_1) 


dec_1 = keras.layers.UpSampling1D(2)(dec_1) 
dec_1 = keras.layers.Conv1D(256, 3,activation='relu', padding='same')(dec_1) 

dec_1 = keras.layers.UpSampling1D(3)(dec_1) 
dec_1 = keras.layers.Conv1D(512, 3,activation='relu', padding='same')(dec_1) 



decoded = keras.layers.Conv1D(MAX_NB_VARIABLES, 1, padding='same', activation = 'linear')(dec_1) 

model = Model(input_sig, decoded) 
encoder=Model(input_sig,encoded)
model.summary()


##

# Compile the VAE model
optm = Adam(lr=1e-3)
model.compile(loss='mean_squared_error',  # categorical_crossentropy, mean_absolute_error, mean_absolute_error
              optimizer=optm,  # RMSprop(), Adagrad, Nadam, Adagrad, Adadelta, Adam, Adamax,
              metrics=['mse'])

# optm = Adam(lr=1e-3)
# model.compile(optimizer=optm,metrics=['mse'] )

if is_timeseries:
     factor = 1. / np.cbrt(2)
else:
     factor = 1. / np.sqrt(2)

if dataset_fold_id is None:
   weight_fn = "./weights/%s/weights_new_var.h5" % dataset_prefix
else:
   weight_fn = "./weights/%s/%s_fold_%d_weights_Final_Paper_Long_Bridge_Traffic_vehicles.h5" % (dataset_prefix, dataset_prefix, dataset_fold_id)
model_checkpoint = ModelCheckpoint(weight_fn, verbose=1, mode='auto',
                                       monitor='val_loss', save_best_only=True,save_weights_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=25, mode='auto',
                                  factor=factor, cooldown=0, min_lr=1e-6, verbose=2)
callback_list = [model_checkpoint, reduce_lr]
# Train the VAE model
# vae.fit(X_train, X_train,epochs=50, batch_size=128)

history=model.fit(X_train, X_train, batch_size=batch_size, epochs=epochs, callbacks=callback_list,
                  verbose=2,shuffle=True, validation_data=(X_test, X_test))



def evaluate_model_autoencoder(model:Model, dataset_id, dataset_prefix, dataset_fold_id=None, batch_size=128, test_data_subset=None,
                    normalize_timeseries=True):
    X_train, y_train, X_test, y_test,X_test_Vali,y_test_Vali,X_test_Reh,y_test_Reh,X_test_Reh2,y_test_Reh2, is_timeseries = load_dataset_at(dataset_id,
                                                          fold_index=dataset_fold_id,
                                                          normalize_timeseries=normalize_timeseries)
    max_timesteps, max_nb_variables = calculate_dataset_metrics(X_test)



    if not is_timeseries:
        X_test = pad_sequences(X_test, maxlen=MAX_NB_VARIABLES[dataset_id], padding='post', truncating='post')
       # y_test = to_categorical(y_test, len(np.unique(y_test)))

    optm = Adam(lr=1e-3)
    model.compile(optimizer=optm, metrics=['mse'])

    if dataset_fold_id is None:
      weight_fn = "./weights/%s/weights_new_var.h5" % dataset_prefix
    else:
      weight_fn = "./weights/%s/%s_fold_%d_weights_Final_Paper_Long_Bridge_Traffic_vehicles.h5" % (dataset_prefix, dataset_prefix, dataset_fold_id)
    model.load_weights(weight_fn)

    if test_data_subset is not None:
        X_test = X_test[:test_data_subset]
        y_test = y_test[:test_data_subset]

    print("\nEvaluating : ")

    
    
    X_predict= model.predict(X_test, batch_size=batch_size)
    X_predict_train= model.predict(X_train, batch_size=batch_size)
    X_predict_Vali= model.predict(X_test_Vali, batch_size=batch_size)
    X_predict_Reh= model.predict(X_test_Reh, batch_size=batch_size)
    X_predict_Reh2= model.predict(X_test_Reh2, batch_size=batch_size)

    return X_test,y_test,X_predict,X_train,X_predict_train,X_test_Vali,X_predict_Vali,X_test_Reh,X_predict_Reh,X_test_Reh2,X_predict_Reh2


if __name__ == "__main__":
  
   x,y,x_pred,xx,xx_pred,yy,yy_pred,re,re_pred,re2,re_pred2=evaluate_model_autoencoder(model, dataset_id, dataset_prefix, dataset_fold_id=None, batch_size=128, test_data_subset=None,
                    normalize_timeseries=True)


   ##*******************************
   
   Fleet_size=0
   Train_size=0
   N=0
   N1=0
   
   #**********************************Data_Modelling********************************************#
   x_pred=x_pred
   xx_pred=xx_pred
   re_pred=re_pred
   yy_pred=yy_pred
   re_pred2=re_pred2
   #**********************************Reconstruction loss*****************************************#
   MS=[]
   Out=[]
   out3=[]
   out4=[]
   tr=[]
   out5=[]
   for i in range(len(xx)):
       kk2 = (mean_absolute_error(xx[i,:,0],(xx_pred[i,:,0])))
       Out.append(kk2)

   for i in range(len(xx)):
         
       k = (mean_absolute_error(xx[i+Train_size+N1,:,0],(xx_pred[i+Train_size+N1,:,0])))
       tr.append(k)           
       
   for i in range((len(x))):
       kk =  (mean_absolute_error(x[i+N,:,0],(x_pred[i+N,:,0])))
       MS.append(kk)
       
   for i in range((len(yy))):   
       kk3 = (mean_absolute_error(yy[i+N,:,0],(yy_pred[i+N,:,0])))
       out3.append(kk3)
       
   for i in range((len(re))):   
       kk4 = (mean_absolute_error(re[i+N,:,0],(re_pred[i+N,:,0])))
       out4.append(kk4)
   for i in range((len(re2))):   
       kk5 = (mean_absolute_error(re2[i+N,:,0],(re_pred2[i+N,:,0])))
       out5.append(kk5)
               
   np.save('./output/%s/S1_Train.npy' %dataset_prefix, tr)
   np.save('./output/%s/S1_Vali.npy' %dataset_prefix, MS)
   np.save('./output/%s/S1_D1.npy' %dataset_prefix, out3)
   np.save('./output/%s/S1_D2.npy' %dataset_prefix, out4)
   np.save('./output/%s/S1_D3.npy' %dataset_prefix, out5)
   


   MS1=[]
   Out1=[]
   out31=[]
   out41=[]
   tr1=[]
   out51=[]
   for i in range(len(xx)):
       kk2 = (mean_absolute_error(xx[i,:,1],(xx_pred[i,:,1])))
       Out1.append(kk2)
   for i in range(len(xx)):
         
       k = (mean_absolute_error(xx[i+Train_size+N1,:,1],(xx_pred[i+Train_size+N1,:,1])))
       tr1.append(k)   
       
   for i in range(len(x)):
       kk =  (mean_absolute_error(x[i+N,:,1],(x_pred[i+N,:,1])))
       MS1.append(kk)
       
   for i in range(len(yy)):   
       kk3 = (mean_absolute_error(yy[i+N,:,1],(yy_pred[i+N,:,1])))
       out31.append(kk3)
       
   for i in range(len(re)):   
       kk4 = (mean_absolute_error(re[i+N,:,1],(re_pred[i+N,:,1])))
       out41.append(kk4)
       
   for i in range(len(re2)):   
       kk5 = (mean_absolute_error(re2[i+N,:,1],(re_pred2[i+N,:,1])))
       out51.append(kk5)     
   np.save('./output/%s/S2_Train.npy' %dataset_prefix, tr1)
   np.save('./output/%s/S2_Vali.npy' %dataset_prefix, MS1)
   np.save('./output/%s/S2_D1.npy' %dataset_prefix, out31)
   np.save('./output/%s/S2_D2.npy' %dataset_prefix, out41)   
   np.save('./output/%s/S2_D3.npy' %dataset_prefix, out51)   

   
   MS2=[]
   Out2=[]
   out32=[]
   out42=[]
   tr2=[]
   out52=[]
   for i in range(len(xx)):
       kk2 = (mean_absolute_error(xx[i,:,2],(xx_pred[i,:,2])))
       Out2.append(kk2)
       
   for i in range(len(xx)):
         
       k = (mean_absolute_error(xx[i+Train_size+N1,:,2],(xx_pred[i+Train_size+N1,:,2])))
       tr2.append(k)      
       
   for i in range(len(x)):
       kk =  (mean_absolute_error(x[i+N,:,2],(x_pred[i+N,:,2])))
       MS2.append(kk)       
       
   for i in range(len(yy)):    
       kk3 = (mean_absolute_error(yy[i+N,:,2],(yy_pred[i+N,:,2])))
       out32.append(kk3)
       
   for i in range(len(re)):    
       kk4 = (mean_absolute_error(re[i+N,:,2],(re_pred[i+N,:,2])))
       out42.append(kk4)
   for i in range(len(re2)):    
       kk5 = (mean_absolute_error(re2[i+N,:,2],(re_pred2[i+N,:,2])))
       out52.append(kk5)          
   np.save('./output/%s/S3_Train.npy' %dataset_prefix, tr2)
   np.save('./output/%s/S3_Vali.npy' %dataset_prefix, MS2)
   np.save('./output/%s/S3_D1.npy' %dataset_prefix, out32)
   np.save('./output/%s/S3_D2.npy' %dataset_prefix, out42)  
   np.save('./output/%s/S3_D3.npy' %dataset_prefix, out52)          
        
#*******************************************************File ended*****************************************************
