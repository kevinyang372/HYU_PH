import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import GridSearchCV, train_test_split
import keras
from keras.models import Sequential, load_model
from sklearn.externals import joblib
from keras.layers.core import Dense, Activation, Dropout
from keras.utils import np_utils
from scipy.stats import ks_2samp
import talos as ta
from keras.optimizers import Adam, Nadam, RMSprop
from keras.losses import logcosh, binary_crossentropy
from keras.activations import relu, elu, sigmoid
from talos.model.layers import hidden_layers
from talos.model.normalizers import lr_normalizer
from talos import Deploy
import os
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))

full_train_res = pd.read_csv('../HYU_data/full_train_res.csv', sep='\t',index_col=0)

y_pred = full_train_res['result']
full_train = full_train_res.drop('result',1)

def preprocess_data(X, scaler=None):
    if not scaler:
        scaler = StandardScaler()
        scaler.fit(X)
    X = scaler.transform(X)
    return X, scaler

train, scaler = preprocess_data(full_train)

def higgs_nn(X_train, Y_train, X_valid, Y_valid, params):
    model = Sequential()
    model.add(Dense(75, input_dim=X_train.shape[1],
                    activation=params['activation'],
                    kernel_initializer='normal'))
    model.add(Dropout(params['dropout']))

    hidden_layers(model, params, 1)
    
    model.add(Dense(1, activation=params['last_activation'],
                    kernel_initializer='normal'))
    
    model.compile(loss=params['losses'],
                  optimizer=params['optimizer'](lr=lr_normalizer(params['lr'],params['optimizer'])),
                  metrics=['acc'])
    
    history = model.fit(X_train, Y_train, 
                        validation_data=[X_valid, Y_valid],
                        batch_size=params['batch_size'],
                        epochs=params['epochs'],
                        verbose=0)
    
    # finally we have to make sure that history object and model are returned
    return history, model  

p = {'lr': (0.5, 5, 10),
     'first_neuron':[4, 8, 16, 32, 64],
     'hidden_layers':[2, 3, 4, 5, 6],
     'batch_size': (2, 10, 30),
     'epochs': [70],
     'dropout': (0, 0.5, 5),
     'weight_regulizer':[None],
     'emb_output_dims': [None],
     'shape':['brick','long_funnel'],
     'optimizer': [Adam, Nadam, RMSprop],
     'losses': [logcosh, binary_crossentropy],
     'activation':[relu, elu],
     'last_activation': [sigmoid]}

h = ta.Scan(train, np.array(y_pred.tolist()), params=p,
            model=higgs_nn,
            dataset_name='higgs_nn',
            experiment_no='2',
            grid_downsample=0.1,
	    val_split=0.3)


Deploy(h, 'higgs_nn')
