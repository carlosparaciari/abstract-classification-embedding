from sklearn.model_selection import ParameterGrid, KFold
import multiprocessing

import pandas as pd
import numpy as np

import tensorflow as tf

## Load the dataset
X = np.load('./X_train.npy',allow_pickle=True)
y = np.load('./y_train.npy',allow_pickle=True)

labels = np.load('./labels.npy',allow_pickle=True)
output_classes = len(labels)

embedding_weigths = np.load('./embedding_weights.npy',allow_pickle=True)
vocab_size, k = embedding_weigths.shape

## Create parameter grid
params = {}

params['n_filters'] = [100,200,300,400,500]
params['window_size'] = [2,3,4,5]
params['dropout_prob'] = [0.25,0.5]
params['l1_param'] = [1e-2,1e-3,1e-4,1e-5]
# params[''] = []

param_grid = ParameterGrid(params)

## Compiler parameters
K = 5
lr = 1e-4
eps = 70
n_cpu = multiprocessing.cpu_count()

## K-Fold Cross-validation
kf = KFold(n_splits=K)
w2v_embedding_layer = tf.keras.layers.Embedding(vocab_size, k,
                                                weights=[embedding_weigths],
                                                trainable=False
                                               )

CV_df = pd.DataFrame(columns=['params',
                              'accuracy_1',
                              'accuracy_2',
                              'accuracy_3',
                              'accuracy_4',
                              'accuracy_5',
                              'accuracy_mean',
                              'accuracy_SE'])

for p in param_grid:
    
    ## Assemble the model
    conv_reg = tf.keras.regularizers.L1(p['l1_param'])
    
    model = tf.keras.Sequential()
    model.add(w2v_embedding_layer)
    model.add(tf.keras.layers.Conv1D(p['n_filters'],
                                     p['window_size'],
                                     activation='relu',
                                     kernel_regularizer=conv_reg
                                    )
             )
    model.add(tf.keras.layers.GlobalMaxPool1D())
    model.add(tf.keras.layers.Dropout(p['dropout_prob']))
    model.add(tf.keras.layers.Dense(output_classes,activation='softmax'))
    
    init_weights = model.get_weights()
    
    ## Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    cv_accuracy = []
    
    ## Fit the model
    results = {}
    results['params'] = p
    
    for i, indices in enumerate(kf.split(X)):
        train_index, val_index = indices
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
    
        history = model.fit(x=X_train,
                            y=y_train,
                            epochs=eps,
                            verbose=1,
                            validation_data=(X_val,y_val),
                            workers=n_cpu
                           )
        
        cv_accuracy.append(history.history['val_accuracy'])
        results['accuracy_{}'.format(i+1)] = history.history['val_accuracy']
        
        model.set_weights(init_weights)
    
    ## Store the results
    results['accuracy_mean'] = np.mean(cv_accuracy,axis=0)
    results['accuracy_SE'] = np.std(cv_accuracy,axis=0)/np.sqrt(K)
    
    CV_df = CV_df.append(results,ignore_index=True)
    
CV_df.to_pickle('./cross_val_results.pkl')