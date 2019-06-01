''' This class is suppossed to implement the best possible DCNN to solve the regression problem'''

# IMPORT  MODULES
import os
os.environ['PYTHONHASHSEED'] = '0'
import numpy as np
np.random.seed(92)
import random
random.seed(92)
import tensorflow as tf
tf.set_random_seed(92)
from keras import backend as K
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.Session(graph = tf.get_default_graph(), config=session_conf)
K.set_session(sess)

import itertools
import matplotlib.pyplot as plt
import numpy as np
import onnxmltools
import pandas as pd
from keras.layers import Dense
from keras.models import Sequential
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn.metrics import log_loss

'''
BEST RESULT ACHIEVED

Test loss: 0.3866816285144852
Test accuracy: 0.8273092359902868
[[29 11]
 [ 3 20]]

features['days_since_modification'] = df['days_since_modification']
labels = ['idcurso', 'tipo', 'ciudad','rating','education_level_c']
 '''

# Load dataframe
df = pd.read_pickle('..\\..\\data\\processed\\regressor_to_matricula_SMART_complete.pkl')

# Separate target from features
target = df['Result'].copy()
df.drop(labels=['Result'], inplace=True, axis=1)

# Convert categorical to dummies
labels = df.columns
for label in labels:
    if label != 'days_since_modification':
        one_hot = pd.get_dummies(df[label], prefix=label)
        # Drop column B as it is now encoded
        df = df.drop(label, axis=1)
        # Join the encoded df
        df = df.join(one_hot)
print(df.shape)
# Delete non categorical data
#df.drop(labels=['days_since_modification'], axis = 1, inplace = True)

# Separate train, validation and test datasets.
df_train, df_test, target_train, target_test = train_test_split(df, target, test_size=0.1, random_state=92)

# Solo usar si hay que ajustar hiperparámetros y no se va a usar cross-validation
df_train, df_validation, target_train, target_validation = train_test_split(df_train, target_train, test_size=0.001, random_state=92)

print(df_train.shape)
print(df_validation.shape)
print(df_test.shape)

# The classes are unbalanced, so we have to balance them to extract info of the correlation. We are going to use
# over sampling for the EXITO case on the training dataset.

df_train['Result'] = target_train
exito = df_train[df_train['Result']]
fracaso = df_train[df_train['Result'] == False]
print('-----------------')
print(exito.shape)
print(fracaso.shape)
print('Oversampling')
exito = exito.sample(n = fracaso.shape[0], replace = True)
print(exito.shape)
print(fracaso.shape)
df_train = pd.concat([exito, fracaso], ignore_index=True)
df_train = df_train.sample(frac = 1)
target_train = df_train['Result'].copy()
df_train.drop(labels=['Result'], axis=1, inplace=True)

# Feature selection and preprocessing for this model
# labels = ['idcurso', 'tipo', 'ciudad', 'rating', 'education_level_c', 'days_since_modification']
# features = df[labels].copy()
# features.to_pickle('..\\..\\data\\processed\\df_for_inference.pkl')
# print(features.shape)

def build_nn(optimizer='adam'):
    net = Sequential()
    net.add(Dense(30, activation='relu', input_shape=(df_train.shape[1],)))
    net.add(Dense(5, activation='relu'))
    net.add(Dense(1, activation='sigmoid', kernel_initializer='random_normal'))

    # Print a summary of the network. It has 7960 parameters.
    print(net.summary())

    # Definition of the settings
    net.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return net


def train_nn(net, x, y):
    # It is going to use a minibatch strategy with a batch size of 50 samples
    # and will run for 10 epochs
    '''Best: -0.512555
     using
     {'batch_size': 1, 'epochs': 32, 'optimizer': 'Adadelta'}
     '''
    batch_size = 20
    epochs = 30

    y = y.astype('int')

    # Training of the network
    net.fit(x, y,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1
            )

    return net


def evaluate_nn(net, x, y):
    y = y.astype('int')
    test_loss, test_acc = net.evaluate(x, y)
    print('Test loss:', test_loss)
    print('Test accuracy:', test_acc)
    return test_loss

def evaluate_grid(grid, x, y):
    y = y.astype('int')
    test_loss = log_loss(y, grid.predict_proba(x))
    print('Test loss:', test_loss)
    return test_loss


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, ['Fracaso', 'Exito'], rotation=45)
    plt.yticks(tick_marks, ['Fracaso Real', 'Exito Real'])

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Clase real')
    plt.xlabel('Clasificado como')


# Create model
model = build_nn()
model = train_nn(model, df_train, target_train)

evaluate_nn(model, df_test, target_test)
# Compute the confusion matrix
y_pred = model.predict_classes(df_test)
confusion_mtx = confusion_matrix(target_test, y_pred)
# plot the confusion matrix
plot_confusion_matrix(confusion_mtx, classes=range(2))
plt.show()
'''
# Create model

model = KerasClassifier(build_fn=build_nn, verbose=0)

# define the grid search parameters
batch_size = [1]
epochs = [13,15,16]
optimizer = ['adam']
param_grid = dict(batch_size=batch_size, epochs=epochs, optimizer=optimizer)
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='neg_log_loss')
grid_result = grid.fit(df_train, target_train)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

evaluate_grid(grid, df_test, target_test)
# Compute the confusion matrix
y_pred = grid.predict(df_test)
confusion_mtx = confusion_matrix(target_test, y_pred)
# plot the confusion matrix
plot_confusion_matrix(confusion_mtx, classes=range(2))
plt.show()
'''

# Save model to onnx
onnx_model = onnxmltools.convert_keras(model)
onnxmltools.utils.save_model(onnx_model, '..\\..\\model\\conversion_probability.onnx')
