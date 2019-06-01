''' This class is suppossed to implement the best possible DCNN to solve the regression problem'''

# IMPORT  MODULES
import itertools
import matplotlib.pyplot as plt
import numpy as np
import onnxmltools
import pandas as pd
from keras.layers import Dense
from keras.models import Sequential
from tensorflow import set_random_seed
from numpy.random import seed
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing


# SET SEED
seed(92)
set_random_seed(92)

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
df = pd.read_pickle('..\\..\\data\\processed\\regressor_to_matricula_SMART.pkl')

# The classes are unbalanced, so we have to balance them to extract info of the correlation
exito = df[df['Result']]
fracaso = df[df['Result'] == False]
fracaso = fracaso.iloc[:int(exito.shape[0] * 1.6), :]
df = pd.concat([exito, fracaso], ignore_index=True)

# Feature selection and preprocessing for this model
labels = ['idcurso', 'tipo', 'ciudad', 'rating', 'education_level_c', 'days_since_modification']
features = df[labels].copy()
features.to_pickle('..\\..\\data\\processed\\df_for_inference.pkl')
print(features.shape)
for label in labels:
    if label !='days_since_modification':
        one_hot = pd.get_dummies(df[label], prefix=label)
        # Drop column B as it is now encoded
        features = features.drop(label, axis=1)
        # Join the encoded df
        features = features.join(one_hot)
print(features.shape)
# Add non categorical data
#features['days_since_modification'] = preprocessing.normalize(df[['days_since_modification']])


# Separate the train and test sets
X_train, X_test, y_train, y_test = train_test_split(features.loc[:, :], df['Result'], test_size=0.2, random_state=92)


def build_nn(optimizer = 'adam'):

    net = Sequential()
    net.add(Dense(30, activation='relu', input_shape=(X_train.shape[1],)))
    net.add(Dense(25, activation='relu'))
    net.add(Dense(10, activation='relu'))
    net.add(Dense(1, activation='sigmoid', kernel_initializer='random_normal'))

    # Print a summary of the network. It has 7960 parameters.
    print(net.summary())

    # Definition of the settings
    net.compile(optimizer = optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return net

def train_nn(net, x, y):
    # It is going to use a minibatch strategy with a batch size of 50 samples
    # and will run for 10 epochs
    '''Best: -0.512555
     using
     {'batch_size': 1, 'epochs': 32, 'optimizer': 'Adadelta'}
     '''
    batch_size = 1
    epochs = 20

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
    print(net.predict_proba(x))
    print('Test loss:', test_loss)
    print('Test accuracy:', test_acc)
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
model = train_nn(model,X_train, y_train)
'''
# Create model

model = KerasClassifier(build_fn=build_nn, verbose=0)

# define the grid search parameters
batch_size = [1,2]
epochs = [12,13,14,15,16]
optimizer = ['SGD','adam']
param_grid = dict(batch_size=batch_size, epochs=epochs, optimizer=optimizer)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=5,scoring='neg_log_loss')
grid_result = grid.fit(X_train, y_train)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
'''
evaluate_nn(model,X_train, y_train)
# Compute the confusion matrix
y_pred = model.predict_classes(X_test)
confusion_mtx = confusion_matrix(y_test, y_pred)
# plot the confusion matrix
plot_confusion_matrix(confusion_mtx, classes=range(2))



# Save model to onnx
onnx_model = onnxmltools.convert_keras(model)
onnxmltools.utils.save_model(onnx_model, '..\\..\\model\\conversion_probability.onnx')
