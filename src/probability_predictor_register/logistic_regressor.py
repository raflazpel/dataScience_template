from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import sgd
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import itertools
import numpy as np
from sklearn.metrics import confusion_matrix, make_scorer
from sklearn.model_selection import GridSearchCV
from numpy.random import seed
from sklearn import preprocessing

seed(92)

# Best result:
'''
Train accuracy is 0.8112449799196787
Test accuracy is 0.7777777777777778
The log loss score is 0.5195078521342978
[[29 11]
 [ 3 20]]
 
labels = ['education_level_c', 'idcurso', 'tipo', 'rating','ciudad'] days_since_modification
 
'''

# Load dataframe
df = pd.read_pickle('..\\..\\data\\processed\\regressor_to_matricula_SMART.pkl')

# The classes are unbalanced, so we have to balance them to extract info of the correlation

exito = df[df['Result']==True]
fracaso = df[df['Result']==False]

fracaso = fracaso.iloc[:int(exito.shape[0]*1.6), :]
df = pd.concat([exito, fracaso], ignore_index=True)
print(df.shape)

labels = ['education_level_c', 'idcurso', 'tipo', 'rating','ciudad']


features = df[labels].copy()
for label in labels:
    one_hot = pd.get_dummies(features[label],prefix=label)
    # Drop column B as it is now encoded
    features = features.drop(label, axis=1)
    # Join the encoded df
    features = features.join(one_hot)


# Add non categorical data
# features['days_since_modification'] = preprocessing.normalize(df[['days_since_modification']])
features['days_since_modification'] = df['days_since_modification']
# Set the seed and split data
np.random.seed(92)
X_train, X_test, y_train, y_test = train_test_split(features.loc[:,:], df['Result'], test_size=0.2, random_state=92)




def test_logistic(X_train, X_test, y_train, y_test):
    # Load and train the model
    clf1 = LogisticRegression(solver='liblinear', multi_class='ovr').fit(X_train, y_train)

    # Score accuracies
    print('Train accuracy is ' + str(clf1.score(X_train, y_train)))
    print('Test accuracy is ' + str(clf1.score(X_test, y_test)))
    print('The log loss score is ' + str(log_loss(y_test, clf1.predict_proba(X_test))))

    # Calculate and print confusion matrix
    y_true = y_test
    y_pred = clf1.predict(X_test)
    print(confusion_matrix(y_true, y_pred))

    # Calculate probabilities for each class
    print('Probabilities for each value are' + str(clf1.predict_proba(X_test)))
    return clf1

def logistic_gridsearch(X_train,X_test, y_train, y_test):

    grid = {"C": np.linspace(0.1, 40, 50), "penalty": ["l1", "l2"]}  # l1 lasso l2 ridge
    logreg = LogisticRegression(solver='liblinear', multi_class='ovr',class_weight='balanced')
    scorer = make_scorer(log_loss, greater_is_better=True, needs_proba=True)
    logreg_cv = GridSearchCV(logreg, grid, cv=5,scoring='neg_log_loss')
    logreg_cv.fit(X_train, y_train)
    print("tuned hpyerparameters :(best parameters) ", logreg_cv.best_params_)
    print("log_loss :", logreg_cv.best_score_)
    print(confusion_matrix(y_test, logreg_cv.predict(X_test)))

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
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Actual class')
    plt.xlabel('Predicted class')



# Predict the values from the validation dataset
model = test_logistic(X_train, X_test, y_train, y_test)
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5)
# Convert predictions classes to one hot vectors

# Convert validation observations to one hot vectors

# compute the confusion matrix
confusion_mtx = confusion_matrix(y_test, y_pred)
# plot the confusion matrix
plot_confusion_matrix(confusion_mtx, classes = range(2))



logistic_gridsearch(X_train, X_test, y_train, y_test)