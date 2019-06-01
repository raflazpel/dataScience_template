# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.metrics import log_loss
from sklearn.model_selection import cross_val_predict
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import sqlalchemy as db

# Load dataset
X, y = load_iris(return_X_y=True)

engine = db.create_engine('postgresql://admin:admin@sl-eu-de-1-portal.9.dblayer.com:17015/compose')

# Data preparation
df = pd.DataFrame(data=X,columns=['SL','SW','PL','PW'])
df['class'] = y

# Lets explore the most representative features

corrmat = df.corr()
top_corr_features = corrmat.index
g=sn.heatmap(df[top_corr_features].corr(),annot=True,cmap="RdYlGn")
plt.show()
# Plot the distribution
plt.subplot(221)
sn.scatterplot(X[:,0], y)
plt.title('Sepal length')


plt.subplot(222)
sn.scatterplot(X[:,1], y)
plt.title('Sepal width')


plt.subplot(223)
sn.scatterplot(X[:,2], y)
plt.title('Petal length')


plt.subplot(224)
sn.scatterplot(X[:,3], y)
plt.title('Petal width')
plt.show()

# Now we prepare the data with the most representative features:
X_train, X_test, y_train, y_test = train_test_split(df.loc[:,['SL','SW','PW']], df['class'], test_size=0.2, random_state=92)

'''
First Possibility Train-Test separation
'''
# Load and train the model
clf1 = LogisticRegression(solver='lbfgs', multi_class='multinomial').fit(df.loc[:,['SL','SW','PW']], df['class'])

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
'''
Second Possibility Cross Validation
'''
# Load and train the model

clf2 = LogisticRegression(solver='lbfgs', multi_class='multinomial')
# The function cross_val_score does not work in this case, because it used the dedault predict function
# so it calculates de log loss over the predicted classes and not over the predicted probabilities
# scores = cross_val_score(clf, X_train, y_train, cv=5,scoring='log_loss')

# TODO Study what we are actually performing in these lines if we want to use it for production
predicted_probab = cross_val_predict(clf2, X, y, cv = 3, method = 'predict_proba')

# Score accuracies
print('The log loss score is ' + str(log_loss(y, predicted_probab)))

'''
ONNX PACKAGE 


initial_type = [('float_input', FloatTensorType([1, 3]))]
onx = convert_sklearn(clf1, initial_types=initial_type)
with open("logreg_iris.onnx", "wb") as f:
    f.write(onx.SerializeToString())

'''