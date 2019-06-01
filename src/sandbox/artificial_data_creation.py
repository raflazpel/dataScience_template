x = np.random.choice(3, 500, p=[0.1, 0.3, 0.6])
y = np.random.choice(2, 500, p=[0.45, 0.55])
z = np.random.choice(3, 500, p=[0.1, 0.2,0.7])

x2 = np.random.choice(3, 500, p=[0.6, 0.3, 0.1])
y2 = np.random.choice(2, 500, p=[0.55, 0.45])
z2 = np.random.choice(3, 500, p=[0.7, 0.2,0.1])

x = np.concatenate([x,x2])
y = np.concatenate([y,y2])
z = np.concatenate([z,z2])
data = np.array([x, y,z])
print(data)


df = pd.DataFrame({'Pais':data[0,:],'Resultado':data[1,:],'Estudios':data[2,:]})
print(df)


# Now we prepare the data with the most representative features:
X_train, X_test, y_train, y_test = train_test_split(df.loc[:,['Pais','Estudios']], df['Resultado'], test_size=0.1, random_state=92)

'''
First Possibility Train-Test separation
'''
# Load and train the model
clf1 = LogisticRegression(solver='lbfgs', multi_class='ovr').fit(df.loc[:,['Pais','Estudios']], df['Resultado'])

# Score accuracies
print('Train accuracy is ' + str(clf1.score(X_train, y_train)))
print('Test accuracy is ' + str(clf1.score(X_test, y_test)))
print('The log loss score is ' + str(log_loss(y_test, clf1.predict_proba(X_test))))

# Calculate and print confusion matrix
y_true = y_test
y_pred = clf1.predict(X_test)
print(confusion_matrix(y_true, y_pred))