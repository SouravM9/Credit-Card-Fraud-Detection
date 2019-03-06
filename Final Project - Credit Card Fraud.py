# Kernel SVM

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
import itertools
from matplotlib.colors import ListedColormap

# Importing the dataset
dataset = pd.read_csv('creditcard.csv')


#Standardize 'Amount' and 'Time' column
dataset['normAmount'] = StandardScaler().fit_transform(np.array(dataset['Amount']).reshape(-1,1))
#dataset['normTime'] = StandardScaler().fit_transform(np.array(dataset['Time']).reshape(-1,1))
dataset = dataset.drop(['Time','Amount'],axis=1)
print(dataset.head())


X = dataset.loc[:, dataset.columns != 'Class'].values
y = dataset.iloc[:, -2].values


# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)


# Fitting Kernel SVM to the Training set
classifier = SVC(kernel='poly', degree=2, gamma='auto', C=1, random_state = 0)
y_score = classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix for ", cm)
score = classifier.score(X_test,y_test)
print("Score for ", score)

classes = [0, 1]
cmap = plt.cm.Blues
plt.imshow(cm, interpolation='nearest', cmap=cmap)
plt.title("Confusion Matrix")
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=0)
plt.yticks(tick_marks, classes)
thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

# Compute ROC curve and ROC area for each class
y_score = classifier.fit(X_train, y_train.ravel()).decision_function(X_test)

fpr, tpr, thresholds = roc_curve(y_test.ravel(),y_score)
roc_auc = auc(fpr,tpr)

# Plot ROC
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b',label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.0])
plt.ylim([-0.1,1.01])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

'''

Discussion:

Score for  linear   0.9993469284570659

Score for  rbf   0.9993153282211173

Score for  poly   0.999336395045083

gamma = auto
degree =2
C = 1

Note: Including 'time' and standardizing it gives same result with same precision as not including it
'''
