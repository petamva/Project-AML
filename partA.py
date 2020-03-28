

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_curve, auc
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier

data = pd.read_csv('arrhythmia.data',header=None)

data.shape

data.info()

pd.isna(data).sum().sum()

data1 = data.apply(pd.to_numeric, errors='coerce')

pd.isna(data1).sum().sum()

pd.isna(data1[13]).sum()

del data[13]

#data=data.replace('?',np.NaN)

classes = data1[279]
attr = data1.iloc[:,:279]

classes.replace(1,0,inplace=True)
classes.replace(list(range(2,17)),1,inplace=True)

attr.columns = list(range(attr.shape[1]))

for i in range(attr.shape[1]):
    attr[i].fillna(attr[i].median(),inplace=True)

attr.info()

pca = PCA(n_components=30, svd_solver='full')

attrReduced=pca.fit(attr).transform(attr)

print('Components kept: 30\nExplained variance=',pca.explained_variance_ratio_.sum())

# outlier detection/visualization

ax = plt.figure(figsize=(12, 5)).gca(title='Attribute Distribution', 
                                     xlabel='Values', ylabel='Attributes')
flierprops = dict(markerfacecolor='0.75', markersize=5, linestyle='none')
whiskerprops = capprops = dict(c='white')
sns.boxplot(data=attrReduced, orient='horizontal', 
    flierprops=flierprops, whiskerprops=whiskerprops, capprops=capprops);      

threshold = 2.5

for i in range(attrReduced.shape[1]):
    zscore = stats.zscore(attrReduced[:,i])
    median = np.nanmedian(attrReduced[:,i])
    attrReduced[:,i] = np.where(np.abs(zscore) < threshold,attrReduced[:,i],median)

ax = plt.figure(figsize=(12, 5)).gca(title='Attribute Distribution', 
                                     xlabel='Values', ylabel='Attributes')
flierprops = dict(markerfacecolor='0.75', markersize=5, linestyle='none')
whiskerprops = capprops = dict(c='white')
sns.boxplot(data=attrReduced, orient='horizontal', 
    flierprops=flierprops, whiskerprops=whiskerprops, capprops=capprops); 
            
# split the data sets into training and tests
x_train, x_test, y_train, y_test = train_test_split (attrReduced,classes, test_size=0.50, random_state=1)


clfANN = MLPClassifier(solver='adam', activation='relu',
                    batch_size=1, tol=1e-19,
                     hidden_layer_sizes=(5), random_state=1, max_iter=100000, verbose=False)

clfANN2 = MLPClassifier(solver='adam', activation='relu',
                       batch_size=1, tol=1e-19,
                       hidden_layer_sizes=(3), random_state=1,
                       max_iter=100000, verbose=False)

#Define decision Tree
clfDT =  tree.DecisionTreeClassifier(max_depth=5,min_samples_split=2,min_samples_leaf=1)
clfDT2 =  tree.DecisionTreeClassifier(max_depth=10,min_samples_split=2,min_samples_leaf=1)

'''
#Define Support vector machine
#clfSVM= svm.SVC(C=100.0, cache_size=200, class_weight=None, coef0=0.0,
#    decision_function_shape='ovo', degree=3, gamma='auto', kernel='rbf',
#    max_iter=-1, probability=False, random_state=None, shrinking=True,
#    tol=0.001, verbose=False)
'''

#Define a Naive Bayes
clfNB = GaussianNB()

#Define a Nearest Neighbours classifiers 
clfNN = KNeighborsClassifier(n_neighbors=4,weights='distance',metric='minkowski')
# different parameters
clfNN2 = KNeighborsClassifier(n_neighbors=15,weights='uniform',metric='minkowski')



#train the classifiers
                         
clfDT.fit(x_train, y_train)
clfDT2.fit(x_train, y_train)
#clfSVM.fit(x_train, y_train)
clfANN.fit(x_train, y_train)
clfANN2.fit(x_train, y_train)
clfNB.fit(x_train, y_train)
clfNN.fit(x_train, y_train)
clfNN2.fit(x_train, y_train)


#MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto',
#       beta_1=0.9, beta_2=0.999, early_stopping=False,
#       epsilon=1e-08, hidden_layer_sizes=(5, 2), learning_rate='constant',
 #      learning_rate_init=0.001, max_iter=200, momentum=0.9,
  #     nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
   #    solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=False,
    #   warm_start=False)


#test the trained model on the test set
y_test_pred_ANN=clfANN.predict(x_test)
y_test_pred_ANN2=clfANN2.predict(x_test)
y_test_pred_DT=clfDT.predict(x_test)
y_test_pred_DT2=clfDT2.predict(x_test)
#y_test_pred_SVM=clfSVM.predict(x_test)
y_test_pred_NB=clfNB.predict(x_test)
y_test_pred_NN=clfNN.predict(x_test)
y_test_pred_NN2=clfNN2.predict(x_test)


confMatrixTestANN=confusion_matrix(y_test, y_test_pred_ANN, labels=None)
confMatrixTestANN2=confusion_matrix(y_test, y_test_pred_ANN2, labels=None)
confMatrixTestDT=confusion_matrix(y_test, y_test_pred_DT, labels=None)
confMatrixTestDT2=confusion_matrix(y_test, y_test_pred_DT2, labels=None)
#confMatrixTestSVM=confusion_matrix(y_test, y_test_pred_SVM, labels=None)
confMatrixTestNB=confusion_matrix(y_test, y_test_pred_NB, labels=None)
confMatrixTestNN=confusion_matrix(y_test, y_test_pred_NN, labels=None)
confMatrixTestNN2=confusion_matrix(y_test, y_test_pred_NN2, labels=None)


print('Test size: ',np.size(x_test,0))
print('Number of classes: ',len(classes.unique()))

print ('Conf matrix Neural Network')
print (confMatrixTestANN)
print ()

print ('Conf matrix Neural Net 2')
print (confMatrixTestANN2)
print ()

print ('Conf matrix Decision Tree')
print (confMatrixTestDT)
print ()

print ('Conf matrix Decision Tree 2')
print (confMatrixTestDT2)
print ()

#print ('Conf matrix Support Vector Classifier')
#print (confMatrixTestSVM)
#print ()

print ('Conf matrix Naive Bayes')
print (confMatrixTestNB)
print ()


print ('Conf matrix Nearest Neighbor')
print (confMatrixTestNN)
print ()

print ('Conf matrix Nearest Neighbor 2')
print (confMatrixTestNN2)
print ()

precisionDT=precision_recall_fscore_support(y_test, y_test_pred_DT, average='macro')[0]
recallDT=precision_recall_fscore_support(y_test, y_test_pred_DT, average='macro')[1]
f1DT=precision_recall_fscore_support(y_test, y_test_pred_DT, average='macro')[2]

precisionNB=precision_recall_fscore_support(y_test, y_test_pred_NB, average='macro')[0]
recallNB=precision_recall_fscore_support(y_test, y_test_pred_NB, average='macro')[1]
f1NB=precision_recall_fscore_support(y_test, y_test_pred_NB, average='macro')[2]

precisionNN=precision_recall_fscore_support(y_test, y_test_pred_NN, average='macro')[0]
recallNN=precision_recall_fscore_support(y_test, y_test_pred_NN, average='macro')[1]
f1NN=precision_recall_fscore_support(y_test, y_test_pred_NN, average='macro')[2]

precisionANN=precision_recall_fscore_support(y_test, y_test_pred_ANN, average='macro')[0]
recallANN=precision_recall_fscore_support(y_test, y_test_pred_ANN, average='macro')[1]
f1ANN=precision_recall_fscore_support(y_test, y_test_pred_ANN, average='macro')[2]

precisionNN2=precision_recall_fscore_support(y_test, y_test_pred_NN2, average='macro')[0]
recallNN2=precision_recall_fscore_support(y_test, y_test_pred_NN2, average='macro')[1]
f1NN2=precision_recall_fscore_support(y_test, y_test_pred_NN2, average='macro')[2]

precisionANN2=precision_recall_fscore_support(y_test, y_test_pred_ANN2, average='macro')[0]
recallANN2=precision_recall_fscore_support(y_test, y_test_pred_ANN2, average='macro')[1]
f1ANN2=precision_recall_fscore_support(y_test, y_test_pred_ANN2, average='macro')[2]

precisionDT2=precision_recall_fscore_support(y_test, y_test_pred_DT2, average='macro')[0]
recallDT2=precision_recall_fscore_support(y_test, y_test_pred_DT2, average='macro')[1]
f1DT2=precision_recall_fscore_support(y_test, y_test_pred_DT2, average='macro')[2]


print('\n\t\t|\t Macro precision\t|\tRecall \t\t|\tf1-score \t')
print('----------------------------------------------------------------------------------------------------')
print('Decision Tree:  |      ',precisionDT,'     | ',recallDT,'  |   ',f1DT)
print('Decision Tree 2:|      ',precisionDT2,'     | ',recallDT2,'  |   ',f1DT2)
print('Naive Bayes:    |      ',precisionNB,'      | ',recallNB,'  |   ',f1NB)
print('Nearest Neigh:  |      ',precisionNN,'     | ',recallNN,'  |   ',f1NN)
print('Nearest Neigh 2:|      ',precisionNN2,'     | ',recallNN2,'  |   ',f1NN2)
print('Neur Network:   |      ',precisionANN,'     | ',recallANN,'  |   ',f1ANN)
print('Neur Network 2: |      ',precisionANN2,'     | ',recallANN2,'  |   ',f1ANN2)

pr_y_test_pred_DT=clfDT.predict_proba(x_test)
#pr_y_test_pred_SVM=clfSVM.predict_proba(x_test)
pr_y_test_pred_NN=clfNN.predict_proba(x_test)
pr_y_test_pred_ANN2=clfANN2.predict_proba(x_test)
pr_y_test_pred_NB=clfNB.predict_proba(x_test)

#clfSVM.predict_proba

#ROC curve
fprDT, tprDT, thresholdsDT = roc_curve(y_test, pr_y_test_pred_DT[:,1],pos_label=None)
#fprSVM, tprSVM, thresholdsSVM = roc_curve(y_test, pr_y_test_pred_SVM[:,1])
fprNN, tprNN, thresholdsNN = roc_curve(y_test, pr_y_test_pred_NN[:,1],pos_label=None)
fprANN, tprANN, thresholdsANN2 = roc_curve(y_test, pr_y_test_pred_ANN2[:,1],pos_label=None)
fprNB, tprNB, thresholdsNB = roc_curve(y_test, pr_y_test_pred_NB[:,1],pos_label=None)

lw=2
plt.plot(fprDT,tprDT,color='blue',label='Decision Tree')
#plt.plot(fprSVM,tprSVM,color='red',label='Support Vector')
plt.plot(fprNN,tprNN,color='red',label='Nearest Neighbor')
plt.plot(fprANN,tprANN,color='green',label='ANN2')
plt.plot(fprNB,tprNB,color='black',label='Naive Bayes')
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.savefig('ROC Curve.png',dpi=400,bbox_inches='tight')
plt.show()


