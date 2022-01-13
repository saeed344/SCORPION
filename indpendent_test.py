import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from PLS import PLS
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import pickle
import math
def cv(clf, X, y, nr_fold):
    ix = []
    for i in range(0, len(y)):
        ix.append(i)
    ix = np.array(ix)

    allACC = []
    allSENS = []
    allSPEC = []
    allMCC = []
    allAUC = []
    for j in range(0, nr_fold):
        train_ix = ((ix % nr_fold) != j)
        test_ix = ((ix % nr_fold) == j)
        train_X, test_X = X[train_ix], X[test_ix]
        train_y, test_y = y[train_ix], y[test_ix]
        clf.fit(train_X, train_y)
        p = clf.predict(test_X)
        pr = clf.predict_proba(test_X)[:,1]
        TP=0
        FP=0
        TN=0
        FN=0
        for i in range(0,len(test_y)):
            if test_y[i]==0 and p[i]==0:
                TP+= 1
            elif test_y[i]==0 and p[i]==1:
                FN+= 1
            elif test_y[i]==1 and p[i]==0:
                FP+= 1
            elif test_y[i]==1 and p[i]==1:
                TN+= 1
        ACC = (TP+TN)/(TP+FP+TN+FN)
        SENS = TP/(TP+FN)
        SPEC = TN/(TN+FP)
        det = math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
        if (det == 0):
            MCC = 0
        else:
            MCC = ((TP*TN)-(FP*FN))/det
        AUC = roc_auc_score(test_y,pr)
        allACC.append(ACC)
        allSENS.append(SENS)
        allSPEC.append(SPEC)
        allMCC.append(MCC)
        allAUC.append(AUC)
    return np.mean(allACC),np.mean(allSENS),np.mean(allSPEC),np.mean(allMCC),np.mean(allAUC)

def test(clf, X, y, Xt, yt):
    train_X, test_X = X, Xt
    train_y, test_y = y, yt
    clf.fit(train_X, train_y)
    p = clf.predict(test_X)
    pr = clf.predict_proba(test_X)[:,1]
    TP=0
    FP=0
    TN=0
    FN=0
    for i in range(0,len(test_y)):
        if test_y[i]==0 and p[i]==0:
            TP+= 1
        elif test_y[i]==0 and p[i]==1:
            FN+= 1
        elif test_y[i]==1 and p[i]==0:
            FP+= 1
        elif test_y[i]==1 and p[i]==1:
            TN+= 1
    ACC = (TP+TN)/(TP+FP+TN+FN)
    SENS = TP/(TP+FN)
    SPEC = TN/(TN+FP)
    det = math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
    if (det == 0):
        MCC = 0
    else:
        MCC = ((TP*TN)-(FP*FN))/det
    AUC = roc_auc_score(test_y,pr)

    return ACC, SENS, SPEC, MCC, AUC

TrainData=pd.read_csv('PF_new.csv')
data=np.array(TrainData)
X1=data[:,1:]
TestData=pd.read_csv('PF_test_new.csv')
data_=np.array(TestData)
Xt1=data_[:,1:]
Y1 = np.ones((250, 1))  # Value can be changed
Y2 = np.zeros((250, 1))
y = np.append(Y1, Y2)
Yt1 = np.ones((63, 1))  # Value can be changed
Yt2 = np.zeros((63, 1))
yt = np.append(Yt1, Yt2)
xgb_model=xgb.XGBClassifier()
xgbresult1=xgb_model.fit(X1,y.ravel())
feature_importance=xgbresult1.feature_importances_
feature_number=-feature_importance
H1=np.argsort(feature_number)
mask=H1[:130]
train_data=X1[:,mask]
test_tata=Xt1[:,mask]

selected_mask=[0,1,3,4,5,8,9,10,11,12,13,14,15,18,20,27,28,30,33,34,35,36,38,40,41,42,46,48,49,50,51,52,55,56,57,63,69,72,74,77,79,80,81,83,87,90,95,103,106,119]
Train_feat = train_data[:,selected_mask]
X=Train_feat
Test_feat = test_tata[:,selected_mask]
Xt=Test_feat

allclf = []
file = open("foldcrossvalidationresults.csv", "w")

# RF
param = [50, 100, 200, 500]
acc = np.zeros(len(param))
sens = np.zeros(len(param))
spec = np.zeros(len(param))
mcc = np.zeros(len(param))
roc = np.zeros(len(param))
for i in range(0, len(param)):
    clf = RandomForestClassifier(n_estimators=param[i], random_state=0)
    acc[i], sens[i], spec[i], mcc[i], roc[i] = cv(clf, X, y, 10)
choose = np.argmax(acc)
rf=RandomForestClassifier(n_estimators=param[choose], random_state=0).fit(X, y)
allclf.append(RandomForestClassifier(n_estimators=param[choose], random_state=0).fit(X, y))
file.write(
    "RF," + str(acc[choose]) + "," + str(sens[choose]) + "," + str(spec[choose]) + "," + str(mcc[choose]) + "," + str(
        roc[choose]) + "," + str(param[choose]) + "\n")

file.close()

########## Test ############################
file = open("independent_test.csv", "w")
for i in range(0, len(allclf)):
    acc, sens, spec, mcc, roc = test(allclf[i], X, y, Xt, yt)
    file.write(str(acc) + "," + str(sens) + "," + str(spec) + "," + str(mcc) + "," + str(roc) + "\n")
file.close()
