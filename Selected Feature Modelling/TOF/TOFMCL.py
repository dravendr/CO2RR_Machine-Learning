#encoding:utf-8
###########import packages##########
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn import svm
from sklearn import neighbors
from sklearn import tree
from sklearn.linear_model import BayesianRidge
from sklearn.tree import ExtraTreeRegressor
from sklearn.tree import ExtraTreeClassifier
from sklearn.preprocessing import label_binarize
from sklearn import linear_model
import lightgbm
import catboost
import xgboost
from catboost import *
from sklearn.metrics import roc_curve, auc
from itertools import cycle
from scipy import interp
###########loading data##########
fdata=pd.read_csv('database_filled_TOF_MCL.csv',encoding="gbk")
raw_data=fdata.loc[:,[
                      'Ionization Potential',#0
                      'Electronegativity',#1
                      'Number of d electrons',#2
                      'Graphene/Carbon Nanosheets or other 2D Structures',#3
                      'Carbon Nanofiber/Nanotubes',#4
                      'Biomass or other Organic Derived',#5  
                      'Main Transition Metal Content (wt. %)',#6
                      'Nitrogen Cotent (wt. %)',#7
                      'Metal-N Coordination Number (XAS)',#8    
                      'Pyridinic N Ratio',#9
                      'Pyrrolic N Ratio',#10
                      'Raman ID/IG Ratio',#11
                      'BET Surface Area (m2/g)',#12
                      'Pyrolysis Temperature (°C)',#13
                      'Pyrolysis Time (h)',#14
                      'Rising Rate (°C min-1)',#15
                      'Flow Cell/H-type Cell',#16
                      'Electrolyte Concentration (M)',#17
                      'Catalyst Loading (mg cm-2)',#18
                      'Carbon Paper/Glassy Carbon',#19
                      'Electrolyte pH',#20
                      'TOF_MCL_2_4',#21
                        ]]
###########train test splitting##########
raw_param=raw_data.iloc[:,0:21]
raw_power=raw_data.iloc[:,21]
print('ready')
nb_classes=3
from catboost import *
def gridsearch(model,param,algorithm_name):
    print('start')
    grid = GridSearchCV(model,param_grid=param,cv=5,n_jobs=-1)
    grid.fit(X_train,y_train)
    print('Best Classifier:',grid.best_params_,'Best Score:', grid.best_score_)
    best_model=grid.best_estimator_
    prediction_train=best_model.predict(X_train)
    prediction_test=best_model.predict(X_test)
    final_result=classification_report(y_test,prediction_test,output_dict=True)
    ##############################################################
    Y_test_labeled = label_binarize(y_test, classes=[i for i in range(nb_classes)])
    y_score=best_model.predict_proba(X_test)
#     y_score=y_score[:,1]
#     print(y_score)
    auc_curve(Y_test_labeled,y_score,algorithm_name)
    ##############################################################
    print(classification_report(y_train,prediction_train))
    print(classification_report(y_test,prediction_test))
    print(final_result['accuracy'])
    ###########generating a figure##########
    print(algorithm_name)
    print(best_model.feature_importances_)
def auc_curve(y_label, y_pre,algorithm_name):
#     y_label = y_label + 1
#     y_pre = y_pre + 1
    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(nb_classes):
        fpr[i], tpr[i], _ = roc_curve(y_label[:, i], y_pre[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_label.ravel(), y_pre.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])


    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(nb_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(nb_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= nb_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])



    # Plot all ROC curves
    lw = 2
    fig=plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(nb_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve (Multi-Class Classification) of %s' %algorithm_name)
    plt.legend(loc="lower right")
    plt.savefig('ROC Curve of %s TOFMCL 3D.png' %algorithm_name)
seed=1493
X_train, X_test, y_train, y_test = train_test_split(raw_param, raw_power, test_size=.1,random_state=seed)
##########CatBoost gridsearch CV for best hyperparameter##########
model_CatClassifier=catboost.CatBoostClassifier(random_state=1,verbose=0,bootstrap_type='Bernoulli')
param_cat = {
 'learning_rate':[0.001,0.0025,0.005,0.0075,0.01,0.025,0.05,0.075,0.1,0.25,0.5,0.75,1],
 'n_estimators':[50,100,200,400],
 "boosting_type":["Plain"],
 'max_depth':[5,7,9,11],
 'subsample':[0.4,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1],
    'reg_lambda':[0,0.001,0.01,0.0001,0.00001]
}
gridsearch(model_CatClassifier,param_cat,'CatBoost')

model_LightGBMClassifier=lightgbm.LGBMClassifier(random_state=1,verbose=-1)
param_light = {
  'boosting_type':['gbdt','rf'],
    'learning_rate':[0.001,0.0025,0.005,0.0075,0.01,0.025,0.05,0.075,0.1,0.25,0.5,0.75,1],
    'subsample':[0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1],
     'n_estimators':[50,100,200,400],
    'max_depth':[5,7,9,11,-1],
 'reg_alpha':[0,0.001,0.01,0.0001,0.00001],
 'reg_lambda':[0,0.001,0.01,0.0001,0.00001]
}
gridsearch(model_LightGBMClassifier,param_light,'LightGBM')

##########XGBoost gridsearch CV for best hyperparameter##########
model_XGBClassifier=xgboost.XGBClassifier(objective ='reg:squarederror',random_state=1,verbosity=0)
param_xg = {
 'booster':['gbtree'],
  'learning_rate':[0.001,0.0025,0.005,0.0075,0.01,0.025,0.05,0.075,0.1,0.25,0.5,0.75,1],
 'n_estimators':[50,100,200,400],
 'max_depth':[5,7,9,11,16],
 'subsample':[0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1],
 'reg_alpha':[0,0.001,0.01,0.0001,0.00001],
 'reg_lambda':[0,0.001,0.01,0.0001,0.00001]
}
gridsearch(model_XGBClassifier,param_xg,'XGBoost')
###########GradientBoost gridsearch CV for best hyperparameter##########
model_GradientBoostingClassifier = ensemble.GradientBoostingClassifier(random_state=1)
###########defining the parameters dictionary##########
param_GB = {
 'learning_rate':[0.001,0.0025,0.005,0.0075,0.01,0.025,0.05,0.075,0.1,0.25,0.5,0.75,1],
  'n_estimators':[50,100,200,400],
 'max_depth':[3,5,7,9,11,13,16],
 'criterion':['friedman_mse','mae','mse'],
 'max_features':['auto','sqrt','log2'],
 'loss':['deviance', 'exponential']
}
gridsearch(model_GradientBoostingClassifier,param_GB,'GradientBoost')

###########RandomForest gridsearch CV for best hyperparameter##########
model_RandomForestClassifier = ensemble.RandomForestClassifier(random_state=1)
###########defining the parameters dictionary##########
param_RF = {
      'n_estimators':[10,50,100,200,400],
     'max_depth':[3,5,7,9,11,None],
     'criterion':['gini','entropy'],
     'max_features':['auto','sqrt','log2']
}
gridsearch(model_RandomForestClassifier,param_RF,'Random Forest')


###########Extra Tree gridsearch CV for best hyperparameter##########
model_ExtraTreeClassifier = ExtraTreeClassifier(random_state=1)
param_ET = {
        'max_depth':[5,6,7,8,9,10,11,None],
        'criterion' : ['gini','entropy'],
        'splitter' : [ "best",'random'],
        'max_features':['auto','sqrt','log2']
}
gridsearch(model_ExtraTreeClassifier,param_ET,'Extra Tree')


###########Decision Tree gridsearch CV for best hyperparameter##########
model_DecisionTreeClassifier = tree.DecisionTreeClassifier(random_state=1)
param_DT = {
        'max_depth':[5,6,7,8,9,10,11,None],
        'criterion' : ['gini','entropy'],
        'splitter' : [ "best",'random'],
        'max_features':['auto','sqrt','log2']
}
gridsearch(model_DecisionTreeClassifier,param_DT,'Decision Tree')


###########AdaBoost gridsearch CV for best hyperparameter##########
model_AdaBoostClassifier = ensemble.AdaBoostClassifier(random_state=1)
param_Ada = {
      'learning_rate':[0.001,0.0025,0.005,0.0075,0.01,0.025,0.05,0.075,0.1,0.25,0.5,0.75,1],
    'n_estimators':[50,100,200,400]
}
gridsearch(model_AdaBoostClassifier,param_Ada,'AdaBoost')
