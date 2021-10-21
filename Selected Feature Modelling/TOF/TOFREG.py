#encoding:utf-8
###########import packages##########
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import xgboost 
import lightgbm
import catboost
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from xgboost import plot_importance
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn import ensemble
from sklearn.tree import ExtraTreeRegressor
from sklearn import svm
from sklearn import neighbors
from sklearn import tree
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from catboost import EFstrType, Pool
###########wrapping root mean square error for later calls##########
def compute_mae_mse_rmse(target,prediction):
    error = []
    for i in range(len(target)):
        error.append(target[i] - prediction[i])
    squaredError = []
    absError = []
    for val in error:
        squaredError.append(val * val)  # target-prediction之差平方
        absError.append(abs(val))  # 误差绝对值
    mae=sum(absError)/len(absError)  # 平均绝对误差MAE
    mse=sum(squaredError)/len(squaredError)  # 均方误差MSE
    RMSE=np.sqrt(sum(squaredError)/len(squaredError))
    R2=r2_score(target,prediction)
    return mae,mse,RMSE,R2
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
                      'TOF@Maximum FE(h-1) (log)'#21
                        ]]



###########defining a wrapper function for later call from each machine learning algorithms##########
raw_input=raw_data.iloc[:,0:21]
raw_output=raw_data.iloc[:,21]
# raw_input_global=raw_data.iloc[:,0:32]
# raw_output_global=raw_data.iloc[:,32]
X=raw_input.values.astype(np.float32)
y=raw_output.values.astype(np.float32)
###########wrap up fuction for later call for OPTIMIZATION##########
def evaluate(pre_2,real_2):
    pre_2=np.array(pre_2)
    real_2=np.array(real_2)
    pre_2_series=pd.Series(pre_2)
    real_2_series=pd.Series(real_2)
    return rmse(pre_2,real_2), round(pre_2_series.corr(real_2_series), 3)
def compare(list_name,limit):
    judge=1
    for a in list_name:
        if a < limit:
            judge=judge*1
        else:
            judge=judge*0
    return judge
def generate_arrays_from_file(path):
    while True:
        with open(path) as f:
            for line in f:
                # create numpy arrays of input data
                # and labels, from each line in the file
                x1, x2, y = process_line(line)
                yield ({'input_1': x1, 'input_2': x2}, {'output': y})

def gridsearch(model,param,algorithm_name):
    grid = GridSearchCV(model,param_grid=param,cv=5,n_jobs=-1)
    grid.fit(X_train,y_train)
    best_model=grid.best_estimator_
    result = best_model.predict(X_test)
    x_prediction_07=result
    y_real_07=y_test.values
    x_prediction_07_series=pd.Series(x_prediction_07)
    y_real_07_series=pd.Series(y_real_07)
    
    result_train = best_model.predict(X_train)
    x_prediction_07_train=result_train
    y_real_07_train=y_train.values
    x_prediction_07_series_train=pd.Series(x_prediction_07_train)
    y_real_07_series_train=pd.Series(y_real_07_train)
    
    ###########evaluating the regression quality##########
    corr_ann = round(x_prediction_07_series.corr(y_real_07_series), 5)
    error_val= compute_mae_mse_rmse(x_prediction_07,y_real_07)
    
    corr_ann_train = round(x_prediction_07_series_train.corr(y_real_07_series_train), 5)
    error_val_train= compute_mae_mse_rmse(x_prediction_07_train,y_real_07_train)
    
    print(algorithm_name)
    print(best_model.feature_importances_)
    print('Best Regressor:',grid.best_params_,'Best Score:', grid.best_score_)
    print(error_val,'TEST R2',error_val[3],'TEST CORR',corr_ann)
    print(error_val_train,'TRAIN R2',error_val_train[3],'TRAIN CORR',corr_ann_train)
    x_y_x=np.arange(0,5,0.1)
    x_y_y=np.arange(0,5,0.1)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x_prediction_07,y_real_07,color='red',label=algorithm_name+' Test Set',alpha=0.75)
    ax.scatter(x_prediction_07_train,y_real_07_train,color='blue',label=algorithm_name+' Training Set',alpha=0.25,marker="^")
    ax.plot(x_y_x,x_y_y)
    plt.legend()
    plt.xlabel(u"Predicted_Log(TOF(h-1))@_Maximum_Faradaic_Efficiency")
    plt.ylabel(u"Real_Log(TOF(h-1))@_Maximum_Faradaic_Efficiency")
    print('finished')
seed=9
X_train, X_test, y_train, y_test = train_test_split(raw_input, raw_output, test_size=.1,random_state=seed)
##########CatBoost gridsearch CV for best hyperparameter##########
model_CatRegressor=catboost.CatBoostRegressor(random_state=1,verbose=0)
param_cat = {
 'learning_rate':[0.001,0.0025,0.005,0.0075,0.01,0.025,0.05,0.075,0.1,0.25,0.5,0.75,1],
 'n_estimators':[50,100,200,400],
 'max_depth':[5,7,9,11],
 "boosting_type":["Plain"],
 'subsample':[0.4,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1],
    'reg_lambda':[0,0.001,0.01,0.0001,0.00001]
}
gridsearch(model_CatRegressor,param_cat,'CatBoost')
##########LGBM gridsearch CV for best hyperparameter##########
model_LightGBMRegressor=lightgbm.LGBMRegressor(random_state=1,verbose=-1)
param_light = {
 'boosting_type':['gbdt','rf'],
    'learning_rate':[0.001,0.0025,0.005,0.0075,0.01,0.025,0.05,0.075,0.1,0.25,0.5,0.75,1],
    'subsample':[0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1],
     'n_estimators':[50,100,200,400],
    'max_depth':[5,7,9,11,-1],
 'reg_alpha':[0,0.001,0.01,0.0001,0.00001],
 'reg_lambda':[0,0.001,0.01,0.0001,0.00001]
}
gridsearch(model_LightGBMRegressor,param_light,'LightGBM')

#########XGBoost gridsearch CV for best hyperparameter##########
model_XGBRegressor=xgboost.XGBRegressor(objective='reg:squarederror',random_state=1,verbosity=0)
param_xg = {
 'booster':['gbtree'],
  'learning_rate':[0.001,0.0025,0.005,0.0075,0.01,0.025,0.05,0.075,0.1,0.25,0.5,0.75,1],
 'n_estimators':[50,100,200,400],
 'max_depth':[5,7,9,11],
 'subsample':[0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1],
 'reg_alpha':[0,0.001,0.01,0.0001,0.00001],
 'reg_lambda':[0,0.001,0.01,0.0001,0.00001]
}
gridsearch(model_XGBRegressor,param_xg,'XGBoost')

###########GradientBoost gridsearch CV for best hyperparameter##########
model_GradientBoostingRegressor = ensemble.GradientBoostingRegressor(random_state=1)
###########defining the parameters dictionary##########
param_GB = {
 'learning_rate':[0.001,0.0025,0.005,0.0075,0.01,0.025,0.05,0.075,0.1,0.25,0.5,0.75,1],
                'criterion':['friedman_mse','mae','mse'],
                 'max_features':['auto','sqrt','log2'],
                 'loss':['ls', 'lad', 'huber', 'quantile'],
    'max_depth':[3,5,7,9,11,13,16]
}
gridsearch(model_GradientBoostingRegressor,param_GB,'GradientBoost')

###########RandomForest gridsearch CV for best hyperparameter##########
model_RandomForestRegressor = ensemble.RandomForestRegressor(random_state=1)
###########defining the parameters dictionary##########
param_RF = {
            'criterion':['mse','mae'],
            'max_features':['auto','sqrt','log2'],
    'n_estimators':[10,50,100,200,400],
     'max_depth':[3,5,7,9,11,None]
}
gridsearch(model_RandomForestRegressor,param_RF,'Random Forest')


###########Extra Tree gridsearch CV for best hyperparameter##########
model_ExtraTreeRegressor = ExtraTreeRegressor(random_state=1)
param_ET = {
        'max_depth':[5,6,7,8,9,10,11,None],
               'max_features':['auto','sqrt','log2'],
               'criterion' : ["mse", "friedman_mse", "mae"],
               'splitter' : [ "best",'random']
}
gridsearch(model_ExtraTreeRegressor,param_ET,'Extra Tree')


###########Decision Tree gridsearch CV for best hyperparameter##########
model_DecisionTreeRegressor = tree.DecisionTreeRegressor(random_state=1)
param_DT = {
        'max_depth':[5,6,7,8,9,10,11,None],
               'max_features':['auto','sqrt','log2'],
               'criterion' : ["mse", "friedman_mse", "mae"],
               'splitter' : [ "best",'random']
}
gridsearch(model_DecisionTreeRegressor,param_DT,'Decision Tree')


###########AdaBoost gridsearch CV for best hyperparameter##########
model_AdaBoostRegressor = ensemble.AdaBoostRegressor(random_state=1)
param_Ada = {
     'learning_rate':[0.001,0.0025,0.005,0.0075,0.01,0.025,0.05,0.075,0.1,0.25,0.5,0.75,1],
    'n_estimators':[50,100,200],
            'loss':['linear', 'square', 'exponential']
}
gridsearch(model_AdaBoostRegressor,param_Ada,'AdaBoost')
