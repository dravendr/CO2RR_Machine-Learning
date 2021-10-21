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
fdata=pd.read_csv('database_filled.csv',encoding="gbk")
raw_data=fdata.loc[:,[
                     'Relative Atomic Mass',#0
                      'Atomic Number',#1
                      'Ionization Potential',#2
                      'Electronegativity',#3
                      'Number of d electrons',#4
                      'ZIF or MOF Derived',#5
                      'Hard or Soft Templated',#6
                      'Graphene/Carbon Nanosheets or other 2D Structures',#7
                      'Polymer Derived',#8
                      'Carbon Nanofiber/Nanotubes',#9
                      'Carbon Black Derived',#10
                      'Biomass or other Organic Derived',#11  
                      'Main Transition Metal Content (wt. %)',#12
                      'Nitrogen Cotent (wt. %)',#13
                      'Metal-N Coordination Number (XAS)',#14    
                      'Pyridinic N Ratio',#15
                      'Pyrrolic N Ratio',#16
                      'Graphitic N Ratio',#17
                      'Oxidized N Ratio',#18
                      'Raman ID/IG Ratio',#19
                      'BET Surface Area (m2/g)',#20
                      'Acid Wash',#21
                      'Pyrolysis Temperature (°C)',#22
                      'Pyrolysis Time (h)',#23
                      'Rising Rate (°C min-1)',#24
                      'Flow Cell/H-type Cell',#25
                      'Electrolyte Concentration (M)',#26
                      'Catalyst Loading (mg cm-2)',#27
                      'Nafion Membrane Thickness (μm)',#28
                      'Carbon Paper/Glassy Carbon',#29
                      'Electrode Area cm2',#30
                      'Electrolyte pH'#31
                        ]]
###########defining a wrapper function for later call from each machine learning algorithms##########
raw_input=raw_data.iloc[:,0:32]
# X=raw_input.values.astype(np.float32)
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
def Get_Average(list):
    sum = 0
    for item in list:     
        sum += item  
    return sum/len(list)
def qualified_count(list_result,standard):
    count = 0
    for item in list_result:
        if item > standard:
            count+=1
    return count
def avg_top_x(list_result,top_number):
    new_list=list_result[:]
    new_list.sort(reverse=True)
    return Get_Average(new_list[0:top_number])

CD_05=fdata.loc[:,['FE of Product (CO) at -0.5V (vs.RHE)']]
CD_06=fdata.loc[:,['FE of Product (CO) at -0.6V (vs.RHE)']]
CD_07=fdata.loc[:,['FE of Product (CO) at -0.7V (vs.RHE)']]
CD_08=fdata.loc[:,['FE of Product (CO) at -0.8V (vs.RHE)']]
CD_09=fdata.loc[:,['FE of Product (CO) at -0.9V (vs.RHE)']]
def gridsearch(model,param,algorithm_name,X_train,X_test,y_train,y_test):
    grid = GridSearchCV(model,param_grid=param,cv=5,n_jobs=-1)
    grid.fit(X_train,y_train)
    best_model=grid.best_estimator_
    result = best_model.predict(X_test)
    x_prediction_07=result
    y_real_07=y_test.values
    x_prediction_07_series=pd.Series(x_prediction_07)
    y_real_07_series=pd.Series(y_real_07[:,0])   
    result_train = best_model.predict(X_train)
    x_prediction_07_train=result_train
    y_real_07_train=y_train.values
    x_prediction_07_series_train=pd.Series(x_prediction_07_train)
    y_real_07_series_train=pd.Series(y_real_07_train[:,0])
    ###########evaluating the regression quality##########
    corr_ann = round(x_prediction_07_series.corr(y_real_07_series), 5)
    error_val= compute_mae_mse_rmse(x_prediction_07,y_real_07[:,0])    
    corr_ann_train = round(x_prediction_07_series_train.corr(y_real_07_series_train), 5)
    error_val_train= compute_mae_mse_rmse(x_prediction_07_train,y_real_07_train[:,0])
    return_list=[]
    return_list.append(x_prediction_07)
    return_list.append(y_real_07[:,0])
    return return_list
def compute_curves(prediction_list,real_list,algorithm_name):
    for i in range(0,len(prediction_list[0])):
        prediction_curve=[prediction_list[0][i],prediction_list[1][i],prediction_list[2][i],prediction_list[3][i],prediction_list[4][i]]
        real_curve=[real_list[0][i],real_list[1][i],real_list[2][i],real_list[3][i],real_list[4][i]]
        x_list=[0.5,0.6,0.7,0.8,0.9]
        prediction_curve_series=pd.Series(prediction_curve)
        real_curve_series=pd.Series(real_curve)
        corr_ann = round(prediction_curve_series.corr(real_curve_series), 5)
        error_val= compute_mae_mse_rmse(prediction_curve,real_curve)
        
        fig=plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(x_list,prediction_curve,label=algorithm_name+'Prediction',color='red')
        ax.plot(x_list,real_curve,label='Real Curve',color='blue')
        plt.legend()
        plt.xlabel(u"Potential V (vs. RHE)")
        plt.ylabel(u"Faradaic Efficiency (%)")
        plt.savefig('ML FE CURVE %s th %s opt.png' %(i,algorithm_name))
        print(algorithm_name)
        print(corr_ann, error_val)
def computing_different_algorithm(model,param,algorithm_name):
    MODEL05=gridsearch(model,param,algorithm_name,X_train, X_test, y_train_05, y_test_05)
    MODEL06=gridsearch(model,param,algorithm_name,X_train, X_test, y_train_06, y_test_06)
    MODEL07=gridsearch(model,param,algorithm_name,X_train, X_test, y_train_07, y_test_07)
    MODEL08=gridsearch(model,param,algorithm_name,X_train, X_test, y_train_08, y_test_08)
    MODEL09=gridsearch(model,param,algorithm_name,X_train, X_test, y_train_09, y_test_09)    
    prediction_list=[MODEL05[0],MODEL06[0],MODEL07[0],MODEL08[0],MODEL09[0]]
    real_list=[MODEL05[1],MODEL06[1],MODEL07[1],MODEL08[1],MODEL09[1]]
    compute_curves(prediction_list,real_list,algorithm_name)
    print('finished')

##########CatBoost gridsearch CV for best hyperparameter##########
model_CatRegressor=catboost.CatBoostRegressor(random_state=1,verbose=0)
param_cat = {
'learning_rate':[0.001,0.0025,0.005,0.0075,0.01,0.025,0.05,0.075,0.1,0.25,0.5],
'n_estimators':[50,100,200,400],
'max_depth':[5,7,9,11],
'subsample':[0.4,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1],
'reg_lambda':[0,0.001,0.01,0.0001,0.00001]
}

##########LGBM gridsearch CV for best hyperparameter##########
model_LightGBMRegressor=lightgbm.LGBMRegressor(random_state=1,verbose=-1)
param_light = {
'boosting_type':['gbdt','rf'],
'learning_rate':[0.001,0.0025,0.005,0.0075,0.01,0.025,0.05,0.075,0.1,0.25,0.5],
'subsample':[0.4,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1],
'n_estimators':[50,100,200,400,None],
'max_depth':[5,7,9,11,-1],
'reg_alpha':[0,0.001,0.01,0.0001,0.00001],
'reg_lambda':[0,0.001,0.01,0.0001,0.00001]
}

 #########XGBoost gridsearch CV for best hyperparameter##########
model_XGBRegressor=xgboost.XGBRegressor(objective='reg:squarederror',random_state=1,verbosity=0)
param_xg = {
'booster':['gbtree'],
'learning_rate':[0.001,0.0025,0.005,0.0075,0.01,0.025,0.05,0.075,0.1,0.25,0.5],
'n_estimators':[50,100,200,400,None],
'max_depth':[5,7,9,11,16],
'subsample':[0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1],
'reg_alpha':[0,0.001,0.01,0.0001,0.00001],
'reg_lambda':[0,0.001,0.01,0.0001,0.00001]
}

###########GradientBoost gridsearch CV for best hyperparameter##########
model_GradientBoostingRegressor = ensemble.GradientBoostingRegressor(random_state=1)
###########defining the parameters dictionary##########
param_GB = {
'learning_rate':[0.001,0.0025,0.005,0.0075,0.01,0.025,0.05,0.075,0.1,0.25,0.5],
'criterion':['friedman_mse','mae','mse'],
'max_features':['auto','sqrt','log2'],
'loss':['ls', 'lad', 'huber', 'quantile']
}

###########RandomForest gridsearch CV for best hyperparameter##########
model_RandomForestRegressor = ensemble.RandomForestRegressor(random_state=1)
###########defining the parameters dictionary##########
param_RF = {
'n_estimators':[10,50,100,200,400],
'max_depth':[3,5,7,9,11,None],
'criterion':['mse','mae'],
'max_features':['auto','sqrt','log2']
}

###########Extra Tree gridsearch CV for best hyperparameter##########
model_ExtraTreeRegressor = ExtraTreeRegressor(random_state=1)
param_ET = {
'max_depth':[5,6,7,8,9,10,11,None],
'max_features':['auto','sqrt','log2'],
'criterion' : ["mse", "friedman_mse", "mae"],
'splitter' : [ "best",'random']
}

###########Decision Tree gridsearch CV for best hyperparameter##########
model_DecisionTreeRegressor = tree.DecisionTreeRegressor(random_state=1)
param_DT = {
'max_depth':[5,6,7,8,9,10,11,None],
'max_features':['auto','sqrt','log2'],
'criterion' : ["mse", "friedman_mse", "mae"],
'splitter' : [ "best",'random']
}

###########AdaBoost gridsearch CV for best hyperparameter##########
model_AdaBoostRegressor = ensemble.AdaBoostRegressor(random_state=1)
param_Ada = {
'learning_rate':[0.001,0.0025,0.005,0.0075,0.01,0.025,0.05,0.075,0.1,0.25,0.5],
'n_estimators':[50,100,200],
'loss':['linear', 'square', 'exponential']
}

seed=8461
X_train, X_test, y_train_05, y_test_05 = train_test_split(raw_input, CD_05, test_size=.012,random_state=seed)
X_train, X_test, y_train_06, y_test_06 = train_test_split(raw_input, CD_06, test_size=.012,random_state=seed)
X_train, X_test, y_train_07, y_test_07 = train_test_split(raw_input, CD_07, test_size=.012,random_state=seed)
X_train, X_test, y_train_08, y_test_08 = train_test_split(raw_input, CD_08, test_size=.012,random_state=seed)
X_train, X_test, y_train_09, y_test_09 = train_test_split(raw_input, CD_09, test_size=.012,random_state=seed)

computing_different_algorithm(model_LightGBMRegressor,param_light,'LightGBM')
computing_different_algorithm(model_XGBRegressor,param_xg,'XGBoost')
computing_different_algorithm(model_CatRegressor,param_cat,'CatBoost')
computing_different_algorithm(model_GradientBoostingRegressor,param_GB,'GradientBoost')
computing_different_algorithm(model_RandomForestRegressor,param_RF,'Random Forest')
computing_different_algorithm(model_ExtraTreeRegressor,param_ET,'Extra Tree')
computing_different_algorithm(model_DecisionTreeRegressor,param_DT,'Decision Tree')
computing_different_algorithm(model_AdaBoostRegressor,param_Ada,'AdaBoost')
