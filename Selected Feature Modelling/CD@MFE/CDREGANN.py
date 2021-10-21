#encoding:utf-8
###########import packages##########
import tensorflow as tf
import keras
from keras import optimizers
from keras import regularizers
from keras import backend as K
from keras.callbacks import EarlyStopping
from keras.callbacks import TensorBoard
from keras.constraints import max_norm
from keras.models import Sequential 
from keras.layers import Dense 
from keras.layers import Dropout 
from keras.models import Model
from keras.layers import BatchNormalization
from keras.wrappers.scikit_learn import KerasClassifier 
from keras.wrappers.scikit_learn import KerasRegressor
from keras.constraints import maxnorm 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
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
from fancyimpute import KNN
from keras.callbacks import EarlyStopping 
from sklearn.metrics import r2_score
# ### ljy改5：限制显存
# gpus = tf.config.experimental.list_physical_devices('GPU')  # 获取GPU列表
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)
#     # 失效： tf.config.experimental.set_per_process_memory_fraction(0.25)
#     # 第一个参数为原则哪块GPU，只有一块则是gpu[0],后面的memory_limt是限制的显存大小，单位为M
#     tf.config.experimental.set_virtual_device_configuration(gpu, [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024*4)]) 

early_stopping=keras.callbacks.EarlyStopping(
 monitor="val_loss", 
 patience=20, 
 verbose=0, 
 mode="auto"
)
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
    RMSE=np.sqrt(sum(squaredError)/len(absError))
    R2=r2_score(target,prediction)
    return mae,mse,RMSE,R2
###########loading data##########
fdata=pd.read_csv('database_filled_MFECD.csv',encoding="gbk")
raw_data=fdata.loc[:,[                      
                      'Ionization Potential',#0
                      'Electronegativity',#1
                      'Number of d electrons',#2
                      'ZIF or MOF Derived',#3
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
                      'Rising Rate (°C min-1)',#14
                      'Flow Cell/H-type Cell',#15
                      'Electrolyte Concentration (M)',#16
                      'Catalyst Loading (mg cm-2)',#17
                      'Electrolyte pH',#18
                      'Partial Current Density at Maximum FE (mA/cm2)'#19
                        ]]

###########data standardization##########
standardized_data = (raw_data-np.mean(raw_data,axis=0))/np.std(raw_data,axis=0)

###########defining a wrapper function for later call from each machine learning algorithms##########
raw_input=standardized_data.iloc[:,0:19]
raw_output=standardized_data.iloc[:,19]
X=raw_input.values.astype(np.float32)
y=raw_output.values.astype(np.float32)
###########fix random seed for reproducability##########
seed=36
###########train test splitting##########
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.1,random_state=seed)
raw_input_global=raw_data.iloc[:,0:19]
raw_output_global=raw_data.iloc[:,19]
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
for batch_size_number in [16,24,32]:
    for reg in [0,0.0001,0.001]:
        for dropout_rate in [0,0.1,0.2,0.3,0.4,0.5]:
            for neurons1 in range(200,1000,100):
                for epochs_number in range(150,850,100):
                    for learning_rate_search in [0.0005,0.00075,0.001,0.0025,0.005,0.0075,0.01,0.02]:
                        for activation1 in ['relu']:
                            regularizer=keras.regularizers.l2(reg)
                            ###########keras ANN model construction##########
                            model = Sequential() 
                            model.add(Dense(neurons1, input_dim=19, kernel_initializer='random_normal',
                                            bias_initializer='random_normal',activation=activation1,kernel_regularizer=regularizer)) 
                            model.add(Dropout(dropout_rate))
                            model.add(Dense(neurons1, input_dim=neurons1, kernel_initializer='random_normal',
                                            bias_initializer='random_normal',activation=activation1,kernel_regularizer=regularizer)) 
                            model.add(Dropout(dropout_rate))
                            model.add(Dense(1, input_dim=neurons1, activation='linear'))
                            adam=optimizers.Adam(lr=learning_rate_search)
                            model.compile(loss='mse', optimizer=adam)
                            print('training...')
                            model.fit(X_train, y_train,verbose=0, batch_size=batch_size_number,epochs=epochs_number,validation_split=0.2,callbacks=[early_stopping])
                            result=model.predict(X_test)
                            result_train=model.predict(X_train)
                            ###########get RMSE and R2 on the test set##########
                            x_prediction_07=result*np.std(raw_output_global,axis=0)+np.mean(raw_output_global,axis=0)
                            y_real_07=np.std(raw_output_global,axis=0)*y_test+np.mean(raw_output_global,axis=0)
                            x_prediction_07_series=pd.Series(x_prediction_07[:,0])
                            y_real_07_series=pd.Series(y_real_07)
                            #training set
                            x_prediction_07_train=result_train*np.std(raw_output_global,axis=0)+np.mean(raw_output_global,axis=0)
                            y_real_07_train=np.std(raw_output_global,axis=0)*y_train+np.mean(raw_output_global,axis=0)
                            x_prediction_07_series_train=pd.Series(x_prediction_07_train[:,0])
                            y_real_07_series_train=pd.Series(y_real_07_train)
                            ###########evaluating the regression quality##########
                            corr_ann = round(x_prediction_07_series.corr(y_real_07_series), 5)
                            error_val= compute_mae_mse_rmse(x_prediction_07[:,0],y_real_07)
                            corr_ann_train = round(x_prediction_07_series_train.corr(y_real_07_series_train), 5)
                            error_val_train= compute_mae_mse_rmse(x_prediction_07_train[:,0],y_real_07_train)
                            print('TEST SET scatter corr',corr_ann,'scatter error',error_val,'TEST R2',error_val[3])
                            print('TRAINING SET scatter corr',corr_ann_train,'scatter error',error_val_train,'R2',error_val_train[3])
                            print(neurons1,epochs_number,learning_rate_search,dropout_rate,batch_size_number,reg,activation1)                      
                            x_y_x=np.arange(-5,400,0.1)
                            x_y_y=np.arange(-5,400,0.1)
                            fig = plt.figure()
                            fig2 = plt.figure()
                            ax = fig.add_subplot(111)
                            ax.scatter(x_prediction_07[:,0],y_real_07,color='red',label='Artificial Neural Network Test Set',alpha=0.75)
                            ax.scatter(x_prediction_07_train[:,0],y_real_07_train,color='blue',label='Artificial Neural Network Training Set',alpha=0.25,marker="^")
                            ax.plot(x_y_x,x_y_y)
                            ax.axis(([-5,40,-5,40]))
                            ax2 = fig2.add_subplot(111)
                            ax2.scatter(x_prediction_07[:,0],y_real_07,color='red',label='Artificial Neural Network Test Set',alpha=0.75)
                            ax2.scatter(x_prediction_07_train[:,0],y_real_07_train,color='blue',label='Artificial Neural Network Training Set',alpha=0.25,marker="^")
                            ax2.plot(x_y_x,x_y_y)
                            ax2.axis(([-5,400,-5,400]))
                            plt.legend()
                            plt.xlabel(u"Predicted_Current_Density_@_Maximum_Faradic_Efficiency V (vs. RHE)")
                            plt.ylabel(u"Real_Current_Density_@_Maximum_Faradic_Efficiency V (vs. RHE)")
                            fig.savefig('small new 0.87 %s %s %s %s %s %s %s.png' %(neurons1,epochs_number,learning_rate_search,dropout_rate,batch_size_number,reg,activation1))
                            fig2.savefig('big new 0.87 %s %s %s %s %s %s %s.png' %(neurons1,epochs_number,learning_rate_search,dropout_rate,batch_size_number,reg,activation1))
                            plt.show()
                            K.clear_session()    
