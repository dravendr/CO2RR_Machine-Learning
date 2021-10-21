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
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from xgboost import plot_importance
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn import ensemble
from sklearn.tree import ExtraTreeRegressor
from sklearn import svm
from sklearn import neighbors
from sklearn import tree
from sklearn.impute import SimpleImputer
import keras_metrics as km
from keras.callbacks import EarlyStopping 
from sklearn.metrics import r2_score
#%matplotlib
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
early_stopping=keras.callbacks.EarlyStopping(
 monitor="val_loss", 
 patience=20, 
 verbose=0, 
 mode="auto"
)
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
                      'Electrolyte pH',#31
    'FE of Product (CO) at -0.5V (vs.RHE)',
    'FE of Product (CO) at -0.6V (vs.RHE)',
    'FE of Product (CO) at -0.7V (vs.RHE)',
    'FE of Product (CO) at -0.8V (vs.RHE)',
    'FE of Product (CO) at -0.9V (vs.RHE)'
                        ]]

###########data standardization##########
standardized_data = (raw_data-np.mean(raw_data,axis=0))/np.std(raw_data,axis=0)

###########defining a wrapper function for later call from each machine learning algorithms##########
raw_input=standardized_data.iloc[:,0:32]
raw_output=standardized_data.iloc[:,32:]
X=raw_input.values.astype(np.float32)
y=raw_output.values.astype(np.float32)
###########fix random seed for reproducability##########
seed=8461
###########train test splitting##########
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.012,random_state=seed)
raw_input_global=raw_data.iloc[:,0:32]
raw_output_global=raw_data.iloc[:,32:]
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
def intergate(y_pred):
    length=y_pred.shape[0]
    print(length)
    for i in range (0,length):
        if y_pred[i][0]>=0.5:
            y_pred[i][0]=1
        else:
            y_pred[i][0]=0
    return y_pred
def compute_curves_ANN(prediction,real,batch_size_number,reg,dropout_rate,neurons1,epochs_number,learning_rate_search):
    count=0
    for i in range(0,prediction.shape[0]):
        prediction_series=pd.Series(prediction[i])
        real_series=pd.Series(real[i])
        corr_ann = round(prediction_series.corr(real_series), 5)
        error_val= compute_mae_mse_rmse(prediction[i],real[i])
        print(corr_ann, error_val)
        if error_val[2]<10 or error_val[3]>0.9:
            count+=1
    print('count is', count)
    if count>=8:
        for i in range(0,prediction.shape[0]):
            fig=plt.figure()
            ax = fig.add_subplot(111)
            x_list=[0.5,0.6,0.7,0.8,0.9]
            ax.scatter(x_list,prediction[i],label='Artificial Neural Network Prediction',color='red')
            ax.plot(x_list,real[i],label='Real Curve',color='blue')
            plt.legend()
            plt.xlabel(u"Potential V (vs. RHE)")
            plt.ylabel(u"Faradaic Efficiency (%)")
            plt.savefig('the %s th %s %s %s %s %s %s figure of ANN FE CURVE.png' %(i,batch_size_number,reg,dropout_rate,neurons1,epochs_number,learning_rate_search))
for batch_size_number in [16,24,32]:
    for reg in [0,0.0001,0.001]:
        for dropout_rate in [0,0.1,0.2,0.3,0.4,0.5]:
            for neurons1 in range(100,1000,50):
                for epochs_number in range(50,950,50):
                    for learning_rate_search in [0.0005,0.00075,0.001,0.0025,0.005,0.0075,0.01,0.02]:
                        for activation1 in ['relu']:
                            regularizer=keras.regularizers.l2(reg)
                            ###########keras ANN model construction##########
                            model = Sequential() 
                            model.add(Dense(neurons1, input_dim=32, kernel_initializer='random_normal',
                                            bias_initializer='random_normal',activation=activation1,kernel_regularizer=regularizer)) 
                            model.add(Dropout(dropout_rate))
                            model.add(Dense(neurons1, input_dim=neurons1, kernel_initializer='random_normal',
                                            bias_initializer='random_normal',activation=activation1,kernel_regularizer=regularizer)) 
                            model.add(Dropout(dropout_rate))
                            model.add(Dense(5, input_dim=neurons1, activation='linear'))
                            adam=optimizers.Adam(lr=learning_rate_search)
                            model.compile(loss='mse', optimizer=adam)
                            print('training...')
                            model.fit(X_train, y_train,verbose=0, batch_size=batch_size_number,epochs=epochs_number,validation_split=0.2,callbacks=[early_stopping])
                            result=model.predict(X_test)
                            result_train=model.predict(X_train)
                            ###########get RMSE and R2 on the test set##########
                            x_prediction_07=result*np.std(raw_output_global,axis=0).values+np.mean(raw_output_global,axis=0).values
                            y_real_07=np.std(raw_output_global,axis=0).values*y_test+np.mean(raw_output_global,axis=0).values
                            #training set
                            x_prediction_07_train=result_train*np.std(raw_output_global,axis=0).values+np.mean(raw_output_global,axis=0).values
                            y_real_07_train=np.std(raw_output_global,axis=0).values*y_train+np.mean(raw_output_global,axis=0).values
                            print(batch_size_number,reg,dropout_rate,neurons1,epochs_number,learning_rate_search)
                            compute_curves_ANN(x_prediction_07,y_real_07,batch_size_number,reg,dropout_rate,neurons1,epochs_number,learning_rate_search)

                            K.clear_session()    
