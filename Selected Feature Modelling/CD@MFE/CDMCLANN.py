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
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from sklearn.metrics import roc_curve, auc
from scipy import interp
from itertools import cycle
early_stopping=keras.callbacks.EarlyStopping(
 monitor="val_loss", 
 patience=20, 
 verbose=0, 
 mode="auto"
)
nb_classes=3
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
                      'CLCD@MFE'#the classification target#19
                        ]]

###########data standardization##########
standardized_data = (raw_data-np.mean(raw_data,axis=0))/np.std(raw_data,axis=0)

###########defining a wrapper function for later call from each machine learning algorithms##########
raw_input=standardized_data.iloc[:,0:19]
raw_output=raw_data.iloc[:,19]

# encode class values as integers
encoder = LabelEncoder()
encoded_Y = encoder.fit_transform(raw_output)
# convert integers to dummy variables (one hot encoding)
dummy_y = np_utils.to_categorical(encoded_Y)

def auc_ANN(y_label,y_pre,neurons1,epochs_number,dropout_rate,batch_size_number,reg,act):  
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
    plt.title('ROC Curve (Multi-Class Classification) of Artificial Neural Network')
    plt.legend(loc="lower right")
    plt.savefig('ROC Curve of %s %s %s %s %s %s CDMFEMCL 3D ANN.png' %(neurons1,epochs_number,dropout_rate,batch_size_number,reg,act))

###########fix random seed for reproducability##########
seed=167
###########train test splitting##########
X_train, X_test, y_train, y_test = train_test_split(raw_input, dummy_y, test_size=.1,random_state=seed)
X_train_o, X_test_o, y_train_o, y_test_o = train_test_split(raw_input, raw_output, test_size=.1,random_state=seed)
raw_input_global=raw_data.iloc[:,0:19]
raw_output_global=raw_data.iloc[:,19]
###########wrap up fuction for later call for OPTIMIZATION##########

accuracy={}
for neurons1 in [100,200,400,600,800]:
    for dropout_rate in [0,0.25,0.5]:
        for batch_size_number in [8,16,32]:
            for reg in [0,0.0001,0.001]:
                for act in ['sigmoid','tanh','relu','softsign']:                        
                    for epochs_number in range(100,800,100):
                        regularizer=keras.regularizers.l2(reg)
                        ###########keras ANN model construction########## 
                        model = Sequential() 
                        model.add(Dense(neurons1, input_dim=19, kernel_initializer='random_normal',
                                        bias_initializer='random_normal',activation=act,kernel_regularizer=regularizer)) 
                        model.add(Dropout(dropout_rate))                        
                        model.add(Dense(neurons1, input_dim=neurons1, kernel_initializer='random_normal',
                                        bias_initializer='random_normal',activation=act,kernel_regularizer=regularizer)) 
                        model.add(Dropout(dropout_rate))
                        model.add(Dense(3, input_dim=neurons1, activation='softmax'))
#                         model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
                        model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
                        model.fit(X_train, y_train,verbose=0, batch_size=batch_size_number,epochs=epochs_number,validation_split=0.2)
                        print(neurons1,epochs_number,dropout_rate,batch_size_number,reg,act)
                        test_pred = model.predict(X_test)
                        train_pred = model.predict(X_train)
                        id_test = np.argmax(test_pred, axis=1)
                        id_train = np.argmax(train_pred, axis=1)
                        y_score=model.predict_proba(X_test)
                        print(classification_report(id_train,np.argmax(y_train, axis=1)))
                        print(classification_report(id_test,np.argmax(y_test, axis=1)))
                        final_result=classification_report(id_test,np.argmax(y_test, axis=1),output_dict=True)
                        ac=final_result['accuracy']
                        accuracy[ac]=[neurons1,epochs_number,dropout_rate,batch_size_number,reg,act]
                        auc_ANN(y_test,y_score,neurons1,epochs_number,dropout_rate,batch_size_number,reg,act)
                        K.clear_session()   
print(accuracy)
