options(java.parameters = "-Xmx24g")

library(xgboost)
library(lightgbm)
library(EIX)
library(Matrix)
library(ModelMetrics)
library(data)
colindex<-c('IP',	'ENEG',	'NDE',	'ZIF','CTUBE','BIO',	'MT',	'NC',	'MNN',	'PYRID',	'PYRRO',	'RAMAN',	'BET',	'PTEMP',	'RR',	'FLOWCELL',	'EC',	'CL',	'EPH',	'CD')
r2_general <-function(preds,actual){ 
  return(1- sum((preds - actual) ^ 2)/sum((actual - mean(actual))^2))
}
nthmax<-function(x,n){
  y<-as.numeric(x)
  y<-order(y,decreasing=TRUE)
  return(x[y[n]])
}
##################################################
data_PC=read.csv("data_CDREG.csv",head=T)
data_PC_train=read.csv("train_CDREG.csv",head=T)
data_PC_test=read.csv("test_CDREG.csv",head=T)
#set.seed(9)
#train <- sample(nrow(data_PC), 0.85*nrow(data_PC))
#data_PC_train=data_PC[train,]
#data_PC_test=data_PC[-train,]
#head(data_PC_train)
#head(data_PC_test)
num_trees<-100
##################################################
set.seed(3)

num_features<-dim(data_PC_train)[2]-1
#LGBM_train<-as.matrix(data_PC[,1:num_features])
LGBM_train<-as.matrix(data_PC_train[,1:dim(data_PC_train)[2]])
LGBM_test<-as.matrix(data_PC_test[,1:dim(data_PC_train)[2]])

ddtrain <- lgb.Dataset(data = LGBM_train[,1:num_features], label = LGBM_train[,dim(data_PC_train)[2]])
ddtest <- lgb.Dataset.create.valid(ddtrain, data = LGBM_test[,1:num_features], label =LGBM_test[,dim(data_PC_train)[2]])
valids <- list(test = ddtest)

# Method 1 of training
params <- list(
  objective = "regression",
   min_data = 1L
  , learning_rate = 0.0075
)
lgbm_model <- lgb.train(
  params
  , ddtrain
  , num_trees
  , valids
  , early_stopping_rounds = 10
)

??lightgbm
model
model_matrix_all<-model.matrix(CD ~ . - 1, data_PC)
model_martix_train <- model.matrix(CD ~ . - 1, data_PC_train)
model_martix_test <- model.matrix(CD ~ . - 1, data_PC_test)



lolli<-lollipop(lgbm_model,model_matrix_all)
plot(lolli,labels='topAll',log_scale=T,threshold = 0.2)

imp<-importance(lgbm_model,model_matrix_all,option='both')
plot(imp,top = 15)
inter<-interactions(lgbm_model,model_matrix_all,option='interactions')
plot(inter)
inter2<-interactions(lgbm_model,model_matrix_all,option='pairs')
plot(inter2)
