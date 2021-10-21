options(java.parameters = "-Xmx24g")
library(xlsx)
library(readxl)
library(hydroGOF)
library(randomForest)
library(ggplot2)
library(circlize)
library(RColorBrewer)
library(dplyr)
library(randomForestExplainer)
library(pdp)
library(tcltk)
library(patchwork)
library(caret)
library(ggrepel)
library(data.table)
library(ggraph)
library(igraph)
library(tidyverse)
library(RColorBrewer) 
library(pdp)
library(Rcpp)
library(randomForest)
library(randomForestExplainer)
library(caret)
library(networkD3)
library(shiny)
library(tidyverse)

source('min_depth_distribution.R')
source('measure_importance.R')
source('min_depth_interactions.R')
source('interaction.R')
colindex<-c('IP',	'ENEG',	'NDE',	'ZIF',	'CTUBE',	'CBLACK',	'BIO',	'MT',	'NC',	'MNN',	'PYRID',	'PYRRO',	'RAMAN',	'BET',	'PTEMP',	'PTIME',	'RR',	'EC',	'CL',	'EPH',	'FE90')
r2_general <-function(preds,actual){ 
  return(1- sum((preds - actual) ^ 2)/sum((actual - mean(actual))^2))
}
nthmax<-function(x,n){
  y<-as.numeric(x)
  y<-order(y,decreasing=TRUE)
  return(x[y[n]])
}
##################################################
#data_PC=read.csv("datatest_PC.csv",head=T)
data_PC_train=read.csv("train_FE90.csv",head=T)
data_PC_test=read.csv("test_FE90.csv",head=T)
#set.seed(39)
#train <- sample(nrow(data_PC), 0.85*nrow(data_PC))
#data_PC_train=data_PC[train,]
#data_PC_test=data_PC[-train,]
#head(data_PC_train)
#head(data_PC_test)
num_trees<-200
##################################################
set.seed(2)
forest_huge <- randomForest(FE90 ~ .,type=classification, data = data_PC_train, ntree=10000, importance = TRUE)
plot(forest_huge, main = "Learning curve of the forest")
forest <- randomForest(FE90 ~ .,type=classification, data = data_PC_train, ntree=num_trees, importance = TRUE)






##################################################
im_frame<-measure_importance(forest)
im_frame[4]<-im_frame[4]/max(im_frame[4])
im_frame[5]<-im_frame[5]/max(im_frame[5])



##################################################
c1<-rep(1:length(colindex),each=length(colindex))
c11<-colindex[c1]
c2<-rep(1:length(colindex),length(colindex))
c22<-colindex[c2]
rd_frame<-data.frame(c11,c22)
colnames(rd_frame)=c('variable','root_variable')
rd_frame<-merge(rd_frame,im_frame[c(1,6)],all.x=T)
pb1 <- tkProgressBar("??????","?????? %", 0, 100) 
for (j in 1:forest$ntree){
  info1<- sprintf("?????? %d%%", round(j*100/forest$ntree)) 
  D<-calculate_max_depth(forest,j)
  interactions_frame_single<-min_depth_interactions_single(forest,j,colindex)
  rD<-calculate_rD(D,interactions_frame_single,j)
  rD<-cbind(interactions_frame_single[1:2],rD)
  rd_frame<-merge(rd_frame,rD,by=c('variable','root_variable'),all=T)
  setTkProgressBar(pb1, j*100/forest$ntree, sprintf("???? (%s)", info1),info1) 
}
close(pb1)
rd_frame[is.na(rd_frame)]<-0



for (k in 1:nrow(rd_frame)){
  rd_frame[k,num_trees+4]<-sum(rd_frame[k,4:num_trees+3])/rd_frame[k,3]
}
r_frame<-rd_frame[c(1,2,num_trees+4)]
colnames(r_frame)<-c("variable" , "root_variable" ,"r")
##################################################
type<-data.frame(label=c(colindex[-21],'FE90'),
                 type=c(rep('Electronic_param',3),rep('Morphology_Param',4),rep('Catal_prop',7),rep('Synthesis_param',3),rep('Test_param',3),rep('Target',1)),
                 color=c(rep('#98dbef',3),rep('#a4e192',4),rep('#ffc177',7),rep('#000000',3),rep('#808000',3),
                         rep('#ffb6d4',1)))
nodes<-data.frame(id=c(1:length(colindex)),label=c(colindex[-21],'FE90'))
nodes<-merge(nodes,type,all.x=T)
nodes<-arrange(nodes,nodes['id'])
edges<-cbind(r_frame,c(rep('x-x',nrow(r_frame))))
colnames(edges)<-c('Source','Target','Weight','Type')
edges[is.na(edges)]<-0
edges[3]<-edges[3]/max(edges[3])
edges[3][edges[3]<0.5]<-0
edges<-edges[-which(edges[3]==0),]
edges<-edges[-which(edges[1]==edges[2]),]
for (j in 1:nrow(edges)){
  j1<-which(edges[j,1]==edges[2])
  j2<-which(edges[j,2]==edges[1])
  j3<-intersect(j1,j2)
  if (length(j3)!=0){
    edges[j,3]<-mean(c(edges[j,3],edges[j3,3]))
    edges<-edges[-j3,]
  }
}
x_y<-data.frame(Source=c(rep('FE90',20)),
                  Target=im_frame$variable,
                  Weight=im_frame[4],
                  Type=c(rep('x-y',20)))
colnames(x_y)<-c('Source','Target','Weight','Type')
edges<-rbind(edges,x_y)
edges[3][edges[3]<=0]<-0
for (j in 1:nrow(nodes[1])){
  edges[edges==nodes[j,1]]<-nodes[j,2]
}
##################################################
interactions_frame_rel <- min_depth_interactions(forest,mean_sample = "relevant_trees", uncond_mean_sample = "relevant_trees")
interactions_frame <- min_depth_interactions(forest)


plot_min_depth_distribution(forest)
plot_min_depth_interactions(interactions_frame)
plot_min_depth_interactions(interactions_frame_rel)
##################################################
write.csv(interactions_frame,paste0('sum/inter_sum','.csv'),row.names=FALSE,fileEncoding='UTF-8')
##################################################
total_inter_counts<-c(3079, 3744, 2945, 2544, 1074, 857, 1071, 3915, 4165, 3992, 4090, 3769, 3580, 3984, 3426, 3064, 3490, 1362, 3368, 3005)
total_inter_counts<-100*(total_inter_counts/max(total_inter_counts))*(total_inter_counts/max(total_inter_counts))*(total_inter_counts/max(total_inter_counts))*(total_inter_counts/max(total_inter_counts))
nodes$total_inter<-c(total_inter_counts,max(total_inter_counts))
##################################################
#write.csv(nodes,paste0('test/nodes----','FE90','.csv'),row.names=FALSE,fileEncoding='UTF-8')
#write.csv(edges,paste0('test/edges----','FE90','.csv'),row.names=FALSE,fileEncoding='UTF-8')
##################################################
#EDG<-read_csv('test/edges----FE90.csv')
#NDS<-read_csv('test/nodes----FE90.csv')
##################################################
EDG<-edges
NDS<-nodes
EDG[1]<-sapply(EDG[1],as.numeric)
EDG[2]<-sapply(EDG[2],as.numeric)
EDG[1]<-EDG[1]-1
NDS[2]<-NDS[2]-1
EDG[3]<-10*EDG[3]*EDG[3]*EDG[3]*EDG[3]
red_index<-((EDG $ Weight> (nthmax(EDG$Weight,21)))&(EDG$Source!=20)&(EDG$Target!=20))[,1]
##################################################
NETWORK <- forceNetwork(Links = EDG,#线性质数据�?  
                        Nodes = NDS,#节点性质数据�?  
                        Source = "Source",#连线的源变量  
                        Target = "Target",#连线的目标变�?  
                        Value = "Weight",#连线的粗细�?  
                        NodeID = "label",#节点名称  
                        Group = "type",#节点的分�?  
                        Nodesize = "total_inter" ,#节点大小，节点数据框�?  
                        ###美化部分 
                        charge=-4000,
                        fontFamily="Arial",#字体设置�?"华文行楷" �?  
                        fontSize = 40, #节点文本标签的数字字体大小（以像素为单位）�?  
                        linkColour=ifelse(red_index,"red","black"),#连线颜色,black,red,blue,    
                        colourScale=JS("d3.scaleOrdinal(d3.schemeCategory10);"), #节点颜色,red，蓝色blue,cyan,yellow�?  
                        #linkWidth,#节点颜色,red，蓝色blue,cyan,yellow�?  
                        #charge = -100,#数值表示节点排斥强度（负值）或吸引力（正值）    
                        opacity = 0.9,  
                        legend=F,#显示节点分组的颜色标�?  
                        arrows=F,#是否带方�?  
                        #bounded=F,#是否启用限制图像的边�?  
                        #opacityNoHover=1.0,#当鼠标悬停在其上时，节点标签文本的不透明度比例的数�? 
                        opacityNoHover=TRUE, #显示节点标签文本
                        #zoom = T#允许放缩，双击放�? 
                        width = 1200, height = 1200
)
NETWORK
saveNetwork(NETWORK,"networkRF2#.html",selfcontained=TRUE)#save HTML

