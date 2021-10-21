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
library(ggepi)
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
#####################################################################
#####                    data info (necessary)                  #####
#####################################################################
label1<-excel_sheets('imm/all-rf-op.xlsx')
label2<-excel_sheets('burden/all-rf-op.xlsx')
label<-c(label1,label2)
colindex<-c('Carbon', 'M.C.', 'Com.1', 'Com.2', 'Oxide', 'Salt', 'Dim.0','Dim.1',
            'Dim.2', 'Hollow', 'D', 'L', 'S.P', 'S.N', 'Zeta', 'SSA', 'Rat', 'Mice',
            'male', 'female', 'M.A', 'M.W', 'OPA', 'IH.', 'IN.N.', 'IN.T.', 'I.T.', 
            'E.D', 'E.T', 'R.D', 'Dose', 'Index')
#####################################################################
#####                    Scatter output                         #####
#####################################################################

source(file = 'scatterplot.R',encoding = 'utf-8')
#label1<-excel_sheets('??your path??/code/imm/all-rf-op.xlsx')
#label2<-excel_sheets('??your path??/code/burden/all-rf-op.xlsx')
label<-c(label1,label2)
scatterlist<-list()
for ( i in 1:12){
  scatterlist[[label1[i]]]<-reg(1,i)
}
scatterlist[[label2[1]]]<-reg(2,1,0,1)
scatterlist[[label2[2]]]<-reg(2,2,0,0.4)
scatterlist[[label2[3]]]<-reg(2,3,0,0.1)

setwd('D:/document/ywj/PC')
save(scatterlist,file='Rda/scatter.rda')
load(file='Rda/scatter.rda')
p<-scatterlist[[1]]
for (i in c(2:6,8,10:12,14,15)){
  p<-p+scatterlist[[i]]
}
p<-p+plot_layout(ncol=4)
#ggsave('plot/scatter/scatter-SI.pdf',width=12,height=9)
p2<-scatterlist[[7]]+scatterlist[[9]]+scatterlist[[13]]
#ggsave('plot/scatter/scatter-fig3.pdf',width=12,height=3)

#####################################################################
#####                    R2 distribution                        #####
#####################################################################
r2_summary<-data.frame(index=label,r2<-NA,rmse<-NA)
for (i in 1:15){
  if (i<=12){
    data<-read.xlsx('imm/all-rf-cor.xlsx',i)
  } else {
    data<-read.xlsx('burden/all-rf-cor.xlsx',i-12)
  }
  r_index1<-mean(data$train[1:10])
  r_index2<-mean(data$test[1:10])
  e_index<-mean(data$rmse_test[1:10])
  r2_summary[i,2]<-r_index1
  r2_summary[i,3]<-r_index2
  r2_summary[i,4]<-e_index
}
write.xlsx(r2_summary,'r2-summary.xlsx')

#source(file = '??your path??/code/r2distribution.R',encoding = 'utf-8')
cordf_all<-model_cor('all-rf-cor.xlsx')
cordf_sbs<-model_cor('sbs-rf-cor.xlsx')
cordf_ann<-model_cor('ann-cor.xlsx')
cordf_svm<-model_cor('all-svm-cor.xlsx')
#save(cordf_all,cordf_sbs,cordf_ann,file='??your path??/code/Rda/cordf.rda')
#load(file='??your path??/code/Rda/cordf.rda')

cor_rmse_output('all-rf-cor.xlsx')
cor_rmse_output('sbs-rf-cor.xlsx')
cor_rmse_output('ann-cor.xlsx')
cor_rmse_output('all-svm-cor.xlsx')


corplist<-list()
corplist[['all']]<-r2_dis(cordf_all,'all')
corplist[['sbs']]<-r2_dis(cordf_sbs,'sbs')
corplist[['ann']]<-r2_dis(cordf_ann,'ann')
corplist[['svm']]<-r2_dis(cordf_svm,'svm',1.05)
#load(file='??your path??/code/Rda/corplot.rda')

#####################################################################
#####                    Importance graph                       #####
#####################################################################

colindex<-c('Carbon', 'M.C.', 'Com.1', 'Com.2', 'Oxide', 'Salt', 'Dim.0','Dim.1',
            'Dim.2', 'Hollow', 'D', 'L', 'S.P', 'S.N', 'Zeta', 'SSA', 'Rat', 'Mice',
            'male', 'female', 'M.A', 'M.W', 'OPA', 'IH.', 'IN.N.', 'IN.T.', 'I.T.', 
            'E.D', 'E.T', 'R.D', 'Dose', 'Index')
imp_sum<-data.frame()
setwd('imm')
for (i in 1:12){
  imp<-read.xlsx('all-rf-imp.xlsx',i)
  imp.mean<-data.frame(t(apply(imp[2:32],2,function(x) mean(x,na.rm=TRUE))))
  imp.mean<-imp.mean/max(imp.mean)
  imp_sum<-rbind(imp_sum,imp.mean)
}
#setwd('??your path??/code/burden')
for (i in 1:3){
  imp<-read.xlsx('all-rf-imp.xlsx',i)
  imp.c<-t(apply(imp[2:30],2,function(x) mean(x,na.rm=TRUE)))
  imp.c<-imp.c/max(imp.c)
  imp.c<-c(imp.c[1:23],NA,NA,imp.c[24:29])
  imp_sum[12+i,]<-imp.c
}
colnames(imp_sum)<-colindex[-32]

edges<-data.frame()
for (i in 1:15){
  edges[(31*i-30):(i*31),1]<-label[i]
}
edges[2]<-NA

for (i in 1:31){
  edges[c(seq(i,15*31,31)),3]<-colindex[i]
}

for (i in 1:15){
  for (j in 1:31){
    edges[i*31+j-31,4]<-imp_sum[i,j]
  }
}
for (i in 1:15){
  edges[(31*i-30):(i*31),5]<-i
}
colnames(edges)=c('from','to','label','mse','group')

edges<-edges %>% 
  group_by(group) %>% 
  arrange(desc(mse), .by_group=TRUE)

edges<-edges[-which(edges[4]==0),]
edges<-edges[-which(is.na(edges[4])),]

for (i in 1:nrow(edges)){
  edges$to[i]<-paste0(edges$group[i],'_',edges$label[i])
}

edges_in1<-data.frame(from=c('O','O'),to=c('immunotoxicity','burden'),
                      label=c('immunotoxicity','burden'),mse=c(0.8,0.8),group=c(0,0))
edges_in2<-data.frame(from=c(rep('immunotoxicity',12)),to=label1,
                      label=label1,mse=c(rep(0.8,12)),group=c(rep(0,12)))
edges_in3<-data.frame(from=c(rep('burden',3)),to=c('Lung','Liver','BALF'),
                      label=c('Lung','Liver','BALF'),mse=c(0.8,0.8,0.8),group=c(0,0,0))
edges_all=rbind(edges_in1,edges_in2,edges_in3,edges)

nodes<-data.frame(name=c('O',edges_all$to),
                  label=c('O',edges_all$label),
                  weight=c(0.8,as.numeric(edges_all$mse)),
                  group=c('A','B','C',rep('B',12),rep('C',3),edges_all$group[-c(1:17)]))
n_leaves<-nrow(nodes)-19+1
nodes$id<-n_leaves/360*90
nodes$id[19:nrow(nodes)]<-seq(1:n_leaves)
#angleָ???Ǳ?ǩ??ת?ĽǶ?
nodes$angle<- 321.3043-360 * nodes$id / n_leaves
# nodes$angle<-ifelse(nodes$angle < -90, nodes$angle+180, nodes$angle)
# nodes$angle<-ifelse(nodes$angle > 90,nodes$angle-180,nodes$angle)
nodes$angle[intersect(which(nodes$id>60),which(nodes$id<211))]<-
  nodes$angle[intersect(which(nodes$id>60),which(nodes$id<211))]-180

nodes$angle[1:18]<-0
#??ɫӳ??ʱ????factor(group)??ѡ?????ʵķ?????ǩ?Զ?Ӧ??ɫ??˳??
for (i in 1:15){
  nodes$group[nodes$group==i]<-letters[i+3] 
}

mycolor1<-c('#000000','#B2DF8A','#FB9A99',
            '#dc7bff','#a77bff','#7b84ff','#7bb8ff','#7beeff',
            '#7bffdc','#7bffa7','#84ff7b','#b8ff7b','#eeff7b',
            '#ffdc7b','#ffa77b','#ff7b84','#ff7bb8','#ff7bee')

mycolor2<-c('#000000','#B2DF8A','#FB9A99',
            '#c526ff','#6e26ff','#2635ff','#268bff','#26e2ff',
            '#26ffc5','#26ff6e','#35ff26','#8bff26','#e2ff26',
            '#ffc526','#ff6e26','#ff2635','#ff268b','#ff26e2')

mygraph <- graph_from_data_frame( edges_all, vertices = nodes, directed = TRUE )

ggraph(mygraph, layout = 'dendrogram', circular = TRUE) + 
  geom_edge_diagonal(aes(colour=..index..)) +
  scale_edge_colour_distiller(palette = "RdPu") +
  geom_node_text(aes(x = x*1.25, y=y*1.25, angle = angle,label=label,color=group),
                 size=2, alpha=1) +
  geom_node_point(aes( x = x*1.07, y=y*1.07, fill=group, size=weight), 
                  shape=21,stroke=0.2,color='black',alpha=1) +
  scale_colour_manual(values= mycolor2) +#??
  scale_fill_manual(values= mycolor1) +#??
  scale_size_continuous( range = c(0.5,6) ) +
  expand_limits(x = c(-1.3, 1.3), y = c(-1.3, 1.3))+
  theme_void() +
  theme(
    legend.position="none",
    plot.margin=unit(c(0,0,0,0),"cm"))+
  coord_fixed(ratio=1)
#setwd('??your path??/code')
ggsave('plot/imp_circle_python.pdf',width=5,height=5)

#####################################################################
#####                    RF analysis                            #####
#####################################################################
setwd('D:/document/ywj/PC')
rf.list<-list()
for (i in 1:12){
  data<-read.xlsx('imm/RF20210202.xlsx',i+1,header=TRUE)
  colnames(data)<-colindex
  # train<-read.xlsx('imm/ss-index-all.xlsx',i)$X0
  # data_train<-data[train,]
  # data_test<-data[-train,]
  seed<-rep(1,12)
  seed[8]<-2
  seed[12]<-8
  set.seed(seed[i])
  rf.list[[label1[i]]]<-local({
    data=data
    randomForest(Index~.,data=data,importance=TRUE,proximity=T,ntree=500,mtry=5)
  })
  rf<-rf.list[[label[i]]]
  pre_train<-predict(rf)
  cor_train<-cor(pre_train,data$Index)
  rmse_train<-rmse(pre_train,data$Index)
  # pre_test<-predict(rf,data_test[,1:(ncol(data_test)-1)])
  # cor_test<-cor(pre_test,data_test$Index)
  # rmse_test<-rmse(pre_test,data_test$Index)
  print(cor_train)
  # print(cor_test)
}
for (i in 1:3){
  data<-read.xlsx('burden/clearance2021.xlsx',i+2,header=TRUE)[-1]
  colnames(data)<-colindex[-c(2,24,25)]
  # train<-read.xlsx('burden/ss-index-all.xlsx',i)$X0
  # data_train<-data[train,]
  # data_test<-data[-train,]
  seed<-c(1,1,2)
  set.seed(seed[i])
  rf.list[[label2[i]]]<-local({
    data=data
    rf<-randomForest(Index~.,data=data,importance=TRUE,proximity=T,ntree=500,mtry=5)
  })
  rf<-rf.list[[label2[i]]]
  pre_train<-predict(rf)
  cor_train<-cor(pre_train,data$Index)
  rmse_train<-rmse(pre_train,data$Index)
  # pre_test<-predict(rf,data_test[,1:(ncol(data_test)-1)])
  # cor_test<-cor(pre_test,data_test$Index)
  # rmse_test<-rmse(pre_test,data_test$Index)
  
  print(cor_train)
  # print(cor_test)
}
save(rf.list,file='Rda/rflist.rda')
#----------------------------------------------
# Importance analysis
#----------------------------------------------
# multi-way importance analysis
md<-list()
mi<-list()
colindexf<-factor(colindex[-32],levels=colindex[-32])
impframe<-data.frame(index=colindexf)
for (i in 1:15){
  print(i)
  rf<-rf.list[[i]]
  imp<-importance(rf)
  incmse<-data.frame(index=names(imp[,2]),incmse=imp[,1]/max(imp[,1]))
  colnames(incmse)[2]<-paste0(i)
  impframe<-merge(impframe,incmse,by='index',all=T)

  min_depth_frame<-min_depth_distribution(rf)
  md[[label[i]]]<-min_depth_frame

  im_frame<-measure_importance(rf)
  im_frame[4]<-im_frame[4]/max(im_frame[4])
  im_frame[5]<-im_frame[5]/max(im_frame[5])
  mi[[label[i]]]<-im_frame
}
colnames(impframe)[2:16]<-label
save(impframe,md,mi,file='Rda/multi-importance.rda')



edges<-data.frame()
for (i in 1:15){
  edges[(31*i-30):(i*31),1]<-label[i]
}
edges[2]<-NA

for (i in 1:31){
  edges[c(seq(i,15*31,31)),3]<-colindex[i]
}

for (i in 1:15){
  for (j in 1:31){
    edges[i*31+j-31,4]<-impframe[j,i+1]
  }
}
for (i in 1:15){
  edges[(31*i-30):(i*31),5]<-i
}
colnames(edges)=c('from','to','label','mse','group')

edges<-edges %>%
  group_by(group) %>%
  arrange(desc(mse), .by_group=TRUE)

edges<-edges[-which(edges[4]<=0),]
edges<-edges[-which(is.na(edges[4])),]

for (i in 1:nrow(edges)){
  edges$to[i]<-paste0(edges$group[i],'_',edges$label[i])
}

edges_in1<-data.frame(from=c('O','O'),to=c('immunotoxicity','burden'),
                      label=c('immunotoxicity','burden'),mse=c(0.8,0.8),group=c(0,0))
edges_in2<-data.frame(from=c(rep('immunotoxicity',12)),to=label1,
                      label=label1,mse=c(rep(0.8,12)),group=c(rep(0,12)))
edges_in3<-data.frame(from=c(rep('burden',3)),to=c('Lung','Liver','BALF'),
                      label=c('Lung','Liver','BALF'),mse=c(0.8,0.8,0.8),group=c(0,0,0))
edges_all=rbind(edges_in1,edges_in2,edges_in3,edges)

nodes<-data.frame(name=c('O',edges_all$to),
                  label=c('O',edges_all$label),
                  weight=c(0.8,as.numeric(edges_all$mse)),
                  group=c('A','B','C',rep('B',12),rep('C',3),edges_all$group[-c(1:17)]))
n_leaves<-nrow(nodes)-19+1
nodes$id<-n_leaves/360*90
nodes$id[19:nrow(nodes)]<-seq(1:n_leaves)
#angleָ???Ǳ?ǩ??ת?ĽǶ?
nodes$angle<- 301.206-360 * nodes$id / n_leaves
# nodes$angle<-ifelse(nodes$angle < -90, nodes$angle+180, nodes$angle)
# nodes$angle<-ifelse(nodes$angle > 90,nodes$angle-180,nodes$angle)
# nodes$angle[intersect(which(nodes$id>35),which(nodes$id<233))]<-
#   nodes$angle[intersect(which(nodes$id>35),which(nodes$id<233))]-180
nodes$angle[1:18]<-0
#??ɫӳ??ʱ????factor(group)??ѡ?????ʵķ?????ǩ?Զ?Ӧ??ɫ??˳??
for (i in 1:15){
  nodes$group[nodes$group==i]<-letters[i+3]
}

mycolor1<-c('#000000','#B2DF8A','#FB9A99',
            '#dc7bff','#a77bff','#7b84ff','#7bb8ff','#7beeff',
            '#7bffdc','#7bffa7','#84ff7b','#b8ff7b','#eeff7b',
            '#ffdc7b','#ffa77b','#ff7b84','#ff7bb8','#ff7bee')

mycolor2<-c('#000000','#B2DF8A','#FB9A99',
            '#c526ff','#6e26ff','#2635ff','#268bff','#26e2ff',
            '#26ffc5','#26ff6e','#35ff26','#8bff26','#e2ff26',
            '#ffc526','#ff6e26','#ff2635','#ff268b','#ff26e2')

mygraph <- graph_from_data_frame( edges_all, vertices = nodes, directed = TRUE )

ggraph(mygraph, layout = 'dendrogram', circular = TRUE) +
  geom_edge_diagonal(aes(colour=..index..)) +
  scale_edge_colour_distiller(palette = "RdPu") +
  geom_node_text(aes(x = x*1.25, y=y*1.25, angle = angle,label=label,color=group),
                 size=2, alpha=1) +
  geom_node_point(aes( x = x*1.07, y=y*1.07, fill=group, size=weight),
                  shape=21,stroke=0.2,color='black',alpha=1) +
  scale_colour_manual(values= mycolor2) +#??
  scale_fill_manual(values= mycolor1) +#??
  scale_size_continuous( range = c(0.5,6) ) +
  expand_limits(x = c(-1.3, 1.3), y = c(-1.3, 1.3))+
  theme_void() +
  theme(
    legend.position="none",
    plot.margin=unit(c(0,0,0,0),"cm"))+
  coord_fixed(ratio=1)
ggsave('plot/imp_circle_r1.pdf',width=5,height=5)


# importance plot
load(file='Rda/rflist.rda')
load(file='Rda/multi-importance.rda')
mdplot<-list()
miplot<-list()

for (i in 1:15){
  print(i)
  min_depth_frame<-md[[i]]
  mdplot[[label[i]]]<-local({
    min_depth_frame=min_depth_frame
    plot_min_depth_distribution(min_depth_frame,k=14)+
      theme(panel.grid.major = element_blank(),panel.grid.minor = element_blank())+
      theme(legend.key.size = unit(0.5,'line'),legend.title = element_text(size=rel(0.6)),
            legend.text = element_text(size=rel(0.5)))
  })
  ggsave(paste0('plot/md/md_',label[i],'.pdf'),width=7,height=7)

  im_frame=mi[[i]]
  im_frame$p_value<-im_frame$p_value/5
  miplot[[label[i]]]<-local({
    im_frame=im_frame
    plot_multi_way_importance(im_frame, x_measure = "mse_increase",
                              y_measure = "node_purity_increase",
                              size_measure = "p_value", no_of_labels = 5)+
      theme(panel.grid.major = element_blank(),panel.grid.minor = element_blank())+
      theme(axis.line=element_line(color='black'),
            axis.ticks.length=unit(0.5,'line'))+
      coord_fixed(ratio=1)+
      theme(legend.position=c(0.1,0.8))
  })
  ggsave(paste0('plot/mi/m_im_',label[i],'.pdf'),width=5,height=5)
}
save(mdplot,miplot,file='Rda/importanceplot.rda')
#----------------------------------------------
# Feature Interaction Calculate
#----------------------------------------------
load(file='Rda/rflist.rda')
load(file='Rda/multi-importance.rda')

inter_list<-list()
for (i in 1:2){
  print(i)
  im_frame<-mi[[i]]
  rf<-rf.list[[i]]
  vars <- important_variables(im_frame, k = 5, measures = c("mean_min_depth","no_of_trees"))
  interactions_frame <- min_depth_interactions(rf, vars)
  interactions_frame <- arrange(interactions_frame,-interactions_frame[,4])
  inter_list[[label[i]]]<-interactions_frame
  #load(paste0('Rda/inter_',Label[i+2],'.rda'))
  #head(interactions_frame[order(interactions_frame$occurrences, decreasing = TRUE), ])
}
save(inter_list,file='Rda/inter1.rda')

fiplot<-list()
for (i in 1:2){
  interactions_frame<-inter_list[[i]]
  hlim<-ceiling(max(interactions_frame[1:25,3],interactions_frame[1:25,6]))
  fip<-plot_min_depth_interactions(interactions_frame,k=25)+
        scale_y_continuous(limits=c(0,hlim+1.5),expand=c(0,0))+
        scale_fill_gradient(low='#00d538',high='#ff5e24')+
        theme(panel.grid.major = element_blank(),panel.grid.minor = element_blank())+
        theme(legend.position=c(0.3,0.8),legend.box="horizontal")
  fiplot[[label[i]]]<-fip
  ggsave(paste0('plot/inter/inter',label[i],'.pdf'),width=7,height=4)
}
save(fiplot,file='Rda/inter_plot.rda')

#----------------------------------------------
# Feature Interaction Calculate
#----------------------------------------------
source('min_depth_distribution.R')
source('measure_importance.R')
source('min_depth_interactions.R')
source('interaction.R')
rdlist<-list()
r_interaction<-list()
pb <- tkProgressBar("?ܽ???","?????? %", 0, 100) 
for (i in 1:length(label)){
  print(paste0(i,'/',length(label)))
  info<- sprintf("?????? %d%%", round(i*100/length(label))) 
  rf<-rf.list[[i]]
  c1<-rep(1:length(colindex),each=length(colindex))
  c11<-colindex[c1]
  c2<-rep(1:length(colindex),length(colindex))
  c22<-colindex[c2]
  rd_frame<-data.frame(c11,c22)
  colnames(rd_frame)=c('variable','root_variable')
  im_frame<-mi[[i]]
  rd_frame<-merge(rd_frame,im_frame[c(1,6)],all.x=T)
  
  pb1 <- tkProgressBar("??????","?????? %", 0, 100) 
  for (j in 1:rf$ntree){
    info1<- sprintf("?????? %d%%", round(j*100/rf$ntree)) 
    D<-calculate_max_depth(rf,j)
    interactions_frame_single<-min_depth_interactions_single(rf,j,colindex)
    rD<-calculate_rD(D,interactions_frame_single,j)
    rD<-cbind(interactions_frame_single[1:2],rD)
    rd_frame<-merge(rd_frame,rD,by=c('variable','root_variable'),all=T)
    setTkProgressBar(pb1, j*100/rf$ntree, sprintf("???? (%s)", info1),info1) 
  }
  close(pb1)
  
  rd_frame[is.na(rd_frame)]<-0
  rdlist[[label[i]]]<-rd_frame
  for (k in 1:nrow(rd_frame)){
    rd_frame[k,504]<-sum(rd_frame[k,4:503])/rd_frame[k,3]
  }
  r_frame<-rd_frame[c(1,2,504)]
  colnames(r_frame)<-c("variable" , "root_variable" ,"r")
  r_interaction[[label[i]]]<-r_frame
  #save(rd_frame, file = paste0("Rda/rd_frame_",label[i],".rda"))
  #save(r_frame,file = paste0("Rda/r_frame_",label[i],".rda"))
  setTkProgressBar(pb, i*100/length(label), sprintf("?ܽ??? (%s)", info),info) 
}
close(pb)
save(r_interaction,file='Rda/r-interaction.rda')
#----------------------------------------------
# Node and Edge files
#----------------------------------------------
load(file='Rda/r-interaction.rda')
load(file='Rda/multi-importance.rda')

type<-data.frame(label=c(colindex[-32],label),
                 type=c(rep('M',16),rep('A',6),rep('E',9),rep('y',15)),
                 color=c(rep('#98dbef',16),rep('#a4e192',6),rep('#ffc177',9),
                         rep('#ffb6d4',15)))

for (i in 1:length(label)){
  nodes<-data.frame(id=c(1:length(colindex)),label=c(colindex[-32],label[i]))
  nodes<-merge(nodes,type,all.x=T)
  nodes<-arrange(nodes,nodes['id'])
  write.csv(nodes,paste0('network/nodes_',label[i],'.csv'),row.names=FALSE,fileEncoding='UTF-8')
  
  r_frame<-r_interaction[[i]]
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
  im_frame<-mi[[i]]
  if (i <13){
    x_y<-data.frame(Source=im_frame$variable,
                  Target=c(rep(label[i],31)),
                  Weight=im_frame[4],
                  Type=c(rep('x-y',31)))
  } else {
    x_y<-data.frame(Source=im_frame$variable,
                    Target=c(rep(label[i],28)),
                    Weight=im_frame[4],
                    Type=c(rep('x-y',28)))
  }
  colnames(x_y)<-c('Source','Target','Weight','Type')
  edges<-rbind(edges,x_y)
  edges[3][edges[3]<=0]<-0
  for (j in 1:nrow(nodes[1])){
    edges[edges==nodes[j,1]]<-nodes[j,2]
  }
  write.csv(edges,paste0('network/edges',label[i],'.csv'),row.names=FALSE,fileEncoding='UTF-8')
}

#----------------------------------------------
# pdp analysis
#----------------------------------------------
pdplist<-list()
#pdpplot<-list()
for (i in 1:15){
  if (i<=12){
    data<-read.xlsx('imm/RF20210202.xlsx',i+1,header=TRUE)
    colnames(data)<-colindex
    train<-read.xlsx('imm/ss-index-all.xlsx',i)$X0
    data_train<-data[train,]
    data_test<-data[-train,]
    seed<-rep(1,12)
    seed[8]<-2
    seed[12]<-8
    set.seed(seed[i])
    rf<-randomForest(Index~.,data=data_train,importance=TRUE,proximity=T,ntree=500,mtry=5)
  } else {
    data<-read.xlsx('burden/clearance2021.xlsx',i-10,header=TRUE)[-1]
    colnames(data)<-colindex[-c(24,25)]
    train<-read.xlsx('burden/ss-index-all.xlsx',i-12)$X0
    data_train<-data[train,]
    data_test<-data[-train,]
    seed<-c(1,1,2)
    set.seed(seed[i-12])
    rf<-randomForest(Index~.,data=data_train,importance=TRUE,proximity=T,ntree=500,mtry=5)
  }
  
  inter_frame<-inter_list[[i]]
  j=1
  k=1
  subpdp<-list()
  subpdpplot<-list()
  while (j<=4){
    interpair<-inter_frame$interaction[k]
    v1<-strsplit(interpair,':')[[1]][1]
    v2<-strsplit(interpair,':')[[1]][2]
    k=k+1
    if (v1!=v2) {
      par<-pdp::partial(rf,pred.var = c(v1, v2), chull = TRUE, progress = "text")
      subpdp[[j]]<-par
      #subpdpplot[[j]]<-autoplot(par,contour = TRUE, legend.title = label[i])+
        # theme_bw()+
        # theme(legend.position=c(0.9,0.8))+
        # theme(panel.grid.major = element_blank(),panel.grid.minor = element_blank())+
        # theme(axis.line=element_line(color='black'),
        #       axis.ticks.length=unit(0.5,'line'))
      # ggsave(paste0('plot/dvpdp/',label[i],'-',v1,'-',v2,'.pdf'),width=5,height=5)
      j<-j+1
    } else {j<-j}
  }
  pdplist[[label[i]]]<-subpdp
  #pdpplot[[label[i]]]<-subpdpplot
}

save(pdplist,file='Rda/pdplist.rda')
save(pdpplot,file='Rda/pdpplot.rda')

for (i in 1:15){
  subpdp<-pdplist[[i]]
  for (j in 1:4){
    par<-subpdp[[j]]
    subpdpplot[[j]]<-local({
      par=par
      ggplot(par, aes(x = par[[1L]], y = par[[2L]],
                                     z = par[["yhat"]], fill = par[["yhat"]])) +
        geom_tile()+
        geom_contour(color = 'white')+
        viridis::scale_fill_viridis(name =label[i], option = 'D') +
        theme_bw()+
        xlab(colnames(par)[1])+
        ylab(colnames(par)[2])+
        theme(panel.grid.major = element_blank(),panel.grid.minor = element_blank())+
        theme(axis.line=element_line(color='black'),axis.text = element_text(size = rel(0.6)),
              axis.ticks.length=unit(0.3,'line'),axis.title = element_text(size = rel(0.6)))+
        theme(legend.key.size = unit(0.5,'line'),legend.title = element_text(size=rel(0.6)),
              legend.text = element_text(size=rel(0.5)),legend.position = c(0.9,0.8))
    ggsave(paste0('plot/dvpdp/',label[i],'-',
                  #colnames(par)[1],'-',colnames(par)[2],
                  #'.pdf'),width=5,height=5)
    })
  }
  pdpplot[[label[i]]]<-subpdpplot
}
  
for (i in 1:15){
  part1<-miplot[[i]]+fiplot[[i]]+plot_layout(width=c(2,4.5),height=c(2,2))+plot_annotation(tag_levels = 'a')
  part2<-pdpplot[[i]][[1]]+pdpplot[[i]][[2]]+pdpplot[[i]][[3]]+pdpplot[[i]][[4]]+
    plot_layout(ncol=2,width=c(2,2),height=c(2,2))
  part3<-mdplot[[i]]+part2+plot_layout(width=c(2.5,4.5))+
    plot_annotation(tag_levels = list(c('c','d','e','f','g')))

  ggsave(plot=part1,file=paste0('plot/SI/',label[i],'-1.pdf'),width=13,height=8)
  ggsave(plot=part3,file=paste0('plot/SI/',label[i],'-2.pdf'),width=13,height=7.7)
}

# layout <- '
# AAABBBB
# AAABBBB
# AAABBBB
# CCCDDEE
# CCCDDEE
# CCCFFGG
# CCCFFGG
# '
# wrap_plots(A = miplot[[1]], B = fiplot[[1]], c = mdplot[[1]], design = layout)

ic<-c(1,7,9,3,3,7,13,14,15)
v1c=c(rep('Zeta',4),rep('SSA',2),rep('D',3))
v2c=c(rep('SSA',4),rep('L',2),rep('L',3))
 
netpdpplot<-list()
for (j in 1:9){
  i<-ic[j]
  v1<-v1c[j]
  v2<-v2c[j]
  
  # if (i<=12){
  #   data<-read.xlsx('imm/RF20210202.xlsx',i+1,header=TRUE)
  #   colnames(data)<-colindex
  #   # train<-read.xlsx('imm/ss-index-all.xlsx',i)$X0
  #   # data_train<-data
  #   # data_test<-data[-train,]
  #   seed<-rep(1,12)
  #   seed[8]<-2
  #   seed[12]<-8
  #   set.seed(seed[i])
  #   rf<-randomForest(Index~.,data=data,importance=TRUE,proximity=T,ntree=500,mtry=5)
  # } else {
  #   data<-read.xlsx('burden/clearance2021.xlsx',i-10,header=TRUE)[-1]
  #   colnames(data)<-colindex[-c(2,24,25)]
  #   # train<-read.xlsx('burden/ss-index-all.xlsx',i-12)$X0
  #   # data_train<-data[train,]
  #   # data_test<-data[-train,]
  #   seed<-c(1,1,2)
  #   set.seed(seed[i-12])
  #   rf<-randomForest(Index~.,data=data,importance=TRUE,proximity=T,ntree=500,mtry=5)
  # }
  data<-datalist[[i]]
  rf<-randomForest(Index~.,data=data,importance=TRUE,proximity=T,ntree=500,mtry=5)
  par<-pdp::partial(rf,pred.var = c(v1, v2), chull = TRUE, progress = "text")
  if (j==6) { par<-par[-which(par$L>15000),] }
  
  netpdpplot[[j]]<-local({
    par=par
    ggplot(par, aes(x = par[[1L]], y = par[[2L]],
                  z = par[["yhat"]], fill = par[["yhat"]])) +
      geom_tile()+
      geom_contour(color = 'white')+
      viridis::scale_fill_viridis(name =label[i], option = 'D') +
      theme_bw()+
      xlab(colnames(par)[1])+
      ylab(colnames(par)[2])+
      theme(panel.grid.major = element_blank(),panel.grid.minor = element_blank())+
      theme(axis.line=element_line(color='black'),axis.ticks.length=unit(0.3,'line'))+
      theme(legend.position = c(0.9,0.8))
      #scale_x_continuous(limits = c(min(par$Zeta),60))
  #ggsave(paste0('plot/netpdp/',label[i],'-',v1,'-',v2,'.pdf'),width=5,height=5)
  })
}
for (i in 1:9){
netpdpplot[[i]]<-netpdpplot[[i]]+
  theme(panel.grid.major = element_blank(),panel.grid.minor = element_blank())+
  theme(axis.line=element_line(color='black'),axis.text = element_text(size = rel(0.6)),
        axis.ticks.length=unit(0.3,'line'),axis.title = element_text(size = rel(0.6)))+
  theme(legend.key.size = unit(0.5,'line'),legend.title = element_text(size=rel(0.6)),
        legend.text = element_text(size=rel(0.5)),legend.position = c(0.88,0.85))
}
netpdprda<-netpdpplot[[1]]+netpdpplot[[2]]+netpdpplot[[3]]+
            netpdpplot[[4]]+netpdpplot[[5]]+netpdpplot[[6]]+
            netpdpplot[[7]]+netpdpplot[[8]]+netpdpplot[[9]]+
            plot_layout(ncol=3,widths=rep(3,9),height=rep(3.9))
save(netpdpplot,file='Rda/netpdp.rda')
ggsave(file='plot/figure-pdp.pdf',width=9,height=9)



#####################################################################
#####                    permutation test                       #####
#####################################################################
setwd('D:/document/ywj/PC')
q2plot<-list()

for (i in 1:15){
  if (i<=12){permutation<-read.xlsx('imm/permutation.xlsx',i)
  } else {permutation<-read.xlsx('burden/permutation.xlsx',i-12)} 
  lm.model<-lm(permutation$q2~permutation$r2)
  coef<-lm.model$coefficients
  q2plot[[label[i]]]<-local({
    coef=coef
    permutation=permutation
    ggplot(data=permutation,aes(r2,q2))+
      geom_point(size=2.5,color='#00BFFF')+
      geom_abline(intercept=coef[1],slope=coef[2],size=1,color='black')+
      scale_x_continuous(limits=c(0,1))+
      scale_y_continuous(limits=c(-2,1))+
      labs(title=label[i])+
      coord_fixed(ratio=2/3)+
      theme_bw()+
      theme(axis.line=element_line(color='black'),
            axis.ticks.length=unit(0.3,'line'))+
      xlab(NULL)+
      ylab(NULL)+
      theme(panel.grid.major = element_blank(),panel.grid.minor = element_blank())+
      annotate('text', x=0.2,y=1,label=paste0('intercept: ',round(coef[1],2)))
  })
}

q2p<-q2plot[[1]]
for (i in 2:15){
  q2p<-q2p+q2plot[[i]]
}
q2p<-q2p+plot_layout(ncol=5)
print(q2p)
ggsave(filename = 'plot/permutation test.pdf',width=15,height=13.5)

#####################################################################
#####                    Correlation mat                        #####
#####################################################################
data<-read.xlsx('imm/RF20210202.xlsx',1,header=TRUE)
colnames(data)<-colindex[-32]
cormat<-data.frame(cor(data))

for (i in 1:12){
  data<-read.xlsx('imm/RF20210202.xlsx',i+1,header=TRUE)
  colnames(data)<-c(colindex[-32],label1[i])
  corsub<-cor(data)
  cormat[1:31,31+i]<-corsub[1:31,32]
  cormat[31+i,1:31]<-corsub[32,1:31]
}
for (i in 1:12){
  cormat[31+i,31+i]=1
}
colnames(cormat)<-c(colindex[-32],label1)
rownames(cormat)<-c(colindex[-32],label1)
write.xlsx(cormat,file = 'cormat-imm.xlsx',col.names = T,row.names = T)

burdenindex<-colindex[-c(2,24,25,32)]
data<-read.xlsx('burden/clearance2021.xlsx',6,header=TRUE)
colnames(data)<-colindex[-c(2,24,25,32)]
cormat1<-data.frame(cor(data))
for (i in 1:3){
  data<-read.xlsx('burden/clearance2021.xlsx',i+2,header=TRUE)[-1]
  colnames(data)<-c(burdenindex,label2[i])
  corsub<-cor(data)
  cormat1[1:28,28+i]<-corsub[1:28,29]
  cormat1[28+i,1:28]<-corsub[29,1:28]
}
for (i in 1:3){
  cormat[28+i,28+i]=1
}
colnames(cormat1)<-c(burdenindex,label2)
rownames(cormat1)<-c(burdenindex,label2)
write.xlsx(cormat1,file = 'cormat-burden.xlsx',col.names = T,row.names = T)

#####################################################################
#####                    Tabplot for data                       #####
#####################################################################
setwd('D:/?ű??ļ?/tabplot-master/R')
filelist<-dir(pattern = '.R')
for (ifile in filelist){
  source(paste0('D:/?ű??ļ?/tabplot-master/R/',ifile))
}
source('D:/?ű??ļ?/tabplot-master/build/createpalette.R')
source('D:/?ű??ļ?/bit_4.0.3/R/????????.R')

#setwd('??your path??/code')
library(xlsx)
#library(tabplot)
library(RColorBrewer)
options(fftempdir = 'D:/fftemp')
data<-read.xlsx('imm/imm-class for tab.xlsx',1)

feature<-data[1:18]
feature$Shape<-as.character(feature$Shape)
col<-colnames(feature)
feature$Shape[which(feature$Shape=='0')]<-'0-Particles'
feature$Shape[which(feature$Shape=='1')]<-'1-D'
feature$Shape[which(feature$Shape=='2')]<-'2-D'
feature$Animal[which(feature$Animal=='Mouse')]<-'Mice'
color_enps<-c(read.xlsx('??????ɫ.xlsx',1,header=FALSE)$X1)
col_enps<-c()
f_color<-brewer.pal(12,"Paired")
pal<-colorRampPalette(f_color)
f_color<-pal(19)
for (i in 1:6){
  col_enps<-c(col_enps,color_enps[c(seq(i,60,by=6))])
}

set.seed(2)
tabplot::tableplot(feature,sortcol=Types,decreasing =FALSE,
                   nBins=100,
                   select=c(Types,LogD,LogL,Shape,Functionalization,Zeta,SSA,
                            Animal,Gender,Weight,Age),
                   pals=list(Types=brewer.pal(8,'Set3'),
                             Shape=brewer.pal(9,"Purples")[c(3,5,7)],
                             Functionalization=f_color)
)
feature$Weight<-log10(feature$Weight)
tabplot::tableplot(feature,sortCol = Types,decreasing=F,
                   select=c(Types,Animal,Gender,Weight,Age),
                   pals=list(Types=brewer.pal(8,'Set3')))

tabplot::tableplot(feature,sortCol = Types,decreasing=F,
                   select=c(Types,Method,Duration,Frequency,Recovery,Dose),
                   pals=list(Types=brewer.pal(8,'Set3')))

dev.new()
set.seed(2)
tableplot(feature,sortcol=LogL,decreasing =T,
                   select=c(LogL,LogD,Types,Shape,Functionalization,Zeta,SSA,
                            Animal,Gender,Weight,Age,
                            Method,Duration,Frequency,Recovery,Dose),
                   pals=list(Types=brewer.pal(8,'Set3'),
                             Shape=brewer.pal(9,"Purples")[c(3,5,7)],
                             Functionalization=f_color,
                             Animal=brewer.pal(11,"Spectral")[c(4,8)],
                             Gender=brewer.pal(4,'Pastel2'),
                             Method=brewer.pal(6,'Pastel1')),
                   scales='lin'
)
ggsave(filename = 'plot/tab-imm.pdf',width=19.2,height = 5.4)

#####################################################################
#####                    Il-1b validation                       #####
#####################################################################
setwd('D:/document/ywj/PC')
data<-read.xlsx('imm/il1b.xlsx',2)

rmse<-read.xlsx('imm/all-rf-cor.xlsx',6)[1:10,5]
rmse_test.mean<-mean(rmse)

ggplot(data=data,aes(x=Observe,y=Predict,color=type,
                     shape=type,fill=type))+
  geom_abline(intercept=0,slope=1,size=1,color='black')+
  geom_abline(intercept=rmse_test.mean,slope=1,size=0.5,linetype="dashed",color='black')+
  geom_abline(intercept=0-rmse_test.mean,slope=1,size=0.5,linetype="dashed",color='black')+
  geom_point(size=1.5)+
  scale_x_continuous(limits=c(-0.5,1))+
  scale_y_continuous(limits=c(-0.5,1))+
  coord_fixed(ratio=1)+
  scale_shape_manual(name="Data Source",
                     values=c(16,17,18))+
  scale_colour_manual(name="Data Source",
                      values=c("#7ae07a","#e07b7b",'#7b7be0'))+
  theme_bw()+
  theme(axis.line=element_line(color='black'),
        axis.ticks.length=unit(0.5,'line'))+
  #axis.text = element_blank())+
  xlab(NULL)+
  ylab(NULL)+
  theme(panel.grid.major = element_blank(),panel.grid.minor = element_blank())

ggsave('plot/val-il-1b.pdf',width=3.5,height=3.5)


#####################################################################
#####                    Feature shuffle                        #####
#####################################################################
setwd('D:/document/ywj/PC')
datalist<-list()
for (i in 1:15){
  if (i<13){
    data<-read.xlsx('imm/RF20210202.xlsx',i+1,header=TRUE)
    colnames(data)<-colindex
  } else {
    data<-read.xlsx('burden/clearance2021.xlsx',i-10,header=TRUE)[-1]
    colnames(data)<-colindex[-c(2,24,25)]
  }
  datalist[[label[i]]]<-data
}
save(datalist,file='Rda/data.rda')

seed<-rep(1,15)
seed[8]<-2
seed[12]<-8
seed[15]<-2
shuffle_list<-list()
for (i in 1:15){
  set.seed(seed[i])
  data<-datalist[[i]]
  rf<-randomForest(Index~.,data=data,importance=TRUE,proximity=T,ntree=500,mtry=5)
  pre_train<-predict(rf)
  rmse_shuffle<-rmse(pre_train,data$Index)
  
  for (j in 1:(ncol(data)-1)){
    of<-data[,j]
    set.seed(j)
    sf<-sample(of,size=length(of),replace = FALSE)
    data[j]<-sf
    rf<-randomForest(Index~.,data=data,importance=TRUE,proximity=T,ntree=500,mtry=5)
    pre_train<-predict(rf)
    rmse_shuffle<-append(rmse_shuffle,rmse(pre_train,data$Index))
  }
  shuffle_frame<-data.frame(n=c(0:(ncol(data)-1)),rmse=rmse_shuffle,increace=NA)
  for (j in 1:nrow(shuffle_frame)){
    shuffle_frame[j,3]<-(shuffle_frame[j,2]-shuffle_frame[1,2])/shuffle_frame[1,2]*100
  }
  shuffle_list[[label[i]]]<-shuffle_frame
}


shuffleplot_r<-list()
for (i in 1:15){
  shuffle_frame<-shuffle_list[[i]]
  colnames(shuffle_frame)<-c('n_feature','rmse','increase')
  shuffleplot_r[[label[i]]]<-local({
    shuffle_frame=shuffle_frame
    ggplot(shuffle_frame,aes(n_feature,increase))+
      geom_line(color='#00CED1')+
      geom_point(color='#00CED1')+
      theme_bw()+
      labs(title=label[i])+
      theme(axis.line=element_line(color='black'))+
      theme(panel.grid.major = element_blank(),panel.grid.minor = element_blank())+
      coord_fixed(ratio=nrow(shuffle_frame)/
                    (max(shuffle_frame$increase)-min(shuffle_frame$increase)))+
      theme(legend.position="none")
  })
}
save(shuffleplot_r,file = 'Rda/shuffleplot_r.rda')
p<-shuffleplot_r[[1]]
for (i in 2:15){
  p<-p+shuffleplot_r[[i]]
}
p<-p+plot_layout(ncol=3)
ggsave('plot/feature-shuffle-r.pdf',width = 9,height=15)


# shuffleplot<-list()
# for (i in 1:15){
#   shuffle<-read.xlsx('feature-shuffle.xlsx',i)
#   train<-shuffle$rmse_train
#   test<-shuffle$rmse_test
#   n<-length(shuffle$rmse_train)/10
#   shuffle_mean<-data.frame(n_feature=c(1:n,1:n),rmse=NA,increase_rmse=NA,type=rep(c('train','test'),each=n))
#   for (j in 1:n){
#     trainj<-c()
#     testj<-c()
#     for (k in 1:10) {
#       trainj<-c(trainj,train[(k-1)*n+j])
#       testj<-c(testj,test[(k-1)*n+j])
#     }
#     shuffle_mean[j,2]<-mean(trainj)
#     shuffle_mean[j,3]<-(shuffle_mean[j,2]-shuffle_mean[1,2])/shuffle_mean[1,2]*100
#     shuffle_mean[j+n,2]<-mean(testj)
#     shuffle_mean[j+n,3]<-(shuffle_mean[j+n,2]-shuffle_mean[1+n,2])/shuffle_mean[1+n,2]*100
#   }
#   
#   shuffleplot[[label[i]]]<-local({
#     shuffle_mean=shuffle_mean
#     ggplot(shuffle_mean,aes(n_feature,increase_rmse,color=type))+
#       geom_line()+
#       geom_point()+
#       theme_bw()+
#       labs(title=label[i])+
#       theme(axis.line=element_line(color='black'))+
#       theme(panel.grid.major = element_blank(),panel.grid.minor = element_blank())+
#       coord_fixed(ratio=n/(max(shuffle_mean$increase_rmse)-min(shuffle_mean$increase_rmse)))+
#       theme(legend.position="none")
#   })
# }
# save(shuffleplot,file = 'Rda/shuffleplot.rda')
# p<-shuffleplot[[1]]
# for (i in 2:15){
#   p<-p+shuffleplot[[i]]
# }
# p<-p+plot_layout(ncol=3)
# ggsave('plot/feature-shuffle.pdf',width = 9,height=15)
