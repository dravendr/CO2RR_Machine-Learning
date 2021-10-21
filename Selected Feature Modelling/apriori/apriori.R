library(arules)
  files_change<-read.transactions("database_filled_TOF4.csv", format="basket", sep=",")
summary(files_change)
rules<-apriori(files_change,parameter=list(support=0.03,confidence=0.2,minlen=2,maxlen=2))
write(rules, file="result_TOF4.csv", sep=",", quote=TRUE, row.names=FALSE)

