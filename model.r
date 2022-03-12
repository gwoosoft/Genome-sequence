
Formatting Data:

I used one-hot-encoder for DNA-genetic sequence dataset.

library(e1071)
library(caret)
library(caTools)
library(ROCR)
require(nnet)
require(randomForest)
require(parallel)
library(doParallel) ## for parallel computing
require(gbm)
require(ROCR)
require(xgboost)
require(Matrix)

cl <- makePSOCKcluster(5) ##for parallel processing
registerDoParallel(cl)
sdf<-read.csv('/home/2021/nyu/fall/gl1858/splice.csv')
my_cols <- c(names(sdf[2:61])) ## take instance column names
mdf <-sdf[my_cols]
head(mdf)
output <- sdf[c(62)]
head(output)

onehotencoder <- function(df_orig) {
df<-cbind(df_orig)
df_clmtyp<-data.frame(clmtyp=sapply(df,class))
df_col_typ<-data.frame(clmnm=colnames(df),clmtyp=df_clmtyp$clmtyp)
for (rownm in 1:nrow(df_col_typ)) {
if (df_col_typ[rownm,"clmtyp"]=="factor") {
clmn_obj<-df[toString(df_col_typ[rownm,"clmnm"])]
dummy_matx<-data.frame(model.matrix( ~.-1, data = clmn_obj))
dummy_matx<-dummy_matx[,c(1,3:ncol(dummy_matx))]
df[toString(df_col_typ[rownm,"clmnm"])]<-NULL
df<-cbind(df,dummy_matx)
df[toString(df_col_typ[rownm,"clmnm"])]<-NULL
} }
return(df)
}


cdf <- onehotencoder(mdf)
data <-cbind(cdf,output)
#head(data)
set.seed(43)
randomized=data[sample(1:nrow(data),nrow(data)),] # Shuffle
tridx=sample(1:nrow(data),0.7*nrow(data),replace=F) #Get indices for 70% of the total number of samples
trdf=randomized[tridx,] # Define training data set
tstdf=randomized[-tridx,] # Define testing data set
table(data$Class)/nrow(data) # Check if class distribution is similar
table(trdf$Class)/nrow(trdf)
table(tstdf$Class)/nrow(tstdf)

A. Random Forest
I choose random forest as I used decision tree for individual classfier and I think it is important to see how pruned
decision tree, which is random forest can be beneficial to balance out bias and variance problem. I choose
500 trees to run the model.

require(randomForest)
trdf_RF=trdf #Take train dataset
trdf_RF$Class=as.factor(trdf_RF$Class) #Take Y from the train dataset
rf_model=randomForest(Class~.,trdf_RF, ntree=500)
ry_pred = predict(rf_model, newdata = tstdf[-228])
cm = table(tstdf[, 228], ry_pred)
cfm<-confusionMatrix(cm)
cfm ##to plot confusion matrix

gbm_model<-gbm(Class~., data=trdf,
distribution="multinomial" ,n.trees=500,shrinkage=0.01,interaction.depth=3, n.minobsinnode=10,
verbose=T, keep.data=T)
gbm_predict<-predict(gbm_model, tstdf[,-c(228)], gbm_model$n.trees, type="response")
gbm_predicted<-round(gbm_predict)
labels = colnames(gbm_predicted)[apply(gbm_predicted, 1, which.max)]
#gbm_prediction<-prediction(labels,tstdf$Class)
p.gbm_predict=apply(gbm_predicted, 1, which.max)
result = data.frame(tstdf$Class, labels)
#print(result)
cm = confusionMatrix(tstdf$Class, as.factor(labels))
print(cm)

##Xgboost
#To create matrix for Xgboost model
train_x = data.matrix(trdf[,-228])
train_y = trdf[,228]
test_x = data.matrix(tstdf[,-228])
test_y = tstdf[,228]
xgb_train = xgb.DMatrix(data=train_x, label=train_y)
xgb_test = xgb.DMatrix(data=test_x, label=test_y)
xgbc = xgboost(data=xgb_train, max.depth=3, nrounds=50)
print(xgbc)

pred = predict(xgbc, xgb_test)
#print(pred)
pred[(pred>3)] = 3 ##as the model prediction gives probabilities we have to convert into factor type to
##run in a confusion matrix
pred_y = as.factor((levels(test_y))[round(pred)])
cm = confusionMatrix(test_y, pred_y)
print(cm)

stackeddf<-data.frame(actual=tstdf$Class,
rfpred=ry_pred, #random forest
gbmpred=as.factor(labels), ##gbm_pred
xgbpred=pred_y) ##xgb_pred
head(stackeddf)
tail(stackeddf)

data_new <- sapply(stackeddf, unclass) # Convert categorical variables
#data_new
stacked_mean<-round(unlist(apply(data_new [,2 :4],1,mean))) #rounded so that it does not have to
choose value between and it is more probabilistic.
data_new<-cbind(data_new,stacked=stacked_mean)
testing<-as.data.frame(data_new) #to convert it back into
(stbl<-table(testing$actual,testing$stacked))
(scfm<-caret::confusionMatrix(stbl))


accuracy<-function(xt)sum(diag(xt))/sum(xt)
xgbtbl <-table(tstdf$Class,pred_y)
gbmtbl <-table(tstdf$Class,as.factor(labels))
rfmtbl <-table(tstdf$Class,ry_pred)
gbm_acc<-accuracy(gbmtbl)
rfm_acc<-accuracy(rfmtbl)
xgb_acc<-accuracy(xgbtbl)
stk_acc<-accuracy(stbl)
detach("package:caret", unload=TRUE) ##To avoid conflict with pRoc library
require(pRoc) ##To call the multiclass ROC

gy_pred<-as.factor(labels)
##GBM
mqa<-multiclass.roc(response=tstdf$Class, predictor=factor(gy_pred, ordered=TRUE), plot=TRUE
,print.auc=TRUE, main='ROC Curves(GBM)')

##RFM
rqa<-multiclass.roc(response=tstdf$Class, predictor=factor(ry_pred, ordered=TRUE), plot=TRUE
,print.auc=TRUE, main='ROC Curves(Rforest)')

##Xboost
xqa<-multiclass.roc(response=tstdf$Class, predictor=factor(pred_y, ordered=TRUE), plot=TRUE
,print.auc=TRUE, main='ROC Curves(Xboost)')

##Stacking
sqa<-multiclass.roc(response=tstdf$Class, predictor=factor(testing$stacked, ordered=TRUE), plot=TRUE
,print.auc=TRUE, main='ROC Curves(Stacked(RF,GBM,XGB))')

To obtain accuracy for each model
gbauc<-auc(mqa)
rtauc<-auc(rqa)
xbauc<-auc(xqa)
stauc<-auc(sqa)
##To create a table for accuracy and AUC
perfdf<-data.frame(Algo=c("gbm","rfm","xgb","Stk"), Acc=c(gbm_acc, rfm_acc,xgb_acc,stk_acc),
AUC=c(gbauc,rtauc,xbauc,stauc))
print(perfdf)

Naïve Bayes VS The methods to reduce bias or variance
nbmodel <- naiveBayes(Class ~., data=trdf)
nbpred <- predict(nbmodel, newdata=tstdf, type="class")
NBA<-multiclass.roc(response=tstdf$Class, predictor=factor(nbpred, ordered=TRUE),
plot=TRUE,lwd=3,
legacy.axes=TRUE,print.auc=TRUE, main='ROC Curves(Nayive Bayes)')

nbtbl <-table(tstdf$Class,nbpred)
nb_acc<-accuracy(nbtbl )
nbauc<-auc(NBA)
perfdf<-data.frame(Algo=c("gbm","rfm","xgb","Stk","NaiB"), Acc=c(gbm_acc,
rfm_acc,xgb_acc,stk_acc,nb_acc), AUC=c(gbauc,rtauc,xbauc,stauc,nbauc))
print(perfdf)
