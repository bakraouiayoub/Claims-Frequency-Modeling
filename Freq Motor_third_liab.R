library(MASS)
library(CASdatasets)
library(ggplot2)
library(keras)
library(rpart)
library(data.table)
library(plyr)
library(rpart.plot)
library(tictoc)
data("freMTPL2freq")
dta_freq <- freMTPL2freq
rm(freMTPL2freq)

dta_freq$VehGas <- factor(dta_freq$VehGas)
                           # Modifications to the variable region for readability in the outputs later on ----

region_orig_lev <- levels(dta_freq$Region) 
region_new_lev <- c("R1","R2","R3","R4","R5","R6","R7","R8","R9","R10","R11","R12","R13","R14","R15",
                    "R16","R17","R18","R19","R20","R21")
modified_levels <- cbind(original_levels=region_orig_lev,modified_levels=region_new_lev) # The original levels of the
                                                                                         # variable region and their 
                                                                                         # modified counterparts

levels(dta_freq$Region) <- region_new_lev
rm(region_orig_lev,region_new_lev) # cleaning the environment





dta_freq$Region <- relevel(dta_freq$Region,ref = "R7")    # The region with the highest exposure is chosen as a reference
dta_freq$ClaimNb <- pmin(dta_freq$ClaimNb, 4) # Setting policies with more than 4 claims to having exactly 4 claims.
dta_freq$Exposure <- pmin(dta_freq$Exposure, 1) # correct for unreasonable observations (that might be data error)

# Descriptive statistics of the entire dataset------------------------------
Exp_by_ClmNb <- with(dta_freq,tapply(Exposure,ClaimNb,sum)) # Exposure by the number of claims

hist(dta_freq$Exposure,xlab = "Exposures",ylab = "Number of Policies",main = "Histogram of the exposures (678013 policies)")
Exp_by_Area <-  with(dta_freq,tapply(Exposure,Area,sum)) # Exposure by Area code
Exp_by_vehpow <-  with(dta_freq,tapply(Exposure,VehPower,sum)) # Exposure by Vehicle Power group

# Frequency by Area code and Vehicle Power group
Freq_by_Area <- with(dta_freq,tapply(ClaimNb,Area,sum))/with(dta_freq,tapply(Exposure,Area,sum))*100 
Freq_by_vehpow <- with(dta_freq,tapply(ClaimNb,VehPower,sum))/with(dta_freq,tapply(Exposure,VehPower,sum))*100  

rm(Exp_by_Area,Exp_by_ClmNb,Exp_by_vehpow,Freq_by_Area,Freq_by_vehpow) # Clean the global environment


# GLM as benchmark


       # Training and Test Dataset -------------

set.seed(254451710)
u <- runif(nrow(dta_freq),min = 0,max = 1)
dta_freq$train <- u < 0.8                 # Training dataset 
dta_freq$test <- !(dta_freq$train)        # Test dataset
               
rm(u)




dta_NN <- dta_freq   # we copy our database into a new one for the Neural Network modeling since we'll have to
                     # do feature pre-processing (scaling and modifications) before applying the NN.

       # Pre-processing of predictors space for GLM modeling  -------------------


dta_freq$AreaGLM <- as.integer(dta_freq$Area)
dta_freq$VehPowerGLM <- as.factor(pmin(dta_freq$VehPower,9))
VehAgeGLM <- cbind(c (0:110) , c(1, rep (2 ,10) , rep (3 ,100)))
dta_freq$VehAgeGLM <- as.factor ( VehAgeGLM [ dta_freq$VehAge +1 ,2])       # Age of the vehicle as a categorical variable
dta_freq$VehAgeGLM <- relevel(dta_freq$VehAgeGLM,ref = "2")                       
rm(VehAgeGLM)

dta_freq$DrivAgeGLM <- ""
dta_freq$DrivAgeGLM[dta_freq$DrivAge %in% 18:20] <- "1"
dta_freq$DrivAgeGLM[dta_freq$DrivAge %in% 21:25] <- "2"
dta_freq$DrivAgeGLM[dta_freq$DrivAge %in% 26:30] <- "3"
dta_freq$DrivAgeGLM[dta_freq$DrivAge %in% 31:40] <- "4"                    # Age of the driver as a categorical variable
dta_freq$DrivAgeGLM[dta_freq$DrivAge %in% 41:50] <- "5"
dta_freq$DrivAgeGLM[dta_freq$DrivAge %in% 51:70] <- "6"
dta_freq$DrivAgeGLM[dta_freq$DrivAge > 70] <- "7"

dta_freq$DrivAgeGLM <- as.factor(dta_freq$DrivAgeGLM)
dta_freq$DrivAgeGLM <- relevel(dta_freq$DrivAgeGLM,ref = "5")

dta_freq$BonusMalusGLM <- as.integer(pmin(dta_freq$BonusMalus,150))
dta_freq$DensityGLM <- as.numeric(log(dta_freq$Density))



     # Model GLM1, GLM2, GLM3 -------------

res.vars <- "ClaimNb"                   # Response variable
mis.vars <- c("IDpol","Exposure")       # Miscelaneous variables
vars <- c("VehPowerGLM","VehAgeGLM","DrivAgeGLM","BonusMalusGLM","VehBrand","VehGas","DensityGLM","Region","AreaGLM")

d.glm1 <- glm(paste(res.vars,paste(vars,sep = "",collapse = "+"),sep = "~"),data = dta_freq,
              subset = train,family = poisson(link = log),offset = log(Exposure))
summary(d.glm1)
#anova(d.glm1,test = "Chisq")

d.glm2 <- update(d.glm1,.~. - AreaGLM) # We drop the predictor AreaGLM
d.glm3 <- update(d.glm1,.~. - VehBrand - AreaGLM) # We drop the predictors AreaGLM and VehBrand

# Loss function: Average Poisson deviance statistic
Dev_Poisson <- function(model)
{
train_dat <- subset(dta_freq,subset = train)
test_dat <- subset(dta_freq,subset = test)
N_hat_train <- fitted(model)
N_hat_test <- predict(model,newdata=test_dat,type="response")
average_in_sample <- 2*(sum(N_hat_train)-sum(train_dat$ClaimNb)+sum(log((train_dat$ClaimNb/N_hat_train)^train_dat$ClaimNb)))/nrow(train_dat)
average_out_of_sample <- 2*(sum(N_hat_test)-sum(test_dat$ClaimNb)+sum(log((test_dat$ClaimNb/N_hat_test)^test_dat$ClaimNb)))/nrow(test_dat)
return(c(model$aic,average_in_sample*100,average_out_of_sample*100))
}
  
Table_5 <- data.frame(rbind(Dev_Poisson(d.glm1),Dev_Poisson(d.glm2),Dev_Poisson(d.glm3)))
rownames(Table_5) <- c("Model GLM1","Model GLM2","Model GLM3")  # The model GLM2 is slightly better than GLM1 in terms of
colnames(Table_5) <- c("AIC","Training Loss %", "Test Loss %")  # the AIC but has a worse test error. Model GLM3  
Table_5                                                         # is not competitive and the component VehBrand is needed. 
              # Remark: the test error might be less than the training error if the test dataset is very small.
 
rm(mis.vars,res.vars,vars,Dev_Poisson) # cleaning the environment







     # Regression Trees-------------------------

#vars_tree1 <- c("Area","VehPower" ,"VehAge", "DrivAge", "BonusMalus","VehBrand", "VehGas","Density","Region")

leaf_size <- function(arbre){return(sum(arbre$frame$var == "<leaf>"))} # function that returns the number of leafs.

tree1 <- rpart(cbind(Exposure,ClaimNb)~ Area+VehPower+VehAge+DrivAge+BonusMalus+VehBrand+VehGas+Density+Region ,
         data = dta_freq,subset = train, method = "poisson",control = rpart.control(xval = 1,minbucket = 10000,cp=0.0005))
rpart.plot(tree1,yesno=0)

# Cross-validation for the parameter cp .
set.seed(123451710)
N <- nrow(dta_freq)
dta_freq$rnd <- runif(N,min = 0,max = 1)
rm(N)

# Function that computes the mean M-fold cross validation error and its sd using parameter cp
CV_error <- function(M,cp_){   
cross_error <- rep(0,M) # Vector that contains the M cross_validation errors
bk <- seq(0,1,length=M+1) # vector that helps us to select the training data in cross validation.
for (i in 2:length(bk)) {
  n_valid <- nrow(dta_freq[dta_freq$train &((dta_freq$rnd > bk[i-1]) & (dta_freq$rnd <= bk[i])), ]) 
  tree_valid <- rpart(cbind(Exposure,ClaimNb)~ Area+VehPower+VehAge+DrivAge+BonusMalus+VehBrand+VehGas+Density+Region ,
                    data = dta_freq,subset = train &((rnd <= bk[i-1]) | (rnd > bk[i])) , method = "poisson",
                        control = rpart.control(xval = 1,minbucket = 10000,cp=cp_))
  N_hat <- predict(tree_valid,newdata=dta_freq[dta_freq$train &((dta_freq$rnd > bk[i-1]) & (dta_freq$rnd <= bk[i])),],
                type="vector") *dta_freq[dta_freq$train &((dta_freq$rnd > bk[i-1]) & (dta_freq$rnd <= bk[i])),"Exposure"]
  N_obs <- dta_freq[dta_freq$train &((dta_freq$rnd > bk[i-1]) & (dta_freq$rnd <= bk[i])),"ClaimNb"]
  cross_error[i-1] <- 2*(sum(N_hat)-sum(N_obs)+sum(log((N_obs/N_hat)^N_obs)))/n_valid
}
return(rbind(error_mean=mean(cross_error)*100,error_sd=sd(cross_error)/sqrt(M)))
}

cp_vector <- c(10^(-5),10^(-4.4),10^(-4.3),10^(-4.2),10^(-4.1),seq(10^(-4),10^(-3.5),0.00005),seq(10^(-3.5),10^(-3),0.0002),
            seq(10^(-2.98),10^(-2.85),0.00005),seq(10^(-2.85),10^(-2),0.0005) ,10^(-1.75),10^(-1.5),10^(-1.25),10^(-1))
tic()
cross_error_bycp <- mapply(CV_error,10,cp_vector)
toc()

mincv_i <- which.min(cross_error_bycp[1,])
cp_opt_mincv <- cp_vector[mincv_i] # optimal cp with minimum cross validation error rule
placement_sdcv <- which(cross_error_bycp[1,]>=((cross_error_bycp[1,mincv_i]/100-cross_error_bycp[2,mincv_i])*100) & # placement of all cp 
                         cross_error_bycp[1,]<=((cross_error_bycp[1,mincv_i]/100+cross_error_bycp[2,mincv_i])*100)) # within 1 sd
cp_opt_sdcv <- cp_vector[placement_sdcv[length(placement_sdcv)]]

tree2 <- rpart(cbind(Exposure,ClaimNb)~ Area+VehPower+VehAge+DrivAge+BonusMalus+VehBrand+VehGas+Density+Region ,
               data = dta_freq,subset = train, method = "poisson",
               control = rpart.control(xval = 1,minbucket = 10000,cp=cp_opt_mincv))

tree3 <- prune(tree2,cp=cp_opt_sdcv)
rm(mincv_i,placement_sdcv)

# Function that computes the training and the test loss given a tree 
arbre_dev_poiss <- function(arbre){
train_dat <- subset(dta_freq,subset = train)
test_dat <- subset(dta_freq,subset = test)  
n_train <- nrow(train_dat)  
n_test <- nrow(test_dat)
N_hat_train <- predict(arbre,newdata=train_dat,type="vector")*train_dat[,"Exposure"]
N_hat_test <- predict(arbre,newdata=test_dat,type="vector")*test_dat[,"Exposure"]
N_obs_train <- train_dat[,"ClaimNb"]
N_obs_test <- test_dat[,"ClaimNb"]
average_train_err <- 2*(sum(N_hat_train)-sum(N_obs_train)+sum(log((N_obs_train/N_hat_train)^N_obs_train)))/n_train
average_test_err <- 2*(sum(N_hat_test)-sum(N_obs_test)+sum(log((N_obs_test/N_hat_test)^N_obs_test)))/n_test
return(c(average_train_err*100,average_test_err*100))

}


Table_6 <- data.frame(rbind(arbre_dev_poiss(tree1),arbre_dev_poiss(tree2),arbre_dev_poiss(tree3)))
rownames(Table_6) <- c("Model tree1 (13 leaves)","Model tree2 (min. CV rule, 27 leaves)",
                       "Model tree3 (1 sd-rule, 7 leaves)")  
colnames(Table_6) <- c("Training Loss %", "Test Loss %")  
# In-sample and out-of-sample losses of the three regression trees and of the GLM approach
rbind(Table_6,Table_5[1,-1])

rm(cross_error_bycp,arbre_dev_poiss,CV_error)


    # Poisson deviance tree boosting machine ----------------------------- 

# Function that returns the training and the test error for a poisson tree boosting machine
Boost_Poiss <- function(J0,M,alpha){
 
   # J0 is the maximum depth of the trees (corresponding to the max distance between the root and all the leaves)
  # M is the number of iterations 
  # alpha is the Shrinkage constant
  
  # initialization: we start with weights as normal exposures
  dta_freq$Pweights <- dta_freq$Exposure
  
  for (i in 1:M) {
    Pboost_fit <- rpart(cbind(Pweights,ClaimNb)~ Area+VehPower+VehAge+DrivAge+BonusMalus+VehBrand+VehGas+Density+Region ,
                        data = dta_freq,subset = train, method = "poisson",
                        control = rpart.control(maxdepth = J0,maxsurrogate = 0,xval = 1,minbucket = 10000,cp=0.00001))
    dta_freq$Pweights[dta_freq$train] <-  dta_freq$Pweights[dta_freq$train]*(predict(Pboost_fit,type="vector")^alpha)
    dta_freq$Pweights[dta_freq$test] <-  dta_freq$Pweights[dta_freq$test]*
      (predict(Pboost_fit,newdata=dta_freq[dta_freq$test,],type="vector")^alpha)
    
  }
  Pboost_fit <- rpart(cbind(Pweights,ClaimNb)~ Area+VehPower+VehAge+DrivAge+BonusMalus+VehBrand+VehGas+Density+Region ,
                      data = dta_freq,subset = train, method = "poisson",
                      control = rpart.control(maxdepth = J0,maxsurrogate = 0,xval = 1,minbucket = 10000,cp=0.00001))
 
  train_dat <- subset(dta_freq,subset = train)
  test_dat <- subset(dta_freq,subset = test)  
  n_train <- nrow(train_dat)  
  n_test <- nrow(test_dat)
  N_hat_train <- (predict(Pboost_fit,newdata=train_dat,type="vector")^alpha)*train_dat[,"Pweights"]
  N_hat_test <- (predict(Pboost_fit,newdata=test_dat,type="vector")^alpha)*test_dat[,"Pweights"]
  N_obs_train <- train_dat[,"ClaimNb"]
  N_obs_test <- test_dat[,"ClaimNb"]
  average_train_err <- 2*(sum(N_hat_train)-sum(N_obs_train)+sum(log((N_obs_train/N_hat_train)^N_obs_train)))/n_train
  average_test_err <- 2*(sum(N_hat_test)-sum(N_obs_test)+sum(log((N_obs_test/N_hat_test)^N_obs_test)))/n_test
  return(c(average_train_err*100,average_test_err*100))   
  
}

Table_7 <- data.frame(rbind(Boost_Poiss(1,30,1),Boost_Poiss(2,50,1),Boost_Poiss(3,50,1)))
rownames(Table_7) <- c("Model Poisson boosting (J=1,iter=30)","Model Poisson boosting (J=2,iter=50)",
                       "Model Poisson boosting (J=3,iter=50)")  
colnames(Table_7) <- c("Training Loss %", "Test Loss %")  
# Training and test errors of the three poisson boosting models (no shrinkage) and the best tree and glm models
rbind(Table_7,Table_6[2,],Table_5[1,-1])

# Function that returns the training and the test error for a GLM poisson boosting machine
Boost_GLM <- function(J0,M,alpha){
  
  # J0 is the maximum depth of the trees (corresponding to the max distance between the root and all the leaves)
  # M is the number of iterations 
  # alpha is the Shrinkage constant
  
  # initialization: we start with weights as normal exposures
  dta_freq$Pweights <- dta_freq$Exposure
  dta_freq$Pweights[dta_freq$train] <-  (dta_freq$Pweights[dta_freq$train]^(1-alpha))*
                                           (predict(d.glm1,type="response")^alpha)
  dta_freq$Pweights[dta_freq$test] <-  (dta_freq$Pweights[dta_freq$test]^(1-alpha))*
    (predict(d.glm1,newdata=dta_freq[dta_freq$test,],type="response")^alpha)
    
  for (i in 2:M) {
    Pboost_fit <- rpart(cbind(Pweights,ClaimNb)~ Area+VehPower+VehAge+DrivAge+BonusMalus+VehBrand+VehGas+Density+Region ,
                        data = dta_freq,subset = train, method = "poisson",
                        control = rpart.control(maxdepth = J0,maxsurrogate = 0,xval = 1,minbucket = 10000,cp=0.00001))
    dta_freq$Pweights[dta_freq$train] <-  dta_freq$Pweights[dta_freq$train]*(predict(Pboost_fit,type="vector")^alpha)
    dta_freq$Pweights[dta_freq$test] <-  dta_freq$Pweights[dta_freq$test]*
      (predict(Pboost_fit,newdata=dta_freq[dta_freq$test,],type="vector")^alpha)
    
  }
  Pboost_fit <- rpart(cbind(Pweights,ClaimNb)~ Area+VehPower+VehAge+DrivAge+BonusMalus+VehBrand+VehGas+Density+Region ,
                      data = dta_freq,subset = train, method = "poisson",
                      control = rpart.control(maxdepth = J0,maxsurrogate = 0,xval = 1,minbucket = 10000,cp=0.00001))
  
  train_dat <- subset(dta_freq,subset = train)
  test_dat <- subset(dta_freq,subset = test)  
  n_train <- nrow(train_dat)  
  n_test <- nrow(test_dat)
  N_hat_train <- (predict(Pboost_fit,newdata=train_dat,type="vector")^alpha)*train_dat[,"Pweights"]
  N_hat_test <- (predict(Pboost_fit,newdata=test_dat,type="vector")^alpha)*test_dat[,"Pweights"]
  N_obs_train <- train_dat[,"ClaimNb"]
  N_obs_test <- test_dat[,"ClaimNb"]
  average_train_err <- 2*(sum(N_hat_train)-sum(N_obs_train)+sum(log((N_obs_train/N_hat_train)^N_obs_train)))/n_train
  average_test_err <- 2*(sum(N_hat_test)-sum(N_obs_test)+sum(log((N_obs_test/N_hat_test)^N_obs_test)))/n_test
  return(c(average_train_err*100,average_test_err*100))   
  
}

Table_9 <- data.frame(rbind(Boost_GLM(3,50,1)))
rownames(Table_9) <- "Model GLMBoost"  
colnames(Table_9) <- c("Training Loss %", "Test Loss %")  
# Training and test errors of the GLM boosting model (no shrinkage) and the best tree boosting and trees.
rbind(Table_9,Table_7[3,],Table_6[2,],Table_5[1,-1])




   #  Neural Network modeling of frequency  ------------------

# 1) We start with fetaure pre-processing of the continuous and categorical variables.

# Function that pre-processes continuous variables
Preprocess.Cont <- function(var1,dat1){    
names(dat1)[names(dat1)==var1] <- "V"
dat1$X <- as.numeric(dat1$V)
dat1$X <- 2*(dat1$X-min(dat1$X))/(max(dat1$X)-min(dat1$X))-1   # we use the minmax scaler on the whole data 
                                                              # (if we scale using the mean and sd, we should 
                                                               # only use training data)
names(dat1)[names(dat1)=="V"] <- var1                                                          
names(dat1)[names(dat1)=="X"] <- paste(var1,"X",sep = "")

return(dat1)
}

# Function that pre-processes categorical (non binary) variables (using dummy coding)
Preprocess.Categ <- function(var1,short_,dat1){
names(dat1)[names(dat1)==var1] <- "V"
n_ <- ncol(dat1)  
dat1$X <- as.integer(dat1$V)
n0 <- length(unique(dat1$X))
for (i in 2:n0) {
dat1[,paste(short_,i,sep = "")] <- as.integer(dat1$X==i)
}
names(dat1)[names(dat1)=="V"] <- var1    
return(dat1[,c(1:n_,(n_+2):ncol(dat1))])
}

#  Features pre-processing of the original database 

Features.Preprocess <- function(dat1){
dat1 <- Preprocess.Cont("Area",dat1)
dat1 <- Preprocess.Cont("VehPower",dat1)
dat1$VehAge <- pmin(dat1$VehAge,20) # capping the age of the car in years at 20
dat1 <- Preprocess.Cont("VehAge",dat1)
dat1$DrivAge <- pmin(dat1$DrivAge,90) # capping the age of the driver in years at 90
dat1 <- Preprocess.Cont("DrivAge",dat1)
dat1$BonusMalus <- pmin(dat1$BonusMalus,150) # capping the Bonus Malus at 150
dat1 <- Preprocess.Cont("BonusMalus",dat1)
dat1 <- Preprocess.Categ("VehBrand","Br",dat1)
dat1$VehGasX <- as.integer(dat1$VehGas)-1.5 # we change VehGas levels to -0.5(for diesel) and 0.5
                                                 # for symmetric activation functions

dat1$Density <- round(log(dat1$Density),2)
dat1 <- Preprocess.Cont("Density",dat1)
dat1 <- Preprocess.Categ("Region","Reg.",dat1)
dat1
}

dta_NN <- Features.Preprocess(dta_NN)

rm(Preprocess.Cont,Preprocess.Categ,Features.Preprocess) # cleaning the environment

dta_NN_tilde <- dta_NN

# 2) We then build our network architectures based on the Loss function and the activation functions chosen

   # Loss function (average poisson deviance)
poisson.loss <- function(pred,obs){return(2*(mean(pred)-mean(obs)+mean(log((obs/pred)^obs))))}

  # Functions defining the Networks

seed_nn <- 100 # Random seed to reproduce the results

# Function for the neural network with exposure as an offset where q is a vector containing the hidden layers units
# Hidden_activ (output_activ) are the acivation functions for the hidden layers and the output respectively.
# init_weights are the initial weights.

NN_with_offset <- function(q,hidden_activ,output_activ,init_weights){
set.seed(seed_nn)
tensorflow::set_random_seed(seed_nn)
#use_session_with_seed(42)
n_q <- length(q)
features_NN <- setdiff(names(dta_NN),names(dta_freq))
q0 <- length(features_NN)
X_train_NN <- as.matrix(dta_NN[dta_freq$train,features_NN]) 
X_test_NN <- as.matrix(dta_NN[dta_freq$test,features_NN])
V_train <- as.matrix(log(dta_NN$Exposure[dta_freq$train]))         # Choosing the volumes (exposures)
V_test <- as.matrix(log(dta_NN$Exposure[dta_freq$test]))
Design <- layer_input(shape = c(q0),dtype = "float32",name = "Design")
LogExp <- layer_input(shape = c(1),dtype = "float32",name = "LogExp")
Network=Design %>% 
  layer_dense(units = q[1],activation = hidden_activ,name = paste("hidden",1,sep = "_"))
if(n_q >1){
for(i in 2:n_q){Network= Network %>% 
                         layer_dense(units = q[i],activation = hidden_activ,name = paste("hidden",i,sep = "_"))}
}
  Network = Network %>% 
         layer_dense(units = 1,activation = "linear",name = "Network",
                     weights = list(array(0,dim = c(q[n_q],1)),array(init_weights,dim = c(1))))
Response= list(Network,LogExp) %>% layer_add() %>% 
          layer_dense(units = 1,activation = output_activ,name = "Response",trainable = FALSE,
                      weights = list(array(1,dim = c(1,1)),array(0,dim = c(1))))
Model <- keras_model(inputs = c(Design,LogExp),outputs = c(Response))
return(Model)
}

# Function for the neural network with exposure as an explanatory variable
NN_without_offset <- function(q,hidden_activ,output_activ,init_weights){
set.seed(seed_nn)
tensorflow::set_random_seed(seed_nn) 
  #use_session_with_seed(42)
n_q <- length(q)
features_NN <- c(setdiff(names(dta_NN),names(dta_freq)),"Exposure")
q0 <- length(features_NN)
X_train_NN <- as.matrix(dta_NN[dta_freq$train,features_NN]) 
X_test_NN <- as.matrix(dta_NN[dta_freq$test,features_NN])
  Design <- layer_input(shape = c(q0),dtype = "float32",name = "Design")
  Network=Design %>% 
    layer_dense(units = q[1],activation = hidden_activ,name = paste("hidden",1,sep = "_"))
 if(n_q>1){
   for(i in 2:n_q){Network= Network %>% 
    layer_dense(units = q[i],activation = hidden_activ,name = paste("hidden",i,sep = "_"))}
}
    Network = Network %>% 
    layer_dense(units = 1,activation = output_activ,name = "Network",
                weights = list(array(0,dim = c(q[n_q],1)),array(init_weights,dim = c(1))))
  Response= Network
  Model <- keras_model(inputs = c(Design),outputs = c(Response))
  
  return(Model)
}


# Function that fits the required model and returns the training error, testing error and the predicted testing freq 

NN_final_Model_fit <- function(q,hidden_activ,output_activ,init_weights,batch_size_,epochs_,offset_){
set.seed(seed_nn) 
tensorflow::set_random_seed(seed_nn)
#use_session_with_seed(42)  
Y_train <- as.matrix(dta_NN$ClaimNb[dta_freq$train])
Y_test <- as.matrix(dta_NN$ClaimNb[dta_freq$test])
if(offset_==TRUE){
  model_final <- NN_with_offset(q,hidden_activ,output_activ,init_weights) %>%
                            compile(optimizer=optimizer_nadam(),loss="poisson")
   features_NN <- setdiff(names(dta_NN),names(dta_freq)) 
  q0 <- length(features_NN)
  X_train_NN <- as.matrix(dta_NN[dta_freq$train,features_NN]) 
  X_test_NN <- as.matrix(dta_NN[dta_freq$test,features_NN])
  V_train <- as.matrix(log(dta_NN$Exposure[dta_freq$train]))         # Choosing the volumes (exposures)
  V_test <- as.matrix(log(dta_NN$Exposure[dta_freq$test])) 
  
  fit <- model_final %>% fit(list(X_train_NN,V_train),Y_train,epochs=epochs_,
                            batch_size=batch_size_,verbose=0,validation_split=0)
  plot(fit)
  dta_NN$fitNN <- rep(0,nrow(dta_NN))
  dta_NN$fitNN[dta_freq$train] <- as.vector(model_final %>% predict(list(X_train_NN,V_train)))
  dta_NN$fitNN[dta_freq$test] <- as.vector(model_final %>% predict(list(X_test_NN,V_test)))
}
else{
  model_final <- NN_without_offset(q,hidden_activ,output_activ,init_weights) %>% 
                          compile(optimizer=optimizer_nadam(),loss="poisson") 
   features_NN <- c(setdiff(names(dta_NN),names(dta_freq)),"Exposure")
  q0 <- length(features_NN)
  X_train_NN <- as.matrix(dta_NN[dta_freq$train,features_NN]) 
  X_test_NN <- as.matrix(dta_NN[dta_freq$test,features_NN])
 
  fit <- model_final %>% fit(X_train_NN,Y_train,epochs=epochs_,
                             batch_size=batch_size_,verbose=0,validation_split=0)
  plot(fit)
  dta_NN$fitNN <- rep(0,nrow(dta_NN))
  dta_NN$fitNN[dta_freq$train] <- as.vector(model_final %>% predict(X_train_NN))
  dta_NN$fitNN[dta_freq$test] <- as.vector(model_final %>% predict(X_test_NN))  
}
training_error <- 100 * poisson.loss(dta_NN$fitNN[dta_freq$train],as.vector(unlist(dta_NN$ClaimNb[dta_freq$train])))
test_error <- 100 * poisson.loss(dta_NN$fitNN[dta_freq$test],as.vector(unlist(dta_NN$ClaimNb[dta_freq$test])))
#NN_test_freq <- sum(dta_NN$fitNN[dta_freq$test])/sum(dta_NN$Exposure[dta_freq$test])
#return(c(training_error,test_error,NN_test_freq))
return(c(training_error,test_error))
}


init_weight <- log(sum(dta_NN$ClaimNb[dta_freq$train])/sum(dta_NN$Exposure[dta_freq$train]))
q1 <- 20
q2 <- 15
q3 <- 10
batch_size <- 10000
epochs <- 100

tic()
Table_NN <- data.frame(rbind(NN_final_Model_fit(q1,"tanh",k_exp,init_weight,batch_size,epochs,offset_ = TRUE),
                             NN_final_Model_fit(q1,"tanh",k_exp,init_weight,batch_size,epochs,offset_ = FALSE),  
                             NN_final_Model_fit(c(q1,q2),"tanh",k_exp,init_weight,batch_size,epochs,offset_ = TRUE),
                             NN_final_Model_fit(c(q1,q2),"tanh",k_exp,init_weight,batch_size,epochs,offset_ = FALSE),
                             NN_final_Model_fit(c(q1,q2,q3),"tanh",k_exp,init_weight,batch_size,epochs,offset_ = TRUE),
                             NN_final_Model_fit(c(q1,q2,q3),"tanh",k_exp,init_weight,batch_size,epochs,offset_ = FALSE)))
rownames(Table_NN) <- c("NN with offset (q1=20)","NN without offset (q1=20)","NN with offset (q1=20,q2=15)",
                        "NN without offset (q1=20,q2=15)","NN with offset (q1=20,q2=15,q3=10)",
                        "NN without offset (q1=20,q2=15,q3=10)")  
colnames(Table_NN) <- c("Training Loss %", "Test Loss %")  
# Training and test errors of the three poisson boosting models (no shrinkage) and the best tree and glm models
rbind(Table_NN,Table_5[1,-1])

toc()

(conclusion_table <- rbind(Table_NN[c(1,3,5),],Table_9,Table_7[3,],Table_6[2,],Table_5[1,-1]))
