library(MASS)
library(CASdatasets)
library(ggplot2)
library(keras)
library(mgcv)
library(data.table)
library(plyr)
library(tictoc)
data("freMTPL2freq")
dta_freq <- freMTPL2freq
rm(freMTPL2freq)

# Loss function (average poisson deviance)
poisson.loss <- function(pred,obs){return(2*(mean(pred)-mean(obs)+mean(log((obs/pred)^obs))))}

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


# Training and Test Dataset -------------

set.seed(254451710)
u <- runif(nrow(dta_freq),min = 0,max = 1)
#u <- sample (c (1: nrow ( dta_freq )), round (0.9* nrow ( dta_freq )), replace = FALSE )
dta_freq$train <- u < 0.8
dta_freq$test <- !(dta_freq$train)
#train_dat <- subset(dta_freq,subset = train)                # Training dataset
#test_dat <- subset(dta_freq,subset = test)                # Test dataset

rm(u)





dta_NN <- dta_freq     # we copy our database into a new one for the Neural Network modeling since we'll have to
                       # do feature pre-processing (scaling and modifications) before applying the NN.

# GLM MODELING                    

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




              # Model GLM1, GLM2 -------------


res.vars <- "ClaimNb"                   # Response variable
mis.vars <- c("IDpol","Exposure")       # Miscelaneous variables
vars <- c("AreaGLM","VehPowerGLM","VehAgeGLM","DrivAgeGLM","BonusMalusGLM","VehBrand","VehGas","DensityGLM","Region")
vars_glm2 <- c("AreaGLM","VehPowerGLM","VehAgeGLM","BonusMalusGLM","VehBrand","VehGas","DensityGLM","Region","DrivAge",
               "log(DrivAge)","I(DrivAge^2)","I(DrivAge^3)","I(DrivAge^4)")

{t.glm1 <- proc.time()
  d.glm1 <- glm(paste(res.vars,paste(vars,sep = "",collapse = "+"),sep = "~"),data = dta_freq,
                subset = train,family = poisson(link = log),offset = log(Exposure))
  (t.glm1 <-proc.time()-t.glm1)
  }
#(glm1.summary <- summary(d.glm1))


{t.glm2 <- proc.time()
  d.glm2 <- glm(paste(res.vars,paste(vars_glm2,sep = "",collapse = "+"),sep = "~"),data = dta_freq,
                subset = train,family = poisson(link = log),offset = log(Exposure))
  (t.glm2 <-proc.time()-t.glm2)
}
#(glm2.summary <- summary(d.glm2))
N_train_glm1<- fitted(d.glm1)
N_train_glm2<- fitted(d.glm2)
N_test_glm1<- predict(d.glm1,newdata=dta_freq[dta_freq$test,],type="response")
N_test_glm2<- predict(d.glm2,newdata=dta_freq[dta_freq$test,],type="response")
N_train_obs <- dta_freq$ClaimNb[dta_freq$train]
N_test_obs <- dta_freq$ClaimNb[dta_freq$test]
exposure_train <- dta_freq$Exposure[dta_freq$train]
exposure_test <- dta_freq$Exposure[dta_freq$test]

# Table 1 is a summary of the training and the test deviance poisson losses and the average predicted freq on the test sample.
empirical_test_freq <- 100*round(sum(N_test_obs)/sum(exposure_test),4) # empirical frequency in the test dataset
Table_1 <- data.frame(rbind(c(t.glm1[1],100*poisson.loss(N_train_glm1,N_train_obs),100*poisson.loss(N_test_glm1,N_test_obs),
                              100*round(sum(N_test_glm1)/sum(exposure_test),4)),
                            c(t.glm2[1],100*poisson.loss(N_train_glm2,N_train_obs),100*poisson.loss(N_test_glm2,N_test_obs),
                              100*round(sum(N_test_glm2)/sum(exposure_test),4))))
rownames(Table_1) <- c("Model GLM1","Model GLM2")
colnames(Table_1) <- c("Run time","Training Loss %", "Test Loss %","Average frequency %")
Table_1
empirical_test_freq
rm(N_train_glm1,N_train_glm2,N_test_glm1,N_test_glm2,N_train_obs,N_test_obs,exposure_train,exposure_test,t.glm1,
   t.glm2,vars,mis.vars,res.vars,vars_glm2) # cleaning 







# Neural Network Modeling

# 1) We start with fetaure pre-processing of the continuous variables -------------------

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
  dat1$VehBrandX <- as.integer(dat1$VehBrand)-1 # We change vehBrand levels to integers {0,1,...}
  dat1$VehGasX <- as.integer(dat1$VehGas)-1.5 # we change VehGas levels to -0.5(for diesel) and 0.5
  # for symmetric activation functions
  
  dat1$Density <- round(log(dat1$Density),2)
  dat1 <- Preprocess.Cont("Density",dat1)
  dat1$RegionX <- as.integer(dat1$Region)-1  # We change Region levels to integers {0,1,...}
  dat1
}

dta_NN <- Features.Preprocess(dta_NN)

rm(Preprocess.Cont,Features.Preprocess) # cleaning the environment

dta_NN_tilde <- dta_NN

# 2) Then we build an fit our network architectures based on the Loss function and the activation functions chosen ---------
seed_nn <- 100 # Random seed to reproduce the results



features_NN <- setdiff(names(dta_NN),c(names(dta_freq),c("VehBrandX","RegionX" )))
q0 <- length(features_NN)

# Training data
X_train_NN <- as.matrix(dta_NN[dta_NN$train,features_NN]) 
V_train <- as.matrix(log(dta_NN$Exposure[dta_NN$train]))         # Choosing the volumes (exposures)
Br_train <- as.matrix(dta_NN$VehBrandX[dta_NN$train])
Re_train <- as.matrix(dta_NN$RegionX[dta_NN$train])

# Testing data
X_test_NN <- as.matrix(dta_NN[dta_NN$test,features_NN])
V_test <- as.matrix(log(dta_NN$Exposure[dta_NN$test]))
Br_test <- as.matrix(dta_NN$VehBrandX[dta_NN$test])
Re_test <- as.matrix(dta_NN$RegionX[dta_NN$test])

Region.dim <- length(unique(dta_NN$RegionX[dta_NN$train]))
VehBr.dim <- length(unique(dta_NN$VehBrandX[dta_NN$train]))


# Function for the neural network with exposure as an offset where q is a vector containing the hidden layers units
# Hidden_activ (output_activ) are the acivation functions for the hidden layers and the output respectively.
# init_weights are the initial weights and embedd_d is the dimension of the embeddings for Region and VehBrand.

NN_embedd_with_offset <- function(q,hidden_activ,output_activ,init_weights,embedd_d){
set.seed(seed_nn)
tensorflow::set_random_seed(seed_nn)
n_q <- length(q)

Design <- layer_input(shape = c(q0),dtype = "float32",name = "Design")
VehBrand <- layer_input(shape = c(1),   dtype = 'int32', name = 'VehBrand')
Region   <- layer_input(shape = c(1),   dtype = 'int32', name = 'Region')
LogExp <- layer_input(shape = c(1),dtype = "float32",name = "LogExp")


VehBrEmb = VehBrand %>% 
           layer_embedding(input_dim = VehBr.dim,output_dim = embedd_d,input_length = 1,name="VehBrEmb") %>% 
           layer_flatten(name = "VehBr_flat")

RegionEmb = Region %>% 
  layer_embedding(input_dim = Region.dim,output_dim = embedd_d,input_length = 1,name="RegionEmb") %>% 
  layer_flatten(name = "Region_flat")

Network=list(Design,VehBrEmb,RegionEmb)  %>% layer_concatenate(name="New_input_lay") %>% 
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
Model <- keras_model(inputs = c(Design,VehBrand,Region,LogExp),outputs = c(Response))
return(Model)
}



# Function that fits the required model and returns the run time,the training error, testing error and the pred testing freq 

NN_final_Model_fit <- function(q,hidden_activ,output_activ,init_weights,embedd_d,batch_size_,epochs_){
set.seed(seed_nn) 
tensorflow::set_random_seed(seed_nn)  
Y_train <- as.matrix(dta_NN$ClaimNb[dta_freq$train])
Y_test <- as.matrix(dta_NN$ClaimNb[dta_freq$test])

final_model <- NN_embedd_with_offset(q,hidden_activ,output_activ,init_weights,embedd_d) %>%  
                        compile(optimizer=optimizer_nadam(),loss="poisson")
summary(final_model)
{t.nn <- proc.time()
  fit <- final_model %>% fit(list(X_train_NN,Br_train,Re_train,V_train),Y_train,epochs=epochs_,
                                batch_size=batch_size_,verbose=0,validation_split=0)  
t.nn <- proc.time()-t.nn  
}
plot(fit)  

Yfit_train <- as.vector(final_model %>% predict(list(X_train_NN,Br_train,Re_train,V_train)))
Yfit_test <- as.vector(final_model %>% predict(list(X_test_NN,Br_test,Re_test,V_test)))
training_error <- 100* poisson.loss(Yfit_train,as.vector(unlist(dta_NN$ClaimNb[dta_NN$train])))
test_error <- 100* poisson.loss(Yfit_test,as.vector(unlist(dta_NN$ClaimNb[dta_NN$test])))   
NN_test_freq <- 100* round(sum(Yfit_test)/sum(dta_NN$Exposure[dta_NN$test]),4)
return(c(t.nn[1],training_error,test_error,NN_test_freq))
}


init_weight <- log(sum(dta_NN$ClaimNb[dta_NN$train])/sum(dta_NN$Exposure[dta_NN$train]))
q1 <- 20
q2 <- 15
q3 <- 10
batch_size <- 10000
epochs <- 600
embedd_d <- 2
hidden_activ <- "tanh"


Table_2 <- data.frame(rbind(Table_1,
                "NN_embedding (d=2)"= NN_final_Model_fit(c(q1,q2,q3),hidden_activ,k_exp,init_weight,embedd_d,batch_size,epochs)))
colnames(Table_1) <- c("Run time","Training Loss %", "Test Loss %","Average frequency %")
Table_2
rm(X_train_NN,X_test_NN,Br_train,Br_test,Re_train,Re_test,V_train,V_test,Region.dim,VehBr.dim,features_NN,seed_nn) # cleaning

