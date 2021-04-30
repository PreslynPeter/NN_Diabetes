library(plyr)
library(readr)
library(dplyr)
library(caret)
library(neuralnet)
library(nnet)

dat <- read_csv("C:/Users/Preslyn/Desktop/diabetes.csv")
glimpse(dat)
library(caTools)
set.seed(100)

spl = sample.split(dat$class, SplitRatio = 0.7)
train = subset(dat, spl==TRUE)
test = subset(dat, spl==FALSE)

X_train <- scale(train[, c(1:2)])
y_train <- train$class
print(dim(train)); print(dim(test))
dim(y_train) <- c(length(y_train), 1) # add extra dimension to vector
X_test <- scale(test[, c(1:2)])
y_test <- test$class
dim(y_test) <- c(length(y_test), 1) # add extra dimension to vector
X_train <- as.matrix(X_train, byrow=TRUE)
X_train <- t(X_train)
y_train <- as.matrix(y_train, byrow=TRUE)
y_train <- t(y_train)

X_test <- as.matrix(X_test, byrow=TRUE)
X_test <- t(X_test)
y_test <- as.matrix(y_test, byrow=TRUE)
y_test <- t(y_test)
#Build Neural Network
getLayerSize <- function(X, y, hidden_neurons, train=TRUE) {
  n_x <- dim(X)[1]
  n_h <- hidden_neurons
  n_y <- dim(y)[1]   
  
  size <- list("n_x" = n_x,
               "n_h" = n_h,
               "n_y" = n_y)
  
  return(size)
}
layer_size <- getLayerSize(X_train, y_train, hidden_neurons = 4)
layer_size
#Initialize Parameters
initializeParameters <- function(X, list_layer_size){

    m <- dim(data.matrix(X))[2]
    
    n_x <- list_layer_size$n_x
    n_h <- list_layer_size$n_h
    n_y <- list_layer_size$n_y
        
    W1 <- matrix(runif(n_h * n_x), nrow = n_h, ncol = n_x, byrow = TRUE) * 0.01
    b1 <- matrix(rep(0, n_h), nrow = n_h)
    W2 <- matrix(runif(n_y * n_h), nrow = n_y, ncol = n_h, byrow = TRUE) * 0.01
    b2 <- matrix(rep(0, n_y), nrow = n_y)
    
    params <- list("W1" = W1,
                   "b1" = b1, 
                   "W2" = W2,
                   "b2" = b2)
    
    return (params)
}
init_params <- initializeParameters(X_train, layer_size)
lapply(init_params, function(x) dim(x))

sigmoid <- function(x){
    return(1 / (1 + exp(-x)))
}

forwardPropagation <- function(X, params, list_layer_size){
    
    m <- dim(X)[2]
    n_h <- list_layer_size$n_h
    n_y <- list_layer_size$n_y
    
    W1 <- params$W1
    b1 <- params$b1
    W2 <- params$W2
    b2 <- params$b2
    
    b1_new <- matrix(rep(b1, m), nrow = n_h)
    b2_new <- matrix(rep(b2, m), nrow = n_y)
    
    Z1 <- W1 %*% X + b1_new
    A1 <- sigmoid(Z1)
    Z2 <- W2 %*% A1 + b2_new
    A2 <- sigmoid(Z2)
    
    cache <- list("Z1" = Z1,
                  "A1" = A1, 
                  "Z2" = Z2,
                  "A2" = A2)

    return (cache)
}
fwd_prop <- forwardPropagation(X_train, init_params, layer_size)
lapply(fwd_prop, function(x) dim(x))

computeCost <- function(X, class, cache) {
    m <- dim(X)[2]
    A2 <- cache$A2
    logprobs <- (log(A2) * class) + (log(1-A2) * (1-class))
    cost <- -sum(logprobs/m)
    return (cost)
}
cost <- computeCost(X_train, y_train, fwd_prop)
cost
backwardPropagation <- function(X, y, cache, params, list_layer_size){
    
    m <- dim(X)[2]
    
    n_x <- list_layer_size$n_x
    n_h <- list_layer_size$n_h
    n_y <- list_layer_size$n_y

    A2 <- cache$A2
    A1 <- cache$A1
    W2 <- params$W2

    dZ2 <- A2 - y
    dW2 <- 1/m * (dZ2 %*% t(A1)) 
    db2 <- matrix(1/m * sum(dZ2), nrow = n_y)
    db2_new <- matrix(rep(db2, m), nrow = n_y)
    
    dZ1 <- (t(W2) %*% dZ2) * (1 - A1^2)
    dW1 <- 1/m * (dZ1 %*% t(X))
    db1 <- matrix(1/m * sum(dZ1), nrow = n_h)
    db1_new <- matrix(rep(db1, m), nrow = n_h)
    
    grads <- list("dW1" = dW1, 
                  "db1" = db1,
                  "dW2" = dW2,
                  "db2" = db2)
    
    return(grads)
}
back_prop <- backwardPropagation(X_train, y_train, fwd_prop, init_params, layer_size)
lapply(back_prop, function(x) dim(x))
updateParameters <- function(grads, params, learning_rate){

    W1 <- params$W1
    b1 <- params$b1
    W2 <- params$W2
    b2 <- params$b2
    
    dW1 <- grads$dW1
    db1 <- grads$db1
    dW2 <- grads$dW2
    db2 <- grads$db2
    
    
    W1 <- W1 - learning_rate * dW1
    b1 <- b1 - learning_rate * db1
    W2 <- W2 - learning_rate * dW2
    b2 <- b2 - learning_rate * db2
    
    updated_params <- list("W1" = W1,
                           "b1" = b1,
                           "W2" = W2,
                           "b2" = b2)
    
    return (updated_params)
}
update_params <- updateParameters(back_prop, init_params, learning_rate = 0.01)
lapply(update_params, function(x) dim(x))

trainModel <- function(X, y, num_iteration, hidden_neurons, lr){
    
    layer_size <- getLayerSize(X, y, hidden_neurons)
    init_params <- initializeParameters(X, layer_size)
    cost_history <- c()
    for (i in 1:num_iteration) {
        fwd_prop <- forwardPropagation(X, init_params, layer_size)
        cost <- computeCost(X, y, fwd_prop)
        back_prop <- backwardPropagation(X, y, fwd_prop, init_params, layer_size)
        update_params <- updateParameters(back_prop, init_params, learning_rate = lr)
        init_params <- update_params
        cost_history <- c(cost_history, cost)
        
        if (i %% 100 == 0) cat("Iteration", i, " | Cost: ", cost, "\n")
    }
    
    model_out <- list("updated_params" = update_params,
                      "cost_hist" = cost_history)
    return (model_out)
}
EPOCHS = 10
HIDDEN_NEURONS = 4
LEARNING_RATE = 0.9

train_model <- trainModel(X_train, y_train, hidden_neurons = HIDDEN_NEURONS, num_iteration = EPOCHS, lr = LEARNING_RATE)
train_model

makePrediction <- function(X, y, hidden_neurons){
    layer_size <- getLayerSize(X, y, hidden_neurons)
    params <- train_model$updated_params
    fwd_prop <- forwardPropagation(X, params, layer_size)
    pred <- fwd_prop$A2
    
    return (pred)
}
y_pred <- makePrediction(X_test, y_test, HIDDEN_NEURONS)
y_pred <- round(y_pred)
y_pred

glimpse(df)
print(dim(X_train)); print(dim(y_test))
train_params <- trainControl(method = "repeatedcv", number = 10, repeats=5)

nnet_model <- train(train[,-6], train$class,
                 method = "nnet",
                 trControl= train_params,
                 preProcess=c("scale","center"),
                 na.action = na.omit
)
prop.table(table(train$class))   #Baseline Accuracy

# Predictions on the training set
nnet_predictions_train <-predict(nnet_model, train)
table(train$class, nnet_predictions_train)
accuracy<- 180/nrow(train)
accuracy_train<-accuracy
accuracy_train<-(188+350)/(188+350)
accuracy_train*100

#Predictions on the test set
nnet_predictions_test <-predict(nnet_model, test)
table(test$class, nnet_predictions_test)
acc<-((150+80)/nrow(test))*100

#Recall on test set 
recall<-(80/150)*100
recall

#Precision on test set
precision<- ((80/150)*100)
precision

#F1score
precision_plus_recall<- precision + recall
precision_prod_recall<- precision * recall
a<-precision_prod_recall/precision_plus_recall
f1_score<- 2*a
f1_score
