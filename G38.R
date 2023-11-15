 

# ---------------------------------------------------------------------------
### "After training write code to classify the test data to species according to
### the class predicted as most probable for each iris in the test set, and 
### compute the misclassification rate"

# code to classify the test data

# FUNCTION INPUT: nn = trained network,
#                 testset = data from iris the network has not seen yet,
#                 test_k = k vector containing classes corresponding to the testset 
# FUNCTION OUTPUT: misclassification: the proportion of misclassified data
# DESCRIPTION: This function runs the test data through the trained network using
#              one forward pass and records the classification that the model 
#              predicts. Then it compares the prediction to the true class. 
predict_test <- function(nn, testset, test_k){
  n <- nrow(testset)
  correct <- 0
  
  for (i in 1:n){
    inp <- testset[i,]
    k <- test_k[i]
    result <- forward(nn, inp)
    h <- result$h
    L <- length(h)
    
    predicted_class <- which.max(h[[L]])
    if (predicted_class == k){
      correct <- correct + 1
    }
    cat('correct class:', k, '\n')
    cat('predicted class:', predicted_class, '\n\n')
  }
  # inverse of the proportion of correctly predicted classes
  misclassification <- 1 - correct/n
  
  misclassification 
}

# pre training
predict_test(nn, test_iris, test_k)

# post training
predict_test(trained_nn, test_iris, test_k)
