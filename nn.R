num.grad <- function(f, x) {
  fx <- f(x)
  grad <- rep(0, length(x))
  h <- 0.00001
  for (i in 1:length(x)) {
    xi <- x[i]
    fxph <- f(xi + h)
    fxmh <- f(xi -h)
    grad[i] <- (fxph - fxmh) / (2 * h)
  }
  return(grad)
}

relu <- function(x) {
  return(pmax(x, 0))
}

softmax <- function(scores) {
  es <- exp(scores)
  return(es / rowSums(es))
}

argmax <- function(v) {
  i.max <- 1
  for (i in 1:length(v)) {
    if (v[i] > v[i.max]) {
      i.max <- i
    }
  }
  return(i.max)
}

generate.spiral <- function(N, D, K) {
  M <- N*K # Total number of examples
  X <- matrix(0, M,D) # data matrix (each row = single example)
  y <- rep(0, M) # class labels
  for (j in 1:K){
    r <- 0
    t <- j * 4
    start <-N*(j-1) +1
    end <- N*j
    for (k in start:end) {
      tp <- t+ runif(1) * 0.2
      X[k, 1] <- r * sin(tp)
      X[k, 2] <- r * cos(tp)
      y[k] <- j
      r <- r + 1/N
      t <- t + 4/N
    }
  }
  return(list(X = X, y = y))
}

parse.iris <- function() {
  X <- as.matrix(iris[,1:4])
  y <- as.integer(as.factor(iris[,5]))
  return(list(X = X, y = y))
}


get.acc <- function(X, y, W1, W2, b1, b2) {
  y.pred <- rep(0, length(y))
  hidden.layer <- relu(X %*% W1 + b1)
  scores <-  hidden.layer %*% W2 + b2 
  for (i in 1:length(y)) {
    y.pred[i] <- argmax(scores[i,])
  }
  return(sum(y.pred == y) / length(y))
}

main <- function() {
  #num <- 100 # number of points per class
  #dim <- 2 # dimensionality
  #class <- 3 # number of classes
  #Xy <- generate.spiral(num, dim, class)
  Xy <- parse.iris()
  X <- Xy$X
  y <- Xy$y
  train <- sample(c(TRUE, FALSE), nrow(X), replace=TRUE, prob=c(0.7,0.3))
  Xtr <- X[train,]
  ytr <- y[train]
  Xte <- X[!train,]
  yte <- y[!train]
  # visualize the data
  #plot(X[, 1], X[, 2], col=y)
  
  D <- ncol(Xtr)
  M <- nrow(Xtr)
  K <- max(y)
  # Initialize weights and biases
  # Neural network
  H <- 128 # Hidden layer size
  W1 <- matrix(rnorm(D*H), nrow=D) * 0.01
  b1 <- rep(0, H)
  W2 <- matrix(rnorm(H*K), nrow=H) * 0.01
  b2 <- rep(0, K)
  # Softmax classifier
  #W <- matrix(rnorm(D*K), nrow= D) * 0.01
  #b <- rep(0, K)
  lambda <- 0.1 # Regularization constant
  alpha <- 0.01 # Step size
  batch.size <- M/10
  maxit <- 100 * M / batch.size
  tr.loss.arr <- rep(0, maxit)
  te.loss.arr <- rep(0, maxit)
  
  for (it in 1:maxit) {
    randperm <- sample(1:length(ytr))
    epoch.loss <- 0
    dW1.old <- 0
    dW2.old <- 0
    db1.old <- 0
    db2.old <- 0
    for (batch in 1:(M / batch.size)) {
      batch.begin <- batch.size * (batch - 1) + 1
      batch.end <- batch.size * batch
      Xbatch <- Xtr[randperm, ][batch.begin:batch.end,]
      ybatch <- ytr[randperm][batch.begin:batch.end]
      # NN
      hidden.layer <- relu(Xbatch %*% W1 + b1)
      scores <-  hidden.layer %*% W2 + b2 # Pass through NN
      # Softmax
      # scores <- X %*% W + b
      probs <- softmax(scores) # Softmax so the probs sum to 1
      correct.logprob <- rep(0, batch.size)
      for (i in 1:batch.size) { # For each example
        correct.logprob[i] <- -log(probs[i, ybatch[i]]) # Take the log prob of the correct class
      }
      data.loss <- sum(correct.logprob) / batch.size
      reg.loss <- 0.5 * lambda * (sum(W1 * W1) + sum(W2 * W2))
      loss <- data.loss + reg.loss
      #cat("it = ", it, ", loss = ", loss, "\n")
      epoch.loss <- epoch.loss + loss
      # Convention for backpropagation: variables are called like the denominator
      # in Leibniz notation
      # Softmax backprop: dLoss / dScores = scores - I(k = y)
      dScores <- probs
      for (i in 1:batch.size) {
        dScores[i, ybatch[i]] <- dScores[i, ybatch[i]] - 1
      }
      dScores <- dScores / batch.size
      # Softmax weight backprop: dLoss / dW = dLoss / dScores * dScores / dW
      # dScores / dW = d/dW (X^T W + b) = X^T
      # dScores / db = d/db (X^T W + b) = 1
      # --> dLoss / dW = (scores - I(k = y)) * X^T; dLoss / db = (scores - I(k = y)) 
      #dW <- t(X) %*% dScores
      #db <- rowSums(dScores)
      #dW <- dW + W * lambda
      # NN backprop
      # dLoss / dW1 = dLoss / dScores * W2 * dReLU / dW1 * X
      # dLoss / dW2 = dLoss / dScores * ReLU(XW1 + b) 
      dW2 <- t(hidden.layer) %*% dScores + lambda * W2
      db2 <- rowSums(dScores)
      dHidden <- dScores %*% t(W2)
      for (i in 1:nrow(dHidden)) {
        for (j in 1:ncol(dHidden)) {
          if (hidden.layer[i, j] <= 0) {
            dHidden[i, j] <- 0 # ReLU backprop
          }
        }
      }
      dW1 <- t(Xbatch) %*% dHidden + lambda * W1
      db1 <- rowSums(dHidden)
      
      # Apply momentum
      momentum <- 0.9
      dW1 <- momentum * dW1.old + dW1
      dW2 <- momentum * dW2.old + dW2
      db1 <- momentum * db1.old + db1
      db2 <- momentum * db2.old + db2
      dW1.old <- dW1
      dW2.old <- dW2
      db1.old <- db1
      db2.old <- db2
      
      # SGD step
      W1 <- W1 - alpha * dW1
      b1 <- b1 - alpha * db1
      W2 <- W2 - alpha * dW2
      b2 <- b2 - alpha * db2
      
    }
    # Train loss
    epoch.loss <- epoch.loss / (M/batch.size)
    tr.loss.arr[it] <- epoch.loss
    # Validation step
    hidden.layer <- relu(Xte %*% W1 + b1)
    scores <-  hidden.layer %*% W2 + b2
    probs <- softmax(scores)
    correct.logprob <- rep(0, length(yte))
    for (i in 1:length(yte)) { 
      correct.logprob[i] <- -log(probs[i, yte[i]]) 
    }
    data.loss <- sum(correct.logprob) / length(yte)
    reg.loss <- 0.5 * lambda * (sum(W1 * W1) + sum(W2 * W2))
    val.loss <- data.loss + reg.loss
    cat("Epoch ", it, ", val loss = ", val.loss,", train loss = ", epoch.loss, "\n")
    te.loss.arr[it] <- loss
  }
  
  
  cat("Train accuracy = ", get.acc(Xtr, ytr, W1, W2, b1, b2), "\n")
  cat("Test accuracy = ", get.acc(Xte, yte, W1, W2, b1, b2), "\n")
 
  colors <- c(2, 4)
  smooth.tr.l <- movavg(tr.loss.arr, type="e", n = 50)
  smooth.te.l <- movavg(te.loss.arr, type="e", n = 50)
  plot(smooth.tr.l, type="l", col=colors[1])
  lines(smooth.te.l, col=colors[2])
  legend(x="topright",legend=c("Train loss", "Test loss"), col=colors, lty=c(1, 1), lwd=2)

}
library(pracma)
main()

