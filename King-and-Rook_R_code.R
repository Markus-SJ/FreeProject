if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(randomForest)) install.packages("randomForest", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(randomForest)

# ------------- Initial Load
king_rook <- read.csv("https://archive.ics.uci.edu/ml/machine-learning-databases/chess/king-rook-vs-king/krkopt.data", header=FALSE)
colnames(king_rook) <- c('w_king_file','w_king_rank','w_rook_file','w_rook_rank',
                         'b_king_file','b_king_rank','result')
king_rook$result <- as.factor(king_rook$result) 
head(king_rook)


# ------------- Outcome Distribution plot
king_rook %>%
  group_by(result) %>% 
  summarize(freq=n()) %>%
  mutate(result = reorder(result, freq)) %>%
  ggplot(aes(result, freq))+
  geom_bar(stat = "identity", color="turquoise") +
  theme(axis.text.x = element_text(angle = 90))

# ------------- Partitioning the Data
set.seed(2021, sample.kind = "Rounding")
test_index <- createDataPartition(king_rook$result, times=1, p=0.8, list=FALSE)
testing  <- king_rook[-test_index,]
whats_left <- king_rook[test_index,]
set.seed(2021, sample.kind = "Rounding")
train_index <- createDataPartition(whats_left$result, times=1, p=0.8, list=FALSE)
training <- whats_left[train_index,]
cross_val <- whats_left[-train_index,]
num_col <- ncol(king_rook)


# ------------- Using knn
# -- The first attempt is using knn:

seq <- 2:25
knn_k <- function(k) {
  knn_fit <- knn3(result ~
                  w_king_file + w_king_rank
                + w_rook_file + w_rook_rank
                + b_king_file + b_king_rank
                ,data = training
                ,k=k)
  knn_hat <- predict(knn_fit, cross_val, type="class")
  confusionMatrix(knn_hat, cross_val$result)$overall["Accuracy"]
}
set.seed(2021, sample.kind = "Rounding")
knn_accuracy <- sapply(seq, knn_k)
plot(seq, knn_accuracy)
seq[which.max(knn_accuracy)]
max(knn_accuracy)

# This method offers just above 0.61 accuracy at its best k=5. Let's try...

# ------------- 
# ______                _                  ______                  _
# | ___ \              | |                 |  ___|                | |      
# | |_/ /__ _ _ __   __| | ___  _ __ ___   | |_ ___  _ __ ___  ___| |_ ___ 
# |    // _` | '_ \ / _` |/ _ \| '_ ` _ \  |  _/ _ \| '__/ _ \/ __| __/ __|
# | |\ \ (_| | | | | (_| | (_) | | | | | | | || (_) | | |  __/\__ \ |_\__ \
# \_| \_\__,_|_| |_|\__,_|\___/|_| |_| |_| \_| \___/|_|  \___||___/\__|___/
# ------------- 
# ------------- Training my mtry 
# Finding the best candidate for my tuning grid
set.seed(2021, sample.kind = "Rounding")
trainMtry <- tuneRF(training[, -num_col], training$result, stepFactor = 1.5, improve = 0.05, ntree = 250)
trainMtry
best_mtry <- 6
# According to the documentation the lower the OOB Error the better the mtry.
# In this case the best is mtry=6

# ------------- Finding the best ntree value
temp_mod <- list()
grid <- data.frame(mtry = c(best_mtry))

# After several tries I have limited the values from 1K to 2K
############# This will take several minutes ############
for (n in c(1000, 1250, 1500, 1750, 2000)) {
  set.seed(2021, sample.kind = "Rounding")
  train_rf <-  train(training[, -num_col], training$result, method = "rf",
                   metric = "Accuracy",
                   trControl = trainControl(method="cv", number = 6),
                   tuneGrid = grid,
                   ntree = n
                   )
  temp_mod[[toString(n)]] <- train_rf
}
model_results <- resamples(temp_mod)
summary(model_results)
best_ntree <- 1250

# These results show that Accuracy goes well over 0.80 using Random Forests 
# for all values of ntree.
# We are going to build the model for the final prediction with ntree = 1250.
# In this case the higher the accuracy (0.8333890) the better the ntree value.

# ------------- Building the final model
grid <- data.frame(mtry = c(best_mtry))
set.seed(2021, sample.kind = "Rounding")
train_rf <-  train(training[, -num_col], 
                   training$result, 
                   method = "rf", 
                   metric = "Accuracy",
                   trControl = trainControl(method = "cv", number = 6),
                   tuneGrid = grid,
                   ntrees = best_ntree
                   )

# ------------- Predicting on test
sv <- predict(train_rf, testing, type = "raw")
mean(sv == testing$result)

# Final accuracy of this model = 0.8495717

# ------------- Adding predictions to the dataset and comparing results
# testing <- testing %>% mutate(pred_result = sv)
# testing %>% filter(b_king_file=='a' & w_rook_file=='b' & w_king_file=='c' & result=="one")

