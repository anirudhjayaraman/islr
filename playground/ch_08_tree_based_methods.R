require(tree)
require(ISLR2)
require(tidyverse)
require(corrplot)
require(gbm)
require(caTools)
require(randomForest)
require(BART)
require(ROCR)

# CLASSIFICATION TREES ========================================================
data("Carseats")
glimpse(Carseats)

# Correlations
numeric_cols <- names(which(sapply(Carseats, class) == 'numeric'))
Carseats %>% select(all_of(numeric_cols)) %>% cor %>% corrplot()

# Price vs Sales
Carseats %>% ggplot() + 
  geom_point(mapping = aes(x = Sales,
                           y = Price)) + theme_bw() + 
  ggtitle("Price as a Function of Sales")

# Some categorical variable relationships
Carseats %>% ggplot() + 
  geom_boxplot(mapping = aes(x = Sales, y = Urban)) + theme_bw() + 
  ggtitle("Sales Distribution by Urban Settlement Status")

Carseats %>% ggplot() + 
  geom_boxplot(mapping = aes(x = Sales, y = US)) + theme_bw() + 
  ggtitle("Sales Distribution by US Residient Status")

Carseats %>% ggplot() + theme_bw() + 
  geom_boxplot(mapping = aes(x = Sales, y = ShelveLoc)) + 
  ggtitle("Sales as a Function of Shelve Location")


# Create a new categorical outcome variable - High - depending on Sales
Carseats <- Carseats %>% 
  mutate(High = factor(if_else(Sales > 8, "Yes", "No")))

# Fit Decision Tree
tree_carseats <- tree(formula = High ~ . -Sales, 
                      data = Carseats)

# Summary provides: 
# - Residual Mean Deviance
# - Misclassification error rate
summary(tree_carseats)

# Plotting Trees - Define a function to plot trees going forward in the code
plot_tree <- function(tree_obj) {
  plot(tree_obj)
  text(tree_obj, pretty = 0)
}

plot_tree(tree_carseats)

# Train and Test Sets to Calibrate the Model ----------------------------------
set.seed(2)
spl <- sample.split(Carseats$Sales, SplitRatio = 0.5)

train <- Carseats %>% filter(spl)
test <- Carseats %>% filter(!spl)

tree_carseats_train <- tree(formula = High ~ . - Sales, data = train)

# Make predictions
test_preds <- predict(tree_carseats_train, newdata = test, type = 'class')
test_pred_table <- table(test$High, test_preds)
test_accuracy <- (test_pred_table[1,1] + test_pred_table[2,2]) / 
  sum(test_pred_table)
print(test_accuracy)

# Pruned Tree after Cross Validation ------------------------------------------
set.seed(10)
# Pruning using misclassification error
cv_tree_01 <- cv.tree(tree_carseats_train, FUN = prune.misclass) 

# Pruning using deviance
cv_tree_02 <- cv.tree(tree_carseats_train)

# Plotting relationships between error / deviance and 
# -> size
# -> cost-complexity parameter

tibble(Size = cv_tree_01$size,
       Deviance = cv_tree_01$dev,
       `Cost Complexity` = cv_tree_01$k) %>% 
  ggplot(aes(x = Size, y = Deviance)) +
  geom_point() +
  stat_smooth(method = "lm", formula = y ~ poly(x, 3), size = 1)


tibble(Size = cv_tree_01$size,
       Deviance = cv_tree_01$dev,
       `Cost Complexity` = cv_tree_01$k) %>% 
  ggplot(aes(x = Size, y = `Cost Complexity`)) +
  geom_point() +
  stat_smooth(method = "lm", formula = y ~ poly(x, 3), size = 1)


# Pruned Tree
pruned_tree <- prune.misclass(tree = tree_carseats_train, best = 4)
plot_tree(pruned_tree)
predict(pruned_tree, test)

# Predictions using pruned tree
test_preds <- predict(pruned_tree, newdata = test, type = 'class')
test_pred_table <- table(test$High, test_preds)
test_accuracy <- (test_pred_table[1,1] + test_pred_table[2,2]) / 
  sum(test_pred_table)
print(test_accuracy)

# REGRESSION TREES ============================================================

data(Boston)

# Glimpse dataset
Boston %>% glimpse()

# Correlations
numeric_cols <- names(which(sapply(Boston, class) == 'numeric'))
Boston %>% select(all_of(numeric_cols)) %>% cor %>% corrplot()

# Train Test Split
set.seed(1)
spl <- sample.split(Boston$medv, SplitRatio = 0.5)
train <- Boston %>% filter(spl)
test <- Boston %>% filter(!spl)

# Fit and Plot Decision Tree
train_boston <- tree(formula = medv ~ ., data = train)
summary(train_boston)
plot_tree(tree_obj = train_boston)

# Already few leaf notes in tree (hence no need to prune)
# Let's see how we ensure more leaves in tree...
train_boston_alt <- tree(formula = medv ~ ., data = train, 
                         control = tree.control(nobs = nrow(train), mindev = 0))
summary(train_boston_alt)
plot_tree(tree_obj = train_boston_alt)

# Let's see what cost complexity pruning does
cv_tree_boston <- cv.tree(object = train_boston_alt)

tibble(Size = cv_tree_boston$size,
       Deviance = cv_tree_boston$dev,
       `Cost Complexity` = cv_tree_boston$k) %>% 
  ggplot(aes(x = Size, y = Deviance)) +
  geom_point() +
  stat_smooth(method = "lm", formula = y ~ poly(x, 10), size = 1) 

# Plot Pruned Tree
pruned_tree_boston <- prune.tree(train_boston_alt, best = 5)
summary(pruned_tree_boston)
plot_tree(pruned_tree_boston)

# Predictions with the pruned tree
test_preds_pruned_tree <- predict(pruned_tree_boston, newdata = test)
test_preds_original_tree <- predict(train_boston, newdata = test)

# Pruned Tree Prediction Quality
preds_01 <- tibble(True = test$medv,
                   Pred = test_preds_pruned_tree)
preds_01 %>% ggplot() +
  geom_point(mapping = aes(x = True, y = Pred)) + 
  theme_bw()
mean((preds_01$True - preds_01$Pred)^2)  # MSE: 32.8292

# Original Tree Prediction Quality
preds_02 <- tibble(True = test$medv,
                   Pred = test_preds_original_tree)
preds_02 %>% ggplot() +
  geom_point(mapping = aes(x = True, y = Pred)) + 
  theme_bw()
mean((preds_02$True - preds_02$Pred)^2)  # MSE: 31.15906

# Bagging and Random Forests ==================================================

set.seed(1)
pl <- sample.split(Boston$medv, SplitRatio = 0.5)
train <- Boston %>% filter(spl)
test <- Boston %>% filter(!spl)

# Fit random forest to training data
rf_boston <- randomForest(medv ~ ., data = train, ntree = 1e4)  
print(rf_boston)  # 89% of variance explained!

# Bagging (A random forest where all predictors will be considered at split)
bag_boston <- randomForest(medv ~ ., 
                           data = train, 
                           mtry = ncol(train) - 1, 
                           ntree = 1e4)
print(bag_boston)  # 88% of variance explained.

rf_test_pred <- predict(rf_boston, newdata = test)
bag_test_pred <- predict(bag_boston, newdata = test)

# Random Forest Performance
tibble(True = test$medv, Pred = rf_test_pred) %>% ggplot() + 
  geom_point(mapping = aes(x = True, y = Pred)) + theme_bw()

# MSE Random Forest
mean((test$medv - rf_test_pred)^2)  # MSE: 14.71414

# Bagging Performance
tibble(True = test$medv, Pred = bag_test_pred) %>% ggplot() + 
  geom_point(mapping = aes(x = True, y = Pred)) + theme_bw()

# MSE Bagging
mean((test$medv - bag_test_pred)^2)  # MSE: 14.12074

# Variable Importance
importance(rf_boston)
importance(bag_boston)

# Boosting ====================================================================

set.seed(1)
pl <- sample.split(Boston$medv, SplitRatio = 0.5)
train <- Boston %>% filter(spl)
test <- Boston %>% filter(!spl)

# Boosting
boost_boston <- gbm(formula = medv ~ ., 
                    data = train, 
                    n.trees = 1e5, 
                    shrinkage = 0.05, 
                    interaction.depth = 4, 
                    n.minobsinnode = 6)

summary(boost_boston)  # plots the  relative influence of predictors

# Partial dependence plots
plot(boost_boston, i = "rm")  # Outcome increases with 'rm'
plot(boost_boston, i = "lstat")  # Outcome decreases with 'lstat'

# Predictions
boost_test_pred <- predict(boost_boston, newdata = test)

# MSE Boosting
mean((test$medv - boost_test_pred)^2)  # MSE: 15.48215

# Plot performance
tibble(True = test$medv,
       Pred = boost_test_pred) %>% ggplot() + 
  geom_point(mapping = aes(x = True, y = Pred)) + theme_bw()

# BART ========================================================================

# BART on Continous Outcome ---------------------------------------------------

# Create train and test data sets
set.seed(1)
pl <- sample.split(Boston$medv, SplitRatio = 0.5)
train <- Boston %>% filter(spl)
test <- Boston %>% filter(!spl)

x_train <- train %>% select(-medv)
y_train <- train %>% pull(medv) # has to be a vector
x_test <- test %>% select(-medv)

# Fit BART
set.seed(1)
bartfit <- gbart(x.train = x_train, 
                 y.train = y_train, 
                 x.test = x_test)
# Predict
bart_test_preds <- bartfit$yhat.test.mean

# MSE
mean((test$medv - bart_test_preds)^2)  # MSE: 15.11634

# BART on Binary Outcome ------------------------------------------------------

data("Carseats")
Carseats <- Carseats %>% 
  mutate(High = if_else(Sales > 8, 1, 0))  # Outcome should be 0 / 1

# Train Test Split
set.seed(2)
spl <- sample.split(Carseats$Sales, SplitRatio = 0.5)

train <- Carseats %>% filter(spl)
test <- Carseats %>% filter(!spl)

x_train_car <- train %>% select(-Sales, -High)
y_train_car <- train %>% pull(High)
x_test_car <- test %>% select(-Sales, -High)

# Fit BART
bartfit_categorical <- pbart(x.train = x_train_car, y.train = y_train_car, x.test = x_test_car)

# Predictions
categorical_bart_preds <- bartfit_categorical$prob.test.mean

# Accuracy of Predictions
roc_curve <- performance(prediction(predictions = categorical_bart_preds, 
                                    labels = test$High), 
                         measure = 'tpr', 
                         x.measure = 'fpr')
auc_roc <- performance(prediction(categorical_bart_preds, 
                                  labels = test$High), 
                       measure = 'auc')@y.values[[1]]  # AUC = 0.897

plot(roc_curve, colorize = TRUE)

