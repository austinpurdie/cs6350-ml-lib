library(dplyr)
library(ggplot2)
library(reshape)
library(RColorBrewer)

adaboost_accuracy <- read.csv("EnsembleLearning/adaboost.accuracy.csv")
bagging_accuracy <- read.csv("EnsembleLearning/bagging-accuracy.csv")
random_forest_accuracy <- read.csv("EnsembleLearning/random-forest-accuracy.csv")
default_adaboost_accuracy <- read.csv("EnsembleLearning/random-forest.accuracy.csv")
default_bagging_accuracy <- read.csv("EnsembleLearning/default-bagging.accuracy.csv")
default_random_forest_accuracy <- read.csv("default-random-forest-acuracy.csv")
grad_cost <- read.csv("LinearRegression/grad_cost.csv")
stoch_grad_cost <- read.csv("LinearRegression/stoch_grad_cost.csv")

###############
# ADABOOST 2a #
###############


colnames(adaboost_accuracy) <- c('index', 'iteration', 'ada_test', 'ada_train', 'stump_test', 'stump_train')

adaboost_accuracy <-
  adaboost_accuracy %>% 
  dplyr::select(!'index')

adaboost_accuracy <- as.data.frame(adaboost_accuracy)

melt_adaboost_accuracy <- melt(adaboost_accuracy, id = 'iteration')

adaboost_accuracy_notstump <-
  melt_adaboost_accuracy %>% 
  filter(variable == 'ada_train' | variable == 'ada_test')

prob2a_adaboost_acc <- ggplot(data = adaboost_accuracy_notstump, aes(x = iteration, y = value, color = variable)) +
    xlab("Iteration") +
    ylab("Accuracy") +
    ggtitle("AdaBoost Accuracy") +
    scale_color_brewer(palette = "Dark2", name = "", labels = c("Test", "Train")) + 
    geom_line()

prob2a_adaboost_acc


adaboost_accuracy_stump <-
  melt_adaboost_accuracy %>% 
  filter(variable == "stump_train" | variable == "stump_test")

prob2a_adaboost_acc_stump <- ggplot(data = adaboost_accuracy_stump, aes(x = iteration, y = value, color = variable)) +
  xlab("Iteration") +
  ylab("Accuracy") +
  ggtitle("AdaBoost Stump Accuracy") +
  scale_color_brewer(palette = "Dark2", name = "", labels = c("Test", "Train")) + 
  geom_line(size = 0.2)

prob2a_adaboost_acc_stump

##############
# BAGGING 2b #
##############

bagging_accuracy <- as.data.frame(bagging_accuracy)

bagging_accuracy <-
  bagging_accuracy %>% 
  select(Iteration:Train)

melt_bagging_accuracy <- melt(bagging_accuracy, id = 'Iteration')

prob2b_bagging_acc <- ggplot(data = melt_bagging_accuracy, aes(x = Iteration, y = value, color = variable)) +
  xlab("Iteration") +
  ylab("Accuracy") +
  ggtitle("Bagging Accuracy") +
  scale_color_brewer(palette = "Dark2", name = "", labels = c("Test", "Train")) + 
  geom_line(size = 0.5)

prob2b_bagging_acc

####################
# RANDOM FOREST 2d #
####################

random_forest_accuracy <- as.data.frame(random_forest_accuracy)

random_forest_accuracy <-
  random_forest_accuracy %>% 
    select(Iteration:'k = 6, Train')

melt_random_forest_accuracy <- melt(random_forest_accuracy, id = 'Iteration')

prob2d_rf_all <- ggplot(data = melt_random_forest_accuracy, aes(x = Iteration, y = value, color = variable)) +
  xlab("Iteration") +
  ylab("Accuracy") +
  ggtitle("Random Forest") +
  scale_color_brewer(palette = "Dark2", name = "") + 
  geom_line(size = 0.5)

prob2d_rf_all

#############
# PROBLEM 3 #
#############

single_train <- 0.972125
single_test <- 0.651833

default_adaboost_accuracy <- as.data.frame(default_adaboost_accuracy)
default_bagging_accuracy <- as.data.frame(default_bagging_accuracy)
default_random_forest_accuracy <- as.data.frame(default_random_forest_accuracy)

default_adaboost_accuracy <-
  default_adaboost_accuracy %>% 
  select(Iteration:'Stump Train Accuracy')

default_bagging_accuracy <-
  default_bagging_accuracy %>% 
  select(Iteration:Train)

default_random_forest_accuracy <-
  default_random_forest_accuracy %>% 
  select(Iteration:Train)

melt_default_adaboost_accuracy <- melt(default_adaboost_accuracy, id = 'Iteration')
melt_default_bagging_accuracy <- melt(default_bagging_accuracy, id = 'Iteration')
melt_default_random_forest_accuracy <- melt(default_random_forest_accuracy, id = 'Iteration')

default_adaboost_accuracy_notstump <-
  melt_default_adaboost_accuracy %>% 
  filter(variable == 'AdaBoost Test Accuracy' | variable == 'AdaBoost Train Accuracy')

default_adaboost_accuracy_stump <-
  melt_default_adaboost_accuracy %>% 
  filter(variable == 'Stump Test Accuracy' | variable == 'Stump Train Accuracy')

prob3_adaboost_acc <- ggplot(data = default_adaboost_accuracy_notstump, aes(x = Iteration, y = value, color = variable)) +
  xlab("Iteration") +
  ylab("Accuracy") +
  ggtitle("AdaBoost Accuracy") +
  scale_color_brewer(palette = "Dark2", name = "", labels = c("Test", "Train")) + 
  geom_line() +
  geom_hline(yintercept = single_test, linetype = 'dashed', color = 'darkred', size = 1) +
  geom_hline(yintercept = single_train, linetype = 'dotted', color = 'darkred', size = 1)

prob3_adaboost_acc

prob3_bagging_acc <- ggplot(data = melt_default_bagging_accuracy, aes(x = Iteration, y = value, color = variable)) +
  xlab("Iteration") +
  ylab("Accuracy") +
  ggtitle("Bagging Accuracy") +
  scale_color_brewer(palette = "Dark2", name = "", labels = c("Test", "Train")) + 
  geom_line(size = 0.5) +
  geom_hline(yintercept = single_test, linetype = 'dashed', color = 'darkred', size = 1) +
  geom_hline(yintercept = single_train, linetype = 'dotted', color = 'darkred', size = 1)

prob3_bagging_acc

prob3_rf <- ggplot(data = melt_default_random_forest_accuracy, aes(x = Iteration, y = value, color = variable)) +
  xlab("Iteration") +
  ylab("Accuracy") +
  ggtitle("Random Forest") +
  scale_color_brewer(palette = "Dark2", name = "") + 
  geom_line(size = 0.5) +
  geom_hline(yintercept = single_test, linetype = 'dashed', color = 'darkred', size = 1) +
  geom_hline(yintercept = single_train, linetype = 'dotted', color = 'darkred', size = 1)

prob3_rf

############
#PROBLEM 4 #
############

grad_cost$iteration = seq(3383)
stoch_grad_cost$iteration = seq(352821)

grad_cost_plot <- ggplot(data = grad_cost, aes(x = iteration, y = X1)) +
  xlab("Iteration") +
  ylab("Cost") +
  ggtitle("Gradient Descent Cost Function Values") +
  scale_color_brewer(palette = "Dark2") +
  geom_line(color = 'darkred')

grad_cost_plot


stoch_grad_cost_plot <- ggplot(data = stoch_grad_cost, aes(x = iteration, y = X1)) +
  xlab("Iteration") +
  ylab("Cost") +
  ggtitle("Stochastic Gradient Descent Cost Function Values") +
  scale_color_brewer(palette = "Dark2") +
  geom_line(color = 'steelblue', size = 0.01)

stoch_grad_cost_plot













