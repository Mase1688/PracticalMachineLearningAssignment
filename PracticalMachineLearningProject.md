## Johns Hopkins University - Practical Machine Learning - Peer Graded Assignment
### Creator:  Mason Haupt
### Date:     2019-01-21

## Description:
### What you should submit
The goal of your project is to predict the manner in which they did the exercise. This is the "classe" variable in the training set. You may use any of the other variables to predict with. You should create a report describing how you built your model, how you used cross validation, what you think the expected out of sample error is, and why you made the choices you did. You will also use your prediction model to predict 20 different test cases.

### Peer Review Portion
Your submission for the Peer Review portion should consist of a link to a Github repo with your R markdown and compiled HTML file describing your analysis. Please constrain the text of the writeup to < 2000 words and the number of figures to be less than 5. It will make it easier for the graders if you submit a repo with a gh-pages branch so the HTML page can be viewed online (and you always want to make it easy on graders :-).

###Course Project Prediction Quiz Portion
Apply your machine learning algorithm to the 20 test cases available in the test data above and submit your predictions in appropriate format to the Course Project Prediction Quiz for automated grading.

### Reproducibility
Due to security concerns with the exchange of R code, your code will not be run during the evaluation by your classmates. Please be sure that if they download the repo, they will be able to view the compiled HTML version of your analysis.

### Prediction Assignment Writeup Background

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).

### Data

The training data for this project are available here:

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

The test data are available here:

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

The data for this project come from this source: http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har. If you use the document you create for this class for any purpose please cite them as they have been very generous in allowing their data to be used for this kind of assignment.

### Method
1. Obtain Datasets (i.e. pml-testing and pml-training)
2. Clean/Tidy (i.e. get rid of NA and irrelevant cells)
3. Split pml-training dataset into 75%/25% training and testing datasets respectively.
4. Build several models.
5. Determine best model for goodness-of-fit without:
    a. Overfitting
    b. Creating too much bias
    c. Inaccuracy
6. Apply chosen model to test data.

### Output
#### 1.

library(data.table)
library(caret)
library(randomForest)
library(rpart)
library(rpart.plot)
library(lattice) 
library(ggplot2)

getwd(##find your working directory if needed)
setwd(##set your working directory if needed)

fileURL <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
fileURL2 <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

download.file(fileURL, destfile = "./pml-training.csv")
download.file(fileURL2, destfile = "./pml-testing.csv")

#### 2.
training <- "./pml-training.csv"
testing <- "./pml-testing.csv"

training <- read.csv(training, na.strings = c("NA","#DIV/0!",""))
testing <- read.csv(testing, na.strings = c("NA","#DIV/0!",""))

str(training)
str(testing)

training <- training[, -c(1:7)]
training <- training[, colSums(is.na(training)) == 0]

testing <- testing[, -c(1:7)]
testing <- testing[, colSums(is.na(testing)) == 0]

#### 3.

inTrain <- createDataPartition(y=training$classe, p = 0.75, list = FALSE)
training <- training[inTrain, ]
training_test <- training[-inTrain, ]

#### 4.
set.seed(14717)

modFitGBM <- train(classe ~., data=training, method = "gbm", verbose = FALSE)
modFitClass <- train(classe ~., data=training, method = "rpart")
modFitRF <- train(classe ~., data=training, method = "rf", prox = TRUE)

#### 5.
predictGBM <- predict(modFitGBM, training_test)
predictClass <- predict(modFitClass, training_test)
predictRF <- predict(modFitRF, training_test)

confusionMatrix(predictGBM, training_test$classe)
confusionMatrix(predictClass, training_test$classe)
confusionMatrix(predictRF, training_test$classe)

GBM:  Accuracy 97%  Sensitivity >=  94.5%  Specificity      99%
DT:   Accuracy 49%  Sensitivity >=  0%     Specificity  >=  77%
RF:   Accuracy 100% Sensitivity     100%   Specificity     100%

#### Results:
##### GBM
Confusion Matrix and Statistics

          Reference
Prediction    A    B    C    D    E
         A 1046   20    0    0    0
         B    6  670   15    0    7
         C    2   17  616   14    6
         D    1    2    6  594   14
         E    0    0    0    0  633

Overall Statistics
                                         
               Accuracy : 0.97           
                 95% CI : (0.964, 0.9753)
    No Information Rate : 0.2875         
    P-Value [Acc > NIR] : < 2.2e-16      
                                         
                  Kappa : 0.962          
 Mcnemar's Test P-Value : NA             

Statistics by Class:

                     Class: A Class: B Class: C Class: D Class: E
Sensitivity            0.9915   0.9450   0.9670   0.9770   0.9591
Specificity            0.9923   0.9905   0.9871   0.9925   1.0000
Pos Pred Value         0.9812   0.9599   0.9405   0.9627   1.0000
Neg Pred Value         0.9965   0.9869   0.9930   0.9954   0.9911
Prevalence             0.2875   0.1932   0.1736   0.1657   0.1799
Detection Rate         0.2851   0.1826   0.1679   0.1619   0.1725
Detection Prevalence   0.2905   0.1902   0.1785   0.1682   0.1725
Balanced Accuracy      0.9919   0.9678   0.9771   0.9847   0.9795

##### Decision Tree(s)
Confusion Matrix and Statistics

          Reference
Prediction   A   B   C   D   E
         A 953 292 268 252 104
         B  20 243  23  99  97
         C  77 174 346 257 186
         D   0   0   0   0   0
         E   5   0   0   0 273

Overall Statistics
                                         
               Accuracy : 0.4947         
                 95% CI : (0.4784, 0.511)
    No Information Rate : 0.2875         
    P-Value [Acc > NIR] : < 2.2e-16      
                                         
                  Kappa : 0.3397         
 Mcnemar's Test P-Value : NA             

Statistics by Class:

                     Class: A Class: B Class: C Class: D Class: E
Sensitivity            0.9033  0.34274   0.5432   0.0000  0.41364
Specificity            0.6496  0.91926   0.7711   1.0000  0.99834
Pos Pred Value         0.5099  0.50415   0.3327      NaN  0.98201
Neg Pred Value         0.9433  0.85378   0.8893   0.8343  0.88587
Prevalence             0.2875  0.19324   0.1736   0.1657  0.17989
Detection Rate         0.2597  0.06623   0.0943   0.0000  0.07441
Detection Prevalence   0.5094  0.13137   0.2835   0.0000  0.07577
Balanced Accuracy      0.7764  0.63100   0.6571   0.5000  0.70599

##### Random Forest
Confusion Matrix and Statistics

          Reference
Prediction    A    B    C    D    E
         A 1055    0    0    0    0
         B    0  709    0    0    0
         C    0    0  637    0    0
         D    0    0    0  608    0
         E    0    0    0    0  660

Overall Statistics
                                    
               Accuracy : 1         
                 95% CI : (0.999, 1)
    No Information Rate : 0.2875    
    P-Value [Acc > NIR] : < 2.2e-16 
                                    
                  Kappa : 1         
 Mcnemar's Test P-Value : NA        

Statistics by Class:

                     Class: A Class: B Class: C Class: D Class: E
Sensitivity            1.0000   1.0000   1.0000   1.0000   1.0000
Specificity            1.0000   1.0000   1.0000   1.0000   1.0000
Pos Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
Neg Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
Prevalence             0.2875   0.1932   0.1736   0.1657   0.1799
Detection Rate         0.2875   0.1932   0.1736   0.1657   0.1799
Detection Prevalence   0.2875   0.1932   0.1736   0.1657   0.1799
Balanced Accuracy      1.0000   1.0000   1.0000   1.0000   1.0000

Chose GBM due to RF overfitting.

#### 6.
predictfinal <- predict(modFitGBM, testing)
predictfinal
 [1] B A B A A E D B A A B C B A E E A B B B
Levels: A B C D E
