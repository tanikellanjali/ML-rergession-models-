# ML-regression-models
Linear , multilinear , logistic regression  and decision tree 
This file includes the application of  three regression machine learning models and decision tree. 

## Linear Regression 
A linear regression models are usually simple and easy to understand. the dependencies are linear so the variations can be easily recorded in linear regression. 
The mathematical formula a linear regression model usually follows is 
Y = mX + B 
Where 
m – slope 
Y - the variable to predict 
X – variable used to predict 
B – constant 
As the formular is easy to follow , the linear regression is the least complex model of all 

## Multi-linear regression
In case of multilinear regression model , The output or a specific value of the data can be predicted based on two or more values . This is usually helpful when the value is dependent on more variables . This can be used for large data sets and we can find better results in comparison to linear regression model 
Y = B0 + B1X1 + B2X2 + B3X3 + ………… BnXn + e
Where 
Y – Dependent Variable 
B0 – Y intercept 
B1 , B2 , …. – coefficients 
E – residual error 

## Decision Tree 
Decision trees can usually perform both classifications and predictions based on the classifications . A decision tree is usually a broad spectrum , where we can create an yea no categorization to easily understand which value depends on what . However with a large amount of data the decision tress get huge and complication to read . 
Being a tree structure , decision trees usually have nodes and branches . The last node of the branch is called a leaf node . 
The algorithm splits the nodes into two at each point based on the class functions . on every split , the algorithm tries to minimize the loss function by splitting it into the smallest number of classes . 
Information gain usually depends on entropy function 
Entropy = sum(PilogPi) where I = 1…j
Pi = probability of occurrence of a certain node 
I = number of iterations or nodes . 

## Logistic Regression 
This is usually used for multiclass regression models . This regression model helps us analyze the possibility of an event occurring in the future  given a data set with independent variables . 
The major difference between logistic and decision tree is , the logistic algorithm uses the log transformation function for scaling the data . The probability is divided by the probability of failure . The coefficients are usually found through maximum likelihood estimation . 

Log ( pi ) = 1 / (1 + exp (-pi))
Where 
Pi is a probability of each value occurrence 


## Code 
Step 1 :
•	Imported the necessary libraries 
•	Downloaded the data with pd as a data frame 
•	Viewed the data and understood the correlations using numpy and pandas functions 
•	Pre processed the object data times into numerical , preparing it for regression models 
•	Used functions like info and isnull for understanding data better 
Step 2 : 
•	Divided the data into independent variables and dependent variables 
•	Divided the X and Y based on train_test_split library at 0.25 division 
•	Fit the values into regressors 
Step 3 :
•	The models trained on training data 
•	Fit test data to predict the data 
•	Mapped the predicted data to test data to understand the change 
•	R2 square error and square root error comparison of all four methods for this data 
Step 4 : 

•	Projected the predicted and test values together on one map , Below are the four graphs for each models 

## Conclusion 
•	Through out all the output and observations , I understand that the dependency of the variable is important and the primary step in any exploratory analysis for machine learning prediction models needs to be finding dependent variables and making sure our prediction models are dominantly on the basis of them . 

•	The important values should always be made into categorical values as from the decision tree classifier , we understand how the values that were categorized worked well and aligned with the predicted values where as the other values varied vastly . 


## Reference 
https://www.ibm.com/topics/linear-regression - linear regression information 
Images for formulas – google images 
Code – 100 days of ML - https://github.com/cchangyou/100-Days-Of-ML-Code

