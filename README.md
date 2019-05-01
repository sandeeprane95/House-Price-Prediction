## House Prices : Advanced Regression Techniques (Kaggle)  



### Problem Description -   
Here, we are presented with a dataset having about 79 attributes that represent the various criteria using which the sale price of a house might be determined. Our goal here is to determine the sale prices of houses given these attributes. To do so, we must train a machine learning model using the training set and then use the developed model to predict the outcome of the test set.  



### File Description -  
1. train.csv – file with the training set  
2. test.csv – file with the test set   
3. data_description.txt - full description of each column, originally prepared by Dean De Cock but lightly edited to match the column names used here.   
4. submission.csv - the output file with the predicted prices for the test sample  



### Approach -  
**Data cleaning** -   
We start by analyzing the data at hand, also known as data exploration. The data was read using pandas library and then we checked the data frame for any missing values. The missing values (NaN) were replaced by either a string value None, an integer 0 or the mode of the column where the field is present. Each column’s missing values were handled on a per-case basis. We also removed the Utilities column as it had the same value for majority of that column and combined the surface area columns of basement, first and second floor into a single column.  
Next the numerical attribute MSSubClass was converted to categorical attribute as it doesn’t have any nominal ordering.  

**Encoding** -   
The categorical attributes need to be converted to integer format. For this, we have two options – Label Encoding and One-hot encoding. The label encoding gave a
better accuracy in the subsequent model and hence it was applied to all the categorical attributes.  
All these operations were performed on both the training as well as the test set.  
The SalePrice attribute was dropped from the dataset and was assigned to another variable as it would be the Y value of the model. Also the Id were dropped in both the sets.  

**Scaling** -   
The MinMaxScaler was applied to all the attributes to scale the values between 0 and 1.  
The SalePrice column is skewed and hence a log (1+x) function was applied to it. This gives us a normal distribution.

**Modelling** -   
Next, we tried out several ML models from sklearn such as Support Vector Regressor, Decision Tree Regressor, Random Forest Regressor and Kernel Ridge Regressor. The models were fitted using the SalePrice and the scaled dataset.  
1. Decision Tree Regressor – Was the worst performing model on the given dataset probably because it uses a single tree for regression.  
2. Support Vector Regressor – A support vector regressor was applied on the dataset with its C parameter set to a value of 5. It gave a good score on k-fold cross validation but a bad score on the actual test set.  
3. Kernel Ridge Regressor – Values of parameters used were alpha = 0.4, coef0 = 2.5 and degree = 0.3. Gave values similar to SVR.  
4. Random Forest – The best model amongst the ones that I tried was Random Forest. Although on k-fold cross validation it had scores higher than the ones given by Kernel Ridge and SVR, it performed well on the test set.  

**Hyperparameter Tuning** –   
The hyperparameters were tuned using GridSearchCV with a cv of 10.  



### Usage Manual -  

**Prerequisites:**  
1. Make sure you have Python 3.x installed.  
2. Certain python modules/packages are essential for successfully running the python file.  
3. You’ll require sklearn, pandas, seaborn, matplotlib and numpy.  

**Steps to execute the .py file on Windows:**  
1. Open Command prompt: Start menu -> Run and type 'cmd'. This will cause the Windows terminal to open.  
2. Type 'cd ' followed by project directory path to change current working directory, and hit Enter.  
3. Run the program using command 'python Prediction.py'  
Alternatively, you can also use an IDE of your choice to execute the code.  

**Steps to execute the .py file on Mac:**  
1. Open Terminal by searching it using the search icon at top right or through 'Launchpad->Other->Terminal'  
2. Type 'cd ' followed by directory path to change current working directory, and hit Enter.  
3. Run the program using command 'python3 Prediction.py'  
