# -*- coding: utf-8 -*-
"""
Created on Sun Jun  5 23:17:02 2022

@author: lenovo
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# loading train data 
# train.csv contains the leads information of last 1 year from Jan 2021 to Dec 2021. And also the target variable indicating if the user will buy the product in next 3 months or not 

train_ad = pd.read_csv(r'Downloads\train_ad.csv')

train_ad.head() ##To view first five rows in the dataset
train_ad.info() #To view the object, float and integer columns and its datatypes in the dataset
train_ad.shape #To check the size of the data
train_ad.columns #To see the column names 

train_ad.describe() #To find the statistical property of data

# checking any null values
train_ad.isnull().sum()

#Visualizing the Missing values on heatmap
sns.heatmap(train_ad.isnull(),cbar=False,cmap='viridis')
train_ad.isna().sum()

# fill null values with farward fill
train_ad = train_ad.fillna(method='ffill') 

# checking null values still exist or not
train_ad.isnull().sum() # no na values

 
#To check imbalance 
train_ad['buy'].value_counts() 

sns.countplot(train_ad['buy'])
# given data is imbalanced dataset.


#### Checking the Outliers
#boxplot - checking presence of outliers
f, ax = plt.subplots(2,4, figsize = (15,7))
plt.suptitle('Boxplot of numerical features')
sns.boxplot(x = train_ad.campaign_var_1, ax = ax[0][0]) # no outliers
sns.boxplot(x = train_ad.campaign_var_2, ax = ax[0][1]) # outliers
sns.boxplot(x = train_ad.user_activity_var_1, ax = ax[0][2]) # outliers
sns.boxplot(x = train_ad.products_purchased, ax = ax[0][3]) # no outliers
sns.boxplot(x = train_ad.user_activity_var_2, ax = ax[1][0]) # outliers
sns.boxplot(x = train_ad.user_activity_var_6, ax = ax[1][1]) #outliers
sns.boxplot(x = train_ad.user_activity_var_11, ax = ax[1][1]) #outliers
f.delaxes(ax[1,3]) 


#creating winsorization techniques to handle outliers
from feature_engine.outliers import Winsorizer
gaussian_winsor = Winsorizer(capping_method='gaussian', tail='both', fold=3)
iqr_winsor = Winsorizer(capping_method='iqr', tail='both',fold=1)
quantiles_winsor = Winsorizer(capping_method='quantiles', tail='both', fold=0.10)

#handling outliers
train_ad[['campaign_var_2']] = iqr_winsor.fit_transform(train_ad[['campaign_var_2']])
train_ad[['user_activity_var_1']] = iqr_winsor.fit_transform(train_ad[['user_activity_var_1']])
train_ad[['user_activity_var_6']] = iqr_winsor.fit_transform(train_ad[['user_activity_var_6']])
train_ad[['user_activity_var_11']] = iqr_winsor.fit_transform(train_ad[['user_activity_var_11']])

    # boxplot for checking outliers after winsorization
sns.boxplot(train_ad.campaign_var_2)
sns.boxplot(train_ad.user_activity_var_1)
sns.boxplot(train_ad.user_activity_var_6)
sns.boxplot(train_ad.user_activity_var_11)
# no outliers

# check for duplicate values
train_ad.duplicated().sum() # no duplicate values


# handling date columns by splitting as 3 columns day, month, year
train_ad['Year'] = train_ad['created_at'].str.split('-').str[0].astype(int)
train_ad['Month'] = train_ad['created_at'].str.split('-').str[1].astype(int)
train_ad['Day'] = train_ad['created_at'].str.split('-').str[1].astype(int)

train_ad.drop('created_at', axis=1, inplace=True)

train_ad['Signup_Year'] = train_ad['signup_date'].str.split('-').str[0].astype(int)
train_ad['Signup_Month'] = train_ad['signup_date'].str.split('-').str[1].astype(int)
train_ad['Signup_Day'] = train_ad['signup_date'].str.split('-').str[1].astype(int)

train_ad.drop('signup_date', axis=1, inplace=True)


#  EDA (Exploratory Data Analysis)
# --------------------------------
 
# Descrptive Analytics
# # Measure of central tendancy - 1st moment business decision 

train_ad.mean()

# Observation: 1) average campaign_var_1 is 6.8
        #      2) average campaign_var_2 is 6.8
        #      3) average products purchased is 2
        
        
train_ad.median()
# Observation:  1.average campaign_var_1 is 6
        #      2) average campaign_var_2 is 7
        #      3) average products purchased is 2
        
        
# # measure of dispersion
train_ad.var() # 2nd moment business decision -var(), std()
# variance - The variance measures the average degree to which each point differs from the mean.

# Observation:  1.varience for campaign_var_1 is 1.255903e+01
        #      2) varience for campaign_var_2 is 7.223273e+00
        #      3) varience for products purchased is 6.120610e-01
'''id                      2.283376e+08
campaign_var_1          1.255903e+01
campaign_var_2          7.223273e+00
products_purchased      6.120610e-01
user_activity_var_1     2.582420e-01
user_activity_var_2     6.057159e-03
user_activity_var_3     9.235431e-02
user_activity_var_4     1.026607e-02
user_activity_var_5     1.295521e-01
user_activity_var_6     3.121355e-01
user_activity_var_7     2.094626e-01
user_activity_var_8     1.388894e-01
user_activity_var_9     1.060351e-02
user_activity_var_10    3.437607e-04
user_activity_var_11    0.000000e+00
user_activity_var_12    4.773816e-04
buy                     3.671360e-02
Year                    1.884338e-01
Month                   1.391138e+01
Day                     1.391138e+01
Signup_Year             1.175274e+00
Signup_Month            1.297515e+01  '''
             
train_ad.std() # standard deviation - Standard deviation is the spread of a group of numbers from the mean.

# Observation:  1.standard deviation for campaign_var_1 is 3.543873
        #      2) standard deviation for campaign_var_2 is 2.687615
        #      3) standard deviation for products purchased is 0.78
'''id                      15110.844257
campaign_var_1              3.543873
campaign_var_2              2.687615
products_purchased          0.782343
user_activity_var_1         0.508175
user_activity_var_2         0.077828
user_activity_var_3         0.303899
user_activity_var_4         0.101322
user_activity_var_5         0.359933
user_activity_var_6         0.558691
user_activity_var_7         0.457671
user_activity_var_8         0.372679
user_activity_var_9         0.102973
user_activity_var_10        0.018541
user_activity_var_11        0.000000
user_activity_var_12        0.021849
buy                         0.191608
Year                        0.434090
Month                       3.729796
Day                         3.729796
Signup_Year                 1.084101
Signup_Month                3.602103  '''        
     
  # Note: While standard deviation is the square root of the variance, variance is the average of all data points within a group.
  
range = max(train_ad.campaign_var_1)-min(train_ad.campaign_var_1)
range #15

# # measure of skewess and kurtosis - 3rd & 4th business moment decisions

train_ad.skew() #3rd moment business decision - skewness - a long tail
# Skewness refers to a distortion or asymmetry that deviates from the symmetrical bell curve, or normal distribution, in a set of data. If the curve is shifted to the left or to the right, it is said to be skewed

# Observations:
    # campaign_var_1       0.396851     positively or right skewed
    # campaign_var_2       0.152701   - positively or right skewed
    # products_purchased   0.244595   - positively or right skewed
'''  user_activity_var_1      0.637745 - positively or right skewed
    user_activity_var_2     12.692765    positively or right skewed
    user_activity_var_3      2.613133   positively or right skewed
    user_activity_var_4      9.665168   positively or right skewed
    user_activity_var_5      1.963116   positively or right skewed
    user_activity_var_6      0.511466   positively or right skewed
    user_activity_var_7      0.968059   positively or right skewed
    user_activity_var_8      1.795581   positively or right skewed
    user_activity_var_9      9.573812   positively or right skewed
    user_activity_var_10    53.900115   positively or right skewed
    user_activity_var_11     0.000000   positively or right skewed
    user_activity_var_12    45.726582   positively or right skewed
    buy                      4.820759   positively or right skewed
    Year                     1.143276   positively or right skewed
    Month                    0.033496   positively or right skewed
    Day                      0.033496   positively or right skewed
    Signup_Year             -0.761400   negative or legt skewed
    Signup_Month             0.013779   positively or right skewed '''
    
train_ad.kurt() # 4th moment business decision- measure of tailedness of probability distribution
# Kurtosis is defined as the standardized fourth central moment of a distribution minus 3 (to make the kurtosis of the normal distribution equal to zero).
# standard normal distribution has kurtosis of 3 (Mesokurtic), 
# kurtosis >3 is  - leptokurtic, <3 is platykurtic

'''id                        -1.200000   platykurtic
campaign_var_1            -0.589848     platykurtic
campaign_var_2            -0.495512     platykurtic
products_purchased        -0.422727     platykurtic
user_activity_var_1       -1.084832     platykurtic
user_activity_var_2      159.112353     leptokurtic
user_activity_var_3        4.828648     leptokurtic
user_activity_var_4       91.418961     leptokurtic
user_activity_var_5        1.939553     platykurtic 
user_activity_var_6       -0.785573     platykurtic
user_activity_var_7       -0.925068     platykuric
user_activity_var_8        1.239976     platykurtic
user_activity_var_9       90.336191     leptokurtic
user_activity_var_10    2903.333333     leptokurtic
user_activity_var_11       0.000000     platykurtic   
user_activity_var_12    2089.000129     leptokurtic
buy                       21.240527     leptokurtic 
Year                      -0.692947     platykurtic
Month                     -1.418563     platykurtic
Day                       -1.418563     platykurtic
Signup_Year                0.630234     platykurtic
Signup_Month              -1.319810     platykurtic               '''
# --------------------------------

#Fifth moment business decision - Graphical representation

# Univariate analysis
# --------------------------------------------------
train_ad.columns

# histogram
train_ad.hist() # overall distribution of data
    
# count plot
ax = sns.countplot(data=train_ad,x="products_purchased")
# most no.of customers purchased 2 products.
ax = sns.countplot(data=train_ad,x="campaign_var_1")
# leads 4 to 7 have most no.of times.
ax = sns.countplot(data=train_ad,x="user_activity_var_1")
# most customers have zero activities & least customers have least activities. i.e. 2
ax = sns.countplot(data=train_ad,x="user_activity_var_2")
# most customers have zero activities
ax = sns.countplot(data=train_ad,x="buy")
# most customers not interested to buy
ax = sns.countplot(data=train_ad,x="Month")
# In 3rd month have more leads and less leads in 4th month.
ax = sns.countplot(data=train_ad,x="products_purchased", hue='Year')
# compared to 2022 in 2021 customers purchased more products.
ax = sns.countplot(data=train_ad,x="Month", hue='buy')
# most customers buying in august, septmeber, oactober, november & december.
ax = sns.countplot(data=train_ad,x="Signup_Month", hue='buy')
#most customers are signed up in march.
# ----------------

# barplot 

sns.barplot(x = train_ad.Year, y = train_ad.products_purchased)
# observation:
    #  .in 2021 more products are purchased.

sns.barplot(x = train_ad.Month, y = train_ad.campaign_var_1)
 # in march more campaign information or leads got.
sns.barplot(x = train_ad.Year, y = train_ad.campaign_var_1)
# in 2022 more campaign information or leads got than 2021. 
 
sns.barplot(x = train_ad.Signup_Month, y = train_ad.user_activity_var_1)
# in february, april, october months users have more activities on the website,
# =======================================================

# Correlation matrix
#correlation 
corr = train_ad.corr()
corr
# Set up the matplotlib plot configuration
f, ax = plt.subplots(figsize=(30, 10))
# Generate a mask for upper traingle
mask = np.triu(np.ones_like(corr, dtype=bool))
# Configure a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)
# Draw the heatmap
sns.heatmap(corr, annot=True, mask = mask, cmap=cmap)

# =====================================================
# Model Building

# splitting data as dependent and independent
X = train_ad.drop('buy', axis=1) # predictors
y = train_ad['buy']              # target

''' For imbalanced datasets Tree Based Algorithms like

Decision trees often perform well on imbalanced datasets because their hierarchical structure allows them to learn signals from both classes. In modern applied machine learning, tree ensembles like Random Forests, Gradient Boosted Trees, etc. almost always outperform singular decision trees.

Here Resampling and SMOTE is covered. '''


# Decision Tree Algorithm

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix, classification_report

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

from sklearn.tree import DecisionTreeClassifier

dec_tree = DecisionTreeClassifier(criterion="gini", max_depth=2)

dec_tree.fit(X_train,y_train)
y_train_pred = dec_tree.predict(X_train)
y_test_pred = dec_tree.predict(X_test)

print(f'Train score {accuracy_score(y_train_pred,y_train)}') # 96.5 %
print(f'Test score {accuracy_score(y_test_pred,y_test)}')  #  96.7 %

# Confusion Matrix
cm = confusion_matrix(y_test, y_test_pred)
cm

pd.crosstab(y_test, y_test_pred, rownames=['Actual'], colnames=['Predictions'])

# Classification Report
print(classification_report(y_test, y_test_pred)) # F1-SCORE for class-1 = 0.52

# ------------------------------------------------

# Random Forest

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=10) 
rfc.fit(X_train, y_train)

y_train_predict = dec_tree.predict(X_train)
y_test_predict = dec_tree.predict(X_test)

print(f'Train score {accuracy_score(y_train_predict, y_train)}') # 96.5 %
print(f'Test score {accuracy_score(y_test_predict, y_test)}')  #  96.7 %


# Confusion Matrix
cm = confusion_matrix(y_test, y_test_predict)
cm

pd.crosstab(y_test, y_test_predict, rownames=['Actual'], colnames=['Predictions'])

# Classification Report
print(classification_report(y_test, y_test_predict)) # F1-SCORE for class-1 = 0.52

# ------------------------------------------------                        
# XGBoost Classifier
from xgboost import XGBClassifier

xgb = XGBClassifier(n_estimators=100)
xgb.fit(X_train, y_train)
         
predict_train = xgb.predict(X_train)
predict_test = xgb.predict(X_test)

print(f'Train score {accuracy_score(predict_train, y_train)}') # 97.4 %
print(f'Test score {accuracy_score(predict_test, y_test)}')  #  97.2 %

# Classification Report
print(classification_report(y_test, predict_test)) # F1-SCORE for class-1 is 0.63



# -------------------

# Solution or validation test data
sample_validation = pd.read_csv(r'Downloads\test_ad.csv')
sample_validation.head()
sample_validation.info()

# create the inputs and outputs

# handling date columns by splitting as 3 columns day, month, year
sample_validation['Year'] = sample_validation['created_at'].str.split('-').str[0].astype(int)
sample_validation['Month'] = sample_validation['created_at'].str.split('-').str[1].astype(int)
sample_validation['Day'] = sample_validation['created_at'].str.split('-').str[1].astype(int)

sample_validation.drop('created_at', axis=1, inplace=True)

# fill null values with backward fill
sample_validation = sample_validation.fillna(method='ffill') 
sample_validation = sample_validation.fillna(method='bfill') 

# checking null values still exist or not
sample_validation.isnull().sum() # no na values


sample_validation['Signup_Year'] = sample_validation['signup_date'].str.split('-').str[0].astype(int)
sample_validation['Signup_Month'] = sample_validation['signup_date'].str.split('-').str[1].astype(int)
sample_validation['Signup_Day'] = sample_validation['signup_date'].str.split('-').str[1].astype(int)

sample_validation.drop('signup_date', axis=1, inplace=True)

# make predictions on the entire training dataset
yhat = rfc.predict(sample_validation)
print(yhat)
