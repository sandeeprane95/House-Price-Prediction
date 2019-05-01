
# coding: utf-8

# Import Packages

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, median_absolute_error


# Read dataset

# In[2]:


trainData = 'D:\\AppliedAI\\house-prices-advanced-regression-techniques\\train.csv'
testData = 'D:\\AppliedAI\\house-prices-advanced-regression-techniques\\test.csv'


# In[3]:


train_df = pd.read_csv(trainData)
test_df = pd.read_csv(testData)
total_df = pd.concat((train_df, test_df))
total_df = total_df.reset_index(drop=True)
total_df.drop(['SalePrice'], axis=1, inplace=True)

y = train_df['SalePrice']


# In[4]:


train_df.columns


# Compute Correlation Matrix
# 

# In[5]:


corrMatrix = train_df.corr()
plt.figure(figsize=(12,12))
sns.heatmap(corrMatrix,annot=False,cmap="RdYlGn")


# Columns with missing values
# 

# In[6]:


def checkForMissing(dataFrame):
    mCol = [col for col in dataFrame.columns if dataFrame[col].isnull().any()]
    mCol_with_sum = (dataFrame.isnull().sum())
    print(mCol_with_sum[mCol_with_sum > 0])


# Handling the missing values
# 

# In[7]:


def handleMissing(dataFrame):    
    lot_frontage = list(dataFrame['LotFrontage'].unique())
    dataFrame['LotFrontage'].fillna(0, inplace = True)
    #dataFrame['LotFrontage'].isnull().any()

    alley = list(dataFrame['Alley'].unique())
    dataFrame['Alley'].fillna('None', inplace = True)
    #dataFrame['Alley'].isnull().any()

    masvnr_type = list(dataFrame['MasVnrType'].unique())
    dataFrame['MasVnrType'].fillna('None', inplace = True)
    #dataFrame['MasVnrType'].isnull().any()

    masvnr_area = list(dataFrame['MasVnrArea'].unique())
    dataFrame['MasVnrArea'].fillna(0, inplace = True)
    #dataFrame['MasVnrArea'].isnull().any()

    bsmt_qual = list(dataFrame['BsmtQual'].unique())
    dataFrame['BsmtQual'].fillna('None', inplace = True)
    #dataFrame['BsmtQual'].isnull().any()

    bsmt_cond = list(dataFrame['BsmtCond'].unique())
    dataFrame['BsmtCond'].fillna('None', inplace = True)
    #dataFrame['BsmtCond'].isnull().any()

    bsmt_exposure = list(dataFrame['BsmtExposure'].unique())
    dataFrame['BsmtExposure'].fillna('None', inplace = True)
    #dataFrame['BsmtExposure'].isnull().any()

    bsmt_fint1 = list(dataFrame['BsmtFinType1'].unique())
    dataFrame['BsmtFinType1'].fillna('None', inplace = True)
    #dataFrame['BsmtFinType1'].isnull().any()

    bsmt_fint2 = list(dataFrame['BsmtFinType2'].unique())
    dataFrame['BsmtFinType2'].fillna('None', inplace = True)
    #dataFrame['BsmtFinType2'].isnull().any()

    electrical = list(dataFrame['Electrical'].unique())
    #dataFrame['Electrical'].value_counts()
    dataFrame['Electrical'].fillna(train_df['Electrical'].mode()[0], inplace = True)
    #dataFrame['Electrical'].isnull().any()

    fireplace_Qu = list(dataFrame['FireplaceQu'].unique())
    dataFrame['FireplaceQu'].fillna('None', inplace = True)
    #dataFrame['FireplaceQu'].isnull().any()

    garage_type = list(dataFrame['GarageType'].unique())
    dataFrame['GarageType'].fillna('None', inplace = True)
    #dataFrame['GarageType'].isnull().any()

    garage_yrblt = list(dataFrame['GarageYrBlt'].unique())
    dataFrame['GarageYrBlt'].fillna(0, inplace = True)
    #dataFrame['GarageYrBlt'].isnull().any()

    garage_cars = list(dataFrame['GarageCars'].unique())
    dataFrame['GarageCars'].fillna(0, inplace = True)
    #dataFrame['GarageCars'].isnull().any()
    
    garage_area = list(dataFrame['GarageArea'].unique())
    dataFrame['GarageArea'].fillna(0, inplace = True)
    #dataFrame['GarageArea'].isnull().any()
    
    garage_fin = list(dataFrame['GarageFinish'].unique())
    dataFrame['GarageFinish'].fillna('None', inplace = True)
    #dataFrame['GarageFinish'].isnull().any()

    garage_qual = list(dataFrame['GarageQual'].unique())
    dataFrame['GarageQual'].fillna('None', inplace = True)
    #dataFrame['GarageQual'].isnull().any()

    garage_cond = list(dataFrame['GarageCond'].unique())
    dataFrame['GarageCond'].fillna('None', inplace = True)
    #dataFrame['GarageCond'].isnull().any()

    pool_qc = list(dataFrame['PoolQC'].unique())
    dataFrame['PoolQC'].fillna('None', inplace = True)
    #dataFrame['PoolQC'].isnull().any()

    fence = list(dataFrame['Fence'].unique())
    dataFrame['Fence'].fillna('None', inplace = True)
    #dataFrame['Fence'].isnull().any()

    misc_feature = list(dataFrame['MiscFeature'].unique())
    dataFrame['MiscFeature'].fillna('None', inplace = True)
    #dataFrame['MiscFeature'].isnull().any()

    ms_zoning = list(dataFrame['MSZoning'].unique())
    dataFrame['MSZoning'].fillna(train_df['MSZoning'].mode()[0], inplace = True)
    #dataFrame['MSZoning'].isnull().any()

    #Drop utilities as we dont have enough samples for the column
    utilities = list(dataFrame['Utilities'].unique())
    dataFrame.drop(['Utilities'], axis=1, inplace=True)

    exterior_1 = list(dataFrame['Exterior1st'].unique())
    dataFrame['Exterior1st'].fillna(train_df['Exterior1st'].mode()[0], inplace = True)
    #dataFrame['Exterior1st'].isnull().any()

    exterior_2 = list(dataFrame['Exterior2nd'].unique())
    dataFrame['Exterior2nd'].fillna(train_df['Exterior2nd'].mode()[0], inplace = True)
    #dataFrame['Exterior2nd'].isnull().any()

    bsmt_finsf1 = list(dataFrame['BsmtFinSF1'].unique())
    dataFrame['BsmtFinSF1'].fillna(0, inplace = True)
    #dataFrame['BsmtFinSF1'].isnull().any()
    
    bsmt_finsf2 = list(dataFrame['BsmtFinSF2'].unique())
    dataFrame['BsmtFinSF2'].fillna(0, inplace = True)
    #dataFrame['BsmtFinSF2'].isnull().any()
    
    bsmt_unfsf = list(dataFrame['BsmtUnfSF'].unique())
    dataFrame['BsmtUnfSF'].fillna(0, inplace = True)
    #dataFrame['BsmtUnfSF'].isnull().any()
    
    totalbsmt_sf = list(dataFrame['TotalBsmtSF'].unique())
    dataFrame['TotalBsmtSF'].fillna(0, inplace = True)
    #dataFrame['TotalBsmtSF'].isnull().any()
    
    bsmt_fullbath = list(dataFrame['BsmtFullBath'].unique())
    dataFrame['BsmtFullBath'].fillna(0, inplace = True)
    #dataFrame['BsmtFullBath'].isnull().any()
    
    bsmt_halfbath = list(dataFrame['BsmtHalfBath'].unique())
    dataFrame['BsmtHalfBath'].fillna(0, inplace = True)
    #dataFrame['BsmtHalfBath'].isnull().any()
    
    functional = list(dataFrame['Functional'].unique())
    dataFrame['Functional'].fillna('Typ', inplace = True)
    #dataFrame['Functional'].isnull().any()
    
    sale_type = list(dataFrame['SaleType'].unique())
    dataFrame['SaleType'].fillna(train_df['SaleType'].mode()[0], inplace = True)
    #dataFrame['SaleType'].isnull().any()
    
    sale_type = list(dataFrame['KitchenQual'].unique())
    dataFrame['KitchenQual'].fillna(train_df['KitchenQual'].mode()[0], inplace = True)
    #dataFrame['KitchenQual'].isnull().any()


# In[8]:


handleMissing(train_df)
handleMissing(test_df)


# In[9]:


checkForMissing(train_df)
checkForMissing(test_df)


# Add new column for Total Area and remove related columns
# 

# In[10]:


train_df['HouseSF'] = train_df['TotalBsmtSF'] + train_df['1stFlrSF'] + train_df['2ndFlrSF']
train_df.drop(['TotalBsmtSF', '1stFlrSF', '2ndFlrSF'], axis=1, inplace=True)

test_df['HouseSF'] = test_df['TotalBsmtSF'] + test_df['1stFlrSF'] + test_df['2ndFlrSF']
test_df.drop(['TotalBsmtSF', '1stFlrSF', '2ndFlrSF'], axis=1, inplace=True)


# Handling the categorical values
# 

# In[11]:


# The MSSubClass Feature is given to be numeric but is actually categorical
train_df['MSSubClass'] = train_df['MSSubClass'].astype(str)
test_df['MSSubClass'] = test_df['MSSubClass'].astype(str)

train_df.drop(['SalePrice','Id'], axis=1, inplace=True)
testID = test_df['Id']
test_df.drop(['Id'], axis=1, inplace=True)


# Label-Encoding was used

# In[12]:


label_encode = LabelEncoder()
attributes = train_df.select_dtypes(include=['object']) 
for attr in attributes:
    train_df[attr] = label_encode.fit_transform(train_df[attr])
    test_df[attr] = label_encode.fit_transform(test_df[attr])


# One-hot encoding (not used)

# In[ ]:


col = train_df.dtypes
train_df = pd.get_dummies(train_df, columns=col[col == 'object'].index.values, drop_first=True)
test_df = pd.get_dummies(test_df, columns=col[col == 'object'].index.values, drop_first=True)


# In[ ]:


col_types = train_df.dtypes
unique_count = train_df.nunique()
unique_count[col_types[col_types == 'object'].index].sort_values(ascending = False)


# In[ ]:


x = train_df


# MinMaxScaler

# In[13]:


def scaler(dataFrame): 
    min_max_scaler = MinMaxScaler()
    return pd.DataFrame(data=min_max_scaler.fit_transform(dataFrame.values), columns=dataFrame.columns.values)


# In[14]:


x = scaler(train_df)
test_df = scaler(test_df)


# Decision Tree Regressor

# In[15]:


train_X, val_X, train_y, val_y = train_test_split(x, y, test_size = 0.33, random_state = 42)
melbourne_model = DecisionTreeRegressor(random_state=0)
melbourne_model.fit(train_X, np.log1p(train_y))
val_predictions = melbourne_model.predict(val_X)
val_predictions = np.expm1(val_predictions)
median_absolute_error(val_y, val_predictions)


# Random Forest Regressor

# In[16]:


forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(train_X, np.log1p(train_y))
melb_preds = forest_model.predict(val_X)
melb_preds = np.expm1(melb_preds)
median_absolute_error(val_y, melb_preds)
#forest_model.get_params()


# K-Fold Cross Validation

# In[17]:


kf = KFold(n_splits=10, shuffle=True, random_state=42)
rmse = 0
mae = 0

for i, j in kf.split(x):
    sample_tr_x, sample_test_x = x.values[i], x.values[j]
    sample_tr_y, sample_test_y = y.values[i], y.values[j]      
    
    model = RandomForestRegressor(random_state=1)   
    model.fit(sample_tr_x, sample_tr_y)
    predicted_val = model.predict(sample_test_x)
    
    #rmse = rmse + np.sqrt(mean_squared_error(sample_test_y, np.expm1(predicted_val)))
    mae += median_absolute_error(sample_test_y, predicted_val) 
    
#rmse/5
mae/10


# Adjust columns of test set as per the train set (For one-hot encoding)

# In[ ]:


for c in train_df.columns:
    if c not in test_df.columns:
            test_df[c] = 0

col_list = []
for c in test_df.columns:
    if c not in train_df.columns:
        col_list.append(c)

test_df.drop(c, axis=1, inplace=True)


# Final Output Write

# In[18]:


model = RandomForestRegressor(random_state=1)  
model.fit(x, y)
predicted_val = model.predict(test_df)

sub = pd.DataFrame()
sub['Id'] = testID
sub['SalePrice'] = predicted_val
sub.to_csv('D:\\AppliedAI\\house-prices-advanced-regression-techniques\\submission.csv',index=False)


# Support Vector Regressor

# In[19]:


from sklearn.svm import SVR
cat_model = SVR(C = 10, kernel = 'rbf')
cat_model.fit(train_X, np.log1p(train_y))
melb_preds = cat_model.predict(val_X)
melb_preds = np.expm1(melb_preds)
median_absolute_error(val_y, melb_preds)


# GridSearchCV with Kernel Ridge Regressor

# In[ ]:


from sklearn import svm, grid_search
from sklearn.model_selection import GridSearchCV
def svc_param_selection(X, y, nfolds):
    a = [0.4, 0.5, 0.6, 0.7]
    c = [1, 1.5, 2, 2.5, 3]
    d = [1,2,3]
    param_grid = {'alpha': a, 'coef0':c, 'degree':d}
    grid_search = GridSearchCV(KernelRidge(kernel='polynomial'), param_grid, cv=nfolds)
    grid_search.fit(X, y)
    grid_search.best_params_
    return grid_search.best_params_
svc_param_selection(train_X,train_y,5)

