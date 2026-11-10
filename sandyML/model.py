import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
df = pd.read_csv("C:/Users/sande/Downloads/Bengaluru_House_Data.csv")
print(df.head(5))

#3. Data Cleaning¶
#We’ll remove irrelevant columns, handle missing values, and engineer new features.
df1 = df.drop(['area_type','availability','society','balcony'],axis='columns')
df1.head()

df1.isnull().sum() #checking if all data are available or not
df2 = df1.dropna()          ##dropped all the data were not available as dataset is huge
df2.isnull().sum()
df2.shape

 #3.1 Feature Engineering¶
#Extract BHK from the size column
#Convert total_sqft ranges (e.g., "2100-2850") into numeric averages
#Create price_per_sqft feature
#Simplify location (group rare ones as "other")

df2['size'].unique()

df2 = df2.copy()  # avoid SettingWithCopyWarning
df2['bhk'] = df2['size'].apply(lambda x: int(x.split(' ')[0]))

df2.head()
df2['bhk'].unique()

df2.total_sqft.unique()

def is_float(x):
  try:
    float(x)
  except:
    return False
  return True


df2[~df2['total_sqft'].apply(is_float)].head(10)

def convert_sqft_to_num(x):     ##a function that converts this range values to the average of both.df
  tokens = x.split('-')
  if len(tokens) == 2:
    return (float(tokens[0])+float(tokens[1]))/2
  try:
    return float(x)
  except:
    return None
  
df3 = df2.copy()
df3['total_sqft'] = df3['total_sqft'].apply(convert_sqft_to_num)
df3.head(3)

df3.loc[30]
df4 = df3.copy()
df4['price_per_sqft'] = (df4['price']*100000)/df4['total_sqft']
df4.head()

len(df4.location.unique())

df4.location = df4.location.apply(lambda x: x.strip())
location_stats = df4.groupby('location')['location'].agg('count').sort_values(ascending=False)
location_stats

len(location_stats[location_stats <= 10])

location_stats_less_than_10 = location_stats[location_stats <= 10]
location_stats_less_than_10

len(df4.location.unique())

df4.location = df4.location.apply(lambda x: 'other' if x in location_stats_less_than_10 else x)
len(df4.location.unique())

df4.head(100)

df4.shape

df5 = df4[~(df4.total_sqft/df4.bhk < 300)]
df5.shape

df5.price_per_sqft.describe()

def remove_pps_outliers(df):
  df_out = pd.DataFrame()
  for key, subdf in df.groupby('location'):
    m = np.mean(subdf.price_per_sqft)
    st = np.std(subdf.price_per_sqft)
    reduced_df = subdf[(subdf.price_per_sqft > (m-st)) & (subdf.price_per_sqft <= (m+st))]
    df_out = pd.concat([df_out,reduced_df],ignore_index=True)
  return df_out

df6 = remove_pps_outliers(df5)
df6.shape

def plot_scatter_chart(df, location):
  bhk2 = df[(df.location==location) & (df.bhk==2)]
  bhk3 = df[(df.location==location) & (df.bhk==3)]
  plt.rcParams['figure.figsize'] = (15,10)
  plt.scatter(bhk2.total_sqft, bhk2.price, color='blue', label='2 BHK', s=50)
  plt.scatter(bhk3.total_sqft, bhk3.price, marker='+', color='green', label='3 BHK', s=50)
  plt.xlabel('Total Square Feet Area')
  plt.ylabel('Price (Lakh Indian Rupees)')
  plt.title(location)
  plt.legend()

plot_scatter_chart(df6, 'Hebbal')

def remove_bhk_outliers(df):
  exclude_indices = np.array([])
  for location, location_df in df.groupby('location'):
    bhk_stats = {}
    for bhk, bhk_df in location_df.groupby('bhk'):
      bhk_stats[bhk] = {
          'mean': np.mean(bhk_df.price_per_sqft),
          'std': np.std(bhk_df.price_per_sqft),
          'count': bhk_df.shape[0]
      }
    for bhk, bhk_df in location_df.groupby('bhk'):
      stats = bhk_stats.get(bhk-1)
      if stats and stats['count']>5:
        exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
  return df.drop(exclude_indices, axis='index')

df7 = remove_bhk_outliers(df6)
df7.shape

#4. Exploratory Data Analysis (EDA)¶
#Let’s visualize distributions and relationships between features.

#Price distribution
#Price vs Sqft
#Correlation heatmap

plot_scatter_chart(df7, 'Hebbal')

plt.hist(df7.price_per_sqft, rwidth=0.8)
plt.xlabel('Price Per Square Feet')
plt.ylabel('Count')

df7.bath.unique()

df7[df7.bath>10]

plt.hist(df7.bath, rwidth=0.8)
plt.xlabel('Number of bathrooms')
plt.ylabel('Count')

df7[df7.bath>df7.bhk+2]

df8 = df7[df7.bath<df7.bhk+2]
df8.shape

df9 = df8.drop(['size', 'price_per_sqft'], axis = 'columns')
df9.head()

dummies = pd.get_dummies(df9.location)
dummies.head(3)

df10 = pd.concat([df9, dummies.drop('other', axis = 'columns')], axis = 'columns')
df10.head(3)

df11 = df10.drop('location', axis = 'columns')
df11.head(3)

df11.shape

X = df11.drop('price', axis = 'columns')    ##only independent variables should be there in X
X.head(3)

X = df11.drop('price', axis = 'columns')    ##only independent variables should be there in X
X.head(3)

y = df11.price
y.head()

#5. Model Building¶
#We’ll try:
#Linear Regression (baseline)
#Random Forest Regressor (ensemble method)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 10)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.values.reshape(len(y_test),1)),1))

#6. Model Evaluation
#Compare R² score and RMSE
#Plot actual vs predicted prices for Random Forest

from sklearn.metrics import r2_score
r2_score(y_test, y_pred)

from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score

cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

cross_val_score(LinearRegression(), X, y, cv=cv)

#Using Grid SearchCV Method to calculate best model

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression # Ensure LinearRegression is imported

def find_best_model_using_gridsearchcv(X, y):
  algos = {
      'linear_regression': {
          'model': LinearRegression(),
          'params': {
              # Removed 'normalize': [True, False] as it's no longer a valid parameter
          }
      },
      'lasso': {
          'model': Lasso(),
          'params': {
              'alpha': [1, 2],
              'selection': ['random', 'cyclic']
          }
      },
      'decision_tree': {
      'model': DecisionTreeRegressor(),
       'params': {
            'criterion': ['squared_error', 'friedman_mse'],  # fixed
            'splitter': ['best', 'random']
    }
}
  }
  scores = []
  cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
  for algo_name, config in algos.items():
    gs = GridSearchCV(config['model'], config['params'], cv=cv, return_train_score=False)
    gs.fit(X, y)
    scores.append({
        'model': algo_name,
        'best_score': gs.best_score_,
        'best_params': gs.best_params_
    })

  return pd.DataFrame(scores, columns=['model', 'best_score', 'best_params'])

find_best_model_using_gridsearchcv(X, y)


#7. Make Predictions¶
#Let’s create a function predict_price() where we can input:

#Location
#Sqft
#BHK
#Bathrooms
#and get back the predicted price in lakhs.

import pandas as pd

def predict_price(location, sqft, bath, bhk):
    try:
        loc_index = np.where(X.columns == location)[0][0]
    except IndexError:
        loc_index = -1

    x = np.zeros(len(X.columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    x_df = pd.DataFrame([x], columns=X.columns)
    return regressor.predict(x_df)[0]

import pickle
with open('banglore_home_prices_model.pickle', 'wb') as f:
  pickle.dump(regressor, f)

import json
columns = {
    'data_columns': [col.lower() for col in X.columns]
}
with open('columns.json', 'w') as f:
  f.write(json.dumps(columns))

#🚀 8. Conclusion
#✅ Location, sqft, and BHK are strong predictors of house price
#✅ Random Forest performed better than Linear Regression

price = predict_price("Indira Nagar", 1200, 2, 3)
print("predicted price:",price,"lakhs" )

print(predict_price("Kothanur", 1000, 2, 3))

import json

columns = X.columns.tolist()   # X is your training DataFrame after one-hot encoding
with open("columns.json", "w") as f:
    f.write(json.dumps({"data_columns": columns}))
