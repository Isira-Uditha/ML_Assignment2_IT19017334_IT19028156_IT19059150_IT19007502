import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

df = pd.read_csv("movie_metadata.csv")

df.head()

df = df[["country", "budget", "genres", "language", "imdb_score" , "gross"]]

df.head()

# remove raws with null values
df = df.dropna()
df.isnull().sum()
df.head()

# split the genres types an assing a rating
df_tepm = df['genres'].str.split('|', expand=True)
df['genres_def'] = df_tepm.count(axis=1)

# remove the genres colomn
df = df.drop('genres', axis=1)

# len(df['country'].unique())

def cutoff_countries(countries, limit):
    country_map = {}
    for i in range(len(countries)):
        if countries.values[i] < limit:
            country_map[countries.index[i]] = 'Other'
        else:
            country_map[countries.index[i]] = countries.index[i]

    return country_map

country_map = cutoff_countries(df.country.value_counts(), 5)
df['country'] = df['country'].map(country_map)
df.country.value_counts()

df['budget'].max()
df['budget'].min()
df = df[df['country'] != 'Other']

df['imdb_score'] = df['imdb_score'].round(decimals=0)

country = LabelEncoder()
df['country'] = country.fit_transform(df['country'])
df["country"].unique()

language = LabelEncoder()
df['language'] = language.fit_transform(df['language'])
df["language"].unique()

X = df.drop("gross", axis=1)
y = df["gross"]

train_features, test_features, train_labels, test_labels = train_test_split(X, y, test_size = 0.25, random_state = 0)

#LinearRegression
linear_reg = LinearRegression()
linear_reg.fit(train_features, train_labels)
y_pred = linear_reg.predict(test_features)
error = np.sqrt(mean_squared_error(test_labels, y_pred))
print("${:,.02f}".format(error))

#DecisionTreeRegressor
dec_tree_reg = DecisionTreeRegressor(random_state=0)
dec_tree_reg.fit(X, y.values)
y_pred = dec_tree_reg.predict(X)
error = np.sqrt(mean_squared_error(y, y_pred))
print("${:,.02f}".format(error))

#RandomForestRegressor
random_forest_reg = RandomForestRegressor(random_state=0)
random_forest_reg.fit(X, y.values)
y_pred = random_forest_reg.predict(X)
error = np.sqrt(mean_squared_error(y, y_pred))
print("${:,.02f}".format(error))

max_depth = [None, 2,4,6,8,10,12]
parameters = {"max_depth": max_depth}

regressor = DecisionTreeRegressor(random_state=0)
gs = GridSearchCV(regressor, parameters, scoring='neg_mean_squared_error')
gs.fit(X, y.values)

regressor = gs.best_estimator_

regressor.fit(X, y.values)
y_pred = regressor.predict(X)
error = np.sqrt(mean_squared_error(y, y_pred))
print("${:,.02f}".format(error))

classifier = LogisticRegression(random_state= 0)
classifier.fit(train_features, train_labels)
y_pred = classifier.predict(test_features)
error = np.sqrt(mean_squared_error(test_labels, y_pred))
print("${:,.02f}".format(error))

accuracy = accuracy_score(test_labels, y_pred)

X = np.array([["USA", 2.37e+08, 'English', 8, 7]])
X

X[:, 0] = country.transform(X[:,0])
X[:, 2] = language.transform(X[:,2])
X = X.astype(float)
X

y_pred = regressor.predict(X)
y_pred

df.head()
