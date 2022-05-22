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
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import max_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
import seaborn as sns

df = pd.read_csv("movie_metadata.csv")

df = df[["num_critic_for_reviews", "duration", "director_facebook_likes", "actor_3_facebook_likes","actor_1_facebook_likes" ,"num_voted_users","cast_total_facebook_likes",
         "facenumber_in_poster", "num_user_for_reviews", "language", "country", "budget", "actor_2_facebook_likes", "imdb_score","aspect_ratio", "movie_facebook_likes", "gross"]]

print('Rows     :',df.shape[0])
print('Columns  :',df.shape[1])
print('\nFeatures :\n     :',df.columns.tolist())
print('\nMissing values    :',df.isnull().values.sum())
print('\nMissing values - Country    :',df['country'].isnull().values.sum())
print('\nMissing values - language    :',df['language'].isnull().values.sum())
print('\nMissing values - budget    :',df['budget'].isnull().values.sum())
print('\nMissing values - aspect_ratio    :',df['aspect_ratio'].isnull().values.sum())
print('\nUnique values : \n',df.nunique())





df = df.dropna(subset=['country','language'],inplace=False)
#df = df.loc[:,~df.columns.duplicated()]

country = LabelEncoder()
df['country'] = country.fit_transform(df['country'])
df["country"].unique()

language = LabelEncoder()
df['language'] = language.fit_transform(df['language'])
df["language"].unique()

### Most Important Features ####
correlations = df.corr()
f,ax = plt.subplots(figsize=(15,15))
sns.heatmap(correlations, annot=True, cmap="YlGnBu", linewidths=.5)

df = df.drop(['actor_3_facebook_likes','cast_total_facebook_likes','facenumber_in_poster','language','aspect_ratio'], axis = 1)

### Pre Processing

def cutoff_countries(countries, limit):
   country_map = {}
   for i in range(len(countries)):
       if countries.values[i] < limit:
           country_map[countries.index[i]] = 'Other'
       else:
           country_map[countries.index[i]] = countries.index[i]
   return country_map
       
country_map = cutoff_countries(df.country.value_counts(), 80)
df['country'] = df['country'].map(country_map)
df.country.value_counts()
df = df[df['country'] != 'Other']

# Filling missing values
df['gross'] = df['gross'].fillna(df['gross'].mean(), inplace=False)
df['num_user_for_reviews'] = df['num_user_for_reviews'].fillna(df['num_user_for_reviews'].mean(), inplace=False)
df['imdb_score'] = df['imdb_score'].fillna(df['imdb_score'].mean(), inplace=False)
df['budget'] = df['budget'].fillna(df['budget'].mean(), inplace=False)
df['num_voted_users'] = df['num_voted_users'].fillna(df['num_voted_users'].mean(), inplace=False)
df['num_critic_for_reviews'] = df['num_critic_for_reviews'].fillna(df['num_critic_for_reviews'].mean(), inplace=False)
df['movie_facebook_likes'] = df['movie_facebook_likes'].fillna(df['movie_facebook_likes'].mean(), inplace=False)
df['actor_1_facebook_likes'] = df['actor_1_facebook_likes'].fillna(df['actor_1_facebook_likes'].mean(), inplace=False)
df['actor_2_facebook_likes'] = df['actor_2_facebook_likes'].fillna(df['actor_2_facebook_likes'].mean(), inplace=False)
df['duration'] = df['duration'].fillna(df['duration'].mean(), inplace=False)
df['director_facebook_likes'] = df['director_facebook_likes'].fillna(df['director_facebook_likes'].mean(), inplace=False)