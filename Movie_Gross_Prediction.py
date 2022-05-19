import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("movie_metadata.csv")

df.head()

df = df[["country", "budget", "genres", "language","director_name" , "gross"]]

df.head()

df = df.dropna()
df.isnull().sum()
df.head()

df1 = df['genres'].str.split('|', expand=True)
df['uniquecount'] = df1.count(axis=1)

df.head()
