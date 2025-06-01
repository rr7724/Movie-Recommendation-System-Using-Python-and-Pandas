#!/usr/bin/env python
# coding: utf-8

# # Movie Recommender System
# 
# In this iPython Notebook, I have created basic Movie Recommender System with Python.
# 
# It is an extension from the project: https://github.com/krishnaik06/Movie-Recommender-in-python
# The Dataset used is subset of MovieLens Dataset
# 
# Extension: Have created a text input bar to add your movie whose recommendation you want. Output will give you top 4 matches that are recommended movies.
# 

# In[181]:


#Import all necessary libraries
import os
os.system('pip install matplotlib')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.html.widgets import *
get_ipython().run_line_magic('matplotlib', 'inline')


# In[158]:


#Get the data into Pandas Dataframe object
column_names = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv('dataset.csv', sep = '\t', names = column_names)
df.head()


# In[159]:


#Get the Movie Titles
movie_titles = pd.read_csv('movieIdTitles.csv')
movie_titles.head()


# In[160]:


#Merge the dataset with movie titles
df = pd.merge(df, movie_titles, on = 'item_id')
df.head()


# ### Do some Exploratory Data Analysis

# In[161]:


df.groupby('title')['rating'].mean().sort_values(ascending = False).head()


# In[162]:


df.groupby('title')['rating'].count().sort_values(ascending = False).head()


# In[163]:


ratings = pd.DataFrame(df.groupby('title')['rating'].mean())
ratings.head()


# In[164]:


ratings['numOfRatings'] = pd.DataFrame(df.groupby('title')['rating'].count())
ratings.head()


# In[165]:


plt.figure(figsize = (10,4))
ratings['numOfRatings'].hist(bins = 70)


# In[166]:


plt.figure(figsize = (10,4))
ratings['rating'].hist(bins = 70)


# In[167]:


sns.jointplot(x='rating', y='numOfRatings', data = ratings, alpha = 0.5)


# ### Create the Recommendation System

# Create a matrix that has the user ids on one access and the movie title on another axis. Each cell will then consist of the rating the user gave to that movie. Note there will be a lot of NaN values, because most people have not seen most of the movies.

# In[168]:


moviemat = df.pivot_table(index='user_id',columns='title',values='rating')
moviemat.head()


# In[169]:


#Most Rated Movies with their Average Ratings
ratings.sort_values('numOfRatings', ascending = False).head(10)


# Now we will create a correlation matrix of every movie with every other movie on user ratings. We will then use that correlation matrix to find top matches that relates the best for a particular movie (having atleast 100 ratings) and the result obtained (recommended movies) will then be added to the ratings dataframe of every movie. Those whose matches could not be obtained using correlation, their value will be converted to "-".

# In[170]:


for i in ratings.index:
    movieUserRatings = moviemat[i]
    similarToThatMovie = moviemat.corrwith(movieUserRatings)
    corr_toMovie = pd.DataFrame(similarToThatMovie, columns = ['Correlation'])
    corr_toMovie.dropna(inplace = True)
    corr_toMovie = corr_toMovie.join(ratings['numOfRatings'])
    result = corr_toMovie[corr_toMovie['numOfRatings'] > 100].sort_values('Correlation', ascending = False).head()
    if result['numOfRatings'].count() >= 5:
        print(i)
        ratings.loc[i, 'FirstMovieRecommendation'] = result.iloc[1:2].index.values[0]
        ratings.loc[i, 'SecondMovieRecommendation'] = result.iloc[2:3].index.values[0]
        ratings.loc[i, 'ThirdMovieRecommendation'] = result.iloc[3:4].index.values[0]
        ratings.loc[i, 'FourthMovieRecommendation'] = result.iloc[4:5].index.values[0]
    


# In[195]:


#Check the result 
ratings.head()


# In[198]:


ratings = ratings.fillna('-')


# In[199]:


#Save the ratings data for later use
ratings.to_csv('MovieRecommendations.csv', encoding='utf-8')


# # Load the Saved Recommendation Data Generated for Reusability

# In[200]:


#Load the dataset saved for reusability from this code block onwards
df_result = pd.read_csv('MovieRecommendations.csv')
df_result.head()


# In[201]:


#Load all the movie names
for i in df_result['title']:
    print(i)


# In[236]:


inputMovieName = widgets.Text()

def getRecommendations(sender):
    searchMovie = inputMovieName.value
    list_result = df_result[df_result['title'] == searchMovie]
    fm = list_result['FirstMovieRecommendation'].values[0]
    sm = list_result['SecondMovieRecommendation'].values[0]
    tm = list_result['ThirdMovieRecommendation'].values[0]
    fourthm = list_result['FourthMovieRecommendation'].values[0]
    finalRecommendationText = '1:' + fm + ' \n2:' + sm + ' \n3:' + tm + ' \n4:' + fourthm
    print('Your Recommendations for the Movie ' + searchMovie + ' are:\n')
    print(finalRecommendationText)
    


# ### How to get Recommendations?
# - Select and Copy any movie from the list of Movie Names above
# - Add that to the text box below
# ##### You will have your Movie Recommendation for that Particular movie :)
# 
# Note:- On every run the paste command will keep on appending the current output. To clear the output just run the below cell again.

# In[241]:


inputMovieName.on_submit(getRecommendations)
inputMovieName


# In[ ]:




