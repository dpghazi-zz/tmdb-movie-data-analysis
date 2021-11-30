#!/usr/bin/env python
# coding: utf-8

# # Project: TMDb Movie Data Analysis
# 
# ## Table of Contents
# <ul>
# <li><a href="#intro">Introduction</a></li>
# <li><a href="#wrangling">Data Wrangling</a></li>
# <li><a href="#eda">Exploratory Data Analysis</a></li>
# <li><a href="#conclusions">Conclusions</a></li>
# </ul>

# <a id='intro'></a>
# ## Introduction
# 
# > **Note**: This project was originally introduced on [Kaggle](https://www.kaggle.com/tmdb/tmdb-movie-metadata)
# >
# > - This data set contains information about 10,000 movies collected from The Movie Database (TMDb), including user ratings and revenue.
# > - The primary goal of this project is to practice my pandas, Numpy, and Matplotlib data analysis techniques.
# > - The investigation process is divided into four different parts.
#                 1) Possible Questions
#                 2) Data Wrangling and Data Cleaning
#                 3) Exploratory Data Analysis
#                 4) Conclusions
#             
# > - Objectives of Investigation
#                 1) Which genres are most popular from year to year?
#                 2) How did film budgets change from each decade?
#                 3) What kinds of properties are associated with movies that have high revenues?

# In[215]:


# import packages 
import pandas as pd
import numpy as np
from numpy import mean
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# <a id='wrangling'></a>
# ## Data Wrangling
# 
# 
# 
# ### General Properties

# In[216]:


#load and inspect data
df=pd.read_csv('tmdb-movies.csv')
df.head()


# In[217]:


#check for data dimensions
df.shape


# In[218]:


#check for null values 
df.info()


# In[219]:


#double check for how many missing values 
df.isnull().sum()


# In[220]:


#chekck for duplicated rows
df.duplicated().sum()


# In[221]:


# check for descriptive statistics
df.describe()


# In[222]:


#check for genres and their counts
df['genres'].value_counts()


# In[223]:


#look at films that have above median popularity points
df_popular=df.query('popularity >= 15')
df_popular.sort_values(by = ['popularity'], ascending = [False]).head()


# > - Gathered Information
#        1) Popularity points is vague. Not clear on how it was measured.Very skewed due to outliers.
#        2) Missing some imdb_id and they aren't integers either. 
#        3) Has 1 duplicated row.
#        4) genres contain multiple values separated by pipe (|) characters and missing values 
#        5) budget_adj & revenue_adj data more suitable ($ inflation over time).
#             
# > - Data Cleaning Plans
#         1) Drop unnecesary columns
#         2) Drop duplicated rows
#         3) Create column for 'decades'
#         4) Separate genres and tally to find highest count per year to determine what was popular 
# 
# ### Data Cleaning 

# In[224]:


#make a copy of the original df
movies=df.copy()


# In[225]:


#drop duplicated rows
movies.drop_duplicates(inplace=True)


# In[226]:


#drop unnecessary columns excluding imdb_id since all values are present in id 
col = ['homepage', 'tagline', 'overview', 'budget', 'revenue','original_title','cast','director',
       'keywords','overview','production_companies','release_date']
movies.drop(col, axis=1, inplace=True)


# In[227]:


#check to see if columns are dropped
movies.head(1)


# In[228]:


#drop null values in genres column
drop = ['genres']
movies.dropna(subset=drop,how='any',inplace=True)


# In[229]:


#use pd .str.split to split series (genres)                                                      *returns as series
#use pd .apply to pass and apply the function (pd.Series) to above (Level 1 =column names)    *stored as new series
#use pd .stack to reshape into a stacked form                                  *creates more rows (separated genres)   
#use pd .reset_index to reset index                                * drop=True drops original index labels 

genre_split = movies['genres'].str.split('|').apply(pd.Series,1).stack().reset_index(level=1, drop=True)

#name the new column 
genre_split.name = 'genre'

#use pd. join to combine columns (b/c it maybe differently indexed)
movies = movies.drop(['genres'], axis=1).join(genre_split)


# In[230]:


#check data to see genres are separated
movies.head(3)


# In[231]:


#check for genre categories and their total count
movies['genre'].value_counts()


# In[232]:


#above shows that there are 20 genres 
#use np unique function to double check 
movies['genre'].unique()


# In[233]:


#create decade column to the movies df 

#create bins and defining exact edges 
edges = [1959, 1970, 1980, 1990, 2000, 2010, 2016]

#values that fall within the edges will be placed under these names accordingly 
names = ['1960', '1970', '1980', '1990', '2000', '2010']

#use pd.cut to categorize bin values into discrete intervals 
#bins are constant size 
#values from release_year are 1-dimensional 
#labels set lists of values correspond to how the age values will be put in bins by decades
movies['decade'] = pd.cut(movies['release_year'], edges, labels=names)
movies.head()


# In[234]:


movies.info()


# <a id='eda'></a>
# ## Exploratory Data Analysis
# 
# ### Research Question 1: Which genres are most produced throughout time?
# 
# 

# In[235]:


#make a copy of the original df
genre=movies.copy()


# In[236]:


#double check (should be 0)
genre.duplicated().sum()


# In[237]:


#use movie production count to measure popularity
#split data into groups
#use pd groupby where first grouping is based on 'release year' and within each release year, group based on 'genre'
#use np size to count the number of elements along the axis 
#use pd reset_index to create column name as 'count'

genre=genre.groupby(['release_year', 'genre']).size().reset_index(name='count')
genre.head(10)


# In[238]:


genre.shape


# In[239]:


genre.info()


# In[240]:


#group based on 'release year' but not as new index        *returns as an object (new df)
#use apply to accompany the following function; lambda 
#lambda function sorts data frame from the column 'count', minimum being 1 count
#reset_index(drop=True) is used to drop the current index of the df and replace it with index of increaseing integer
#reset_index(drop=True) won't drop columns (for visualization)

popular=genre.groupby(by='release_year', as_index=False).apply(lambda x: x.sort_values('count').tail(1)).reset_index(drop=True)
popular


# In[241]:


#check for most popular genres
popular['genre'].unique()


# In[242]:


#check how many Drama and Comedy bars to expect on graph
popular['genre'].value_counts()


# In[243]:


# there should be 9 Comedy and 47 Drama bars
#set figure size
plt.figure(figsize = (50,20))

#use seaborn to barplot with bar color respective to popular genre 
sns.barplot(x = 'release_year', y = 'count', data = popular, hue='genre',dodge=False)

plt.title("Which genres are most popular from year to year?", fontsize = 30)
plt.xlabel("Year (1960 to 2015)", fontsize = 30)
plt.ylabel("Movie Production Count", fontsize = 30)
plt.legend(prop={"size":30})

plt.show()


# > - Answers
#         1) Top two most popular genres based on production count are Drama and Comedy
#         2) Drama has been most popular generally from year to year 
# 

# ### Research Question 2: How did film budgets change from each decade in average?

# In[244]:


#set copy of cleaned df 
budget = movies.copy()


# In[245]:


#double check
budget.head()


# In[246]:


#double check again 
budget.duplicated().sum()


# In[247]:


budget=budget.groupby('decade')['budget_adj'].mean().reset_index(name='budget')
budget


# In[248]:


#set figure size 
plt.figure(figsize = (30,10))

#use seaborn to lineplot average film budget per decade 
sns.barplot(x = 'decade', y = 'budget', data = budget)

plt.title("Average Film Budget", fontsize = 30)
plt.xlabel("Decade", fontsize = 20)
plt.ylabel("Budget ($ millions)", fontsize = 20)


plt.show()


# > - Answers
#         1) 1990s saw the highest film budget on average while 1970s was the lowest.
#         2) There is an increase in film production from 1970s to 1990s.
#         3) There is a decrease in film production from 1990s to 2010s.
#         4) Since the data contains year up to 2015, film budget in 2010s will increase when updated.

# ### Research Question 3: What kinds of properties are associated with movies that have high revenues?

# In[249]:


#set copy of original df, better to use the original b/c cleaned df includes many duplicated movies per genre 
rev = df.copy()
rev.head()


# In[250]:


#double check 
rev.duplicated().sum()


# In[251]:


#drop duplicates
rev.drop_duplicates(inplace=True)


# In[252]:


#make new df where it's sorted by decreasing revenue, should have different id
high_rev = rev.sort_values(by = ['revenue_adj'], ascending = [False])
high_rev.head()


# In[253]:


#use groupby take the top 100 highest revenue movies 
high_rev = high_rev.groupby('revenue_adj').head(100)
high_rev.head(1)


# In[254]:


#use pd corr function to see correlation of the high_rev df 
high_rev_corr=high_rev.corr(method='pearson')
high_rev_corr


# In[255]:


#use seaborn to visualize the correlation above 
sns.heatmap(high_rev_corr, 
            xticklabels=high_rev_corr.columns,
            yticklabels=high_rev_corr.columns,
            cmap='RdBu_r',
            annot=True,
            linewidth=0.5);


# In[256]:


#vote count has highest correlation of .67
sns.relplot(x='revenue_adj', y='vote_count', data=high_rev_corr, kind="scatter")
plt.show()


# In[257]:


#revenue adjusted has second highest correlation of .6
sns.relplot(x='revenue_adj', y='budget_adj', data=high_rev_corr, kind="scatter")
plt.show()


# In[258]:


#popularity has the 3rd highest correlation of .56
sns.relplot(x='revenue_adj', y='popularity', data=high_rev_corr, kind="scatter")
plt.show()


# > - Answers
#         1) Vote Count, Adjusted Budget, and Popularity Points show highest association with movies                                         that have high revenues

# <a id='conclusions'></a>
# ## Conclusions
# > **Summary**
# > - The top two most popular genres based on production count are Drama and Comedy.
# > - Drama has been most popular generally from 1960 to 2015.
# > - 1990s saw the highest film budget on average while 1970s was the lowest.
# > - There is an increase in film production budget from 1970s to 1990s.
# > - There is a decrease in film production budget from 1990s to 2010s.
# > - Since the data contains information up to 2015, film budget in 2010s will increase when updated (due to more films produced).
# > - Vote Count, Adjusted Budget, and Popularity Points show highest association with movies that have high revenues.
# 
# > **Limitations**: 
# > - Popularity is deemed subjective and the popularity data contains many outliers. It was unsuitable to use to measure popularity because we don't know how it was derived.
# > - The original dataset only contains data from TMDb but there are other sites such as Rotten Tomatoes and Metacritic that collect possibly more accurate data.
# > - I should have aimed to answer the 3rd question first in order to maximize my use of cleaned dataframe.
# 
