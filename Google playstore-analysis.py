#!/usr/bin/env python
# coding: utf-8

# GOOGLE PLAYSTORE-ANALYSIS
# 
# Ojective:
# 
# Google Play Store team is about to launch a new feature where in certain apps that are 
# promising are boosted in visibility. The boost will manifest in multiple ways – higher priority in 
# recommendations sections (“Similar apps”, “You might also like”, “New and updated games”). 
# These will also get a boost in visibility in search results. This feature will help bring more 
# attention to newer apps that have potential.
# The task is to understand what makes an app perform well - size? price? category? multiple 
# factors together? Analyze the data and present your insights in a format consumable by 
# business – the final output of the analysis would be presented to business as insights with 
# supporting data/visualizations

# IMPORT LIBRARY

# In[109]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
color = sns.color_palette()
import plotly.graph_objects as go
import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# In[107]:


pip install plotly


# In[4]:


print(os.listdir())


# In[5]:


df=pd.read_csv('C:/Users/yuvak/OneDrive/Desktop/playstore-analysis.csv')


# In[6]:


df.head()


# In[7]:


df.info()


# In[8]:


df.describe()


# # 1.Data cleaning-Missing value identification & treating.
# 
# a.)Drop record values where rating is missing since rating is our target/study variable.

# In[9]:


df.isnull().sum()

Missing values

@Rating -1474
@Type   -01
@Content 
 Rating -01
@Current
 ver    -08
@Andriod
 ver    -03
# In[10]:


df1=df.dropna(subset=['Rating'])


# In[11]:


df1.isnull().sum()


# b.)Check null values for the andriod ver column.

# In[12]:


df1['Android Ver'].isnull().sum()


# In[13]:


df1[df1.isna().any(axis=1)]


# In[14]:


df.loc[[4453,4490,10472]]


# #Two missing values from 3rd record.
# 
# 1.content rating.
# 
# 2.andriod ver.
# 
# $droping down the 3rd record.

# In[15]:


df2=df1.drop(10472)


# In[17]:


try:
    df2.loc[[10472]]
except KeyError:
    print("successfully deleted")


# Replace remaining missing values with mode.

# In[18]:


df3=df2.copy(deep=True)
df3['Android Ver']=df2['Android Ver'].fillna(df2['Android Ver'].mode()[0])


# In[19]:


df3.loc[[4453,4490]]


# Missing values of Android ver is now 4.1 and up

# C.)Current ver - replace with most common value.

# In[20]:


df3[df3.isna().any(axis=1)]


# Most common value current ver

# In[21]:


mode_cv=df3['Current Ver'].value_counts().idxmax()
print(mode_cv)


# In[22]:


df4=df3.copy(deep=True)
df4[df4.isna().any(axis=1)]


# In[23]:


df4['Current Ver']=df3['Current Ver'].fillna(mode_cv)
df4.loc[[15,1553,6322,7333]]


# # 2.Data cleanup and correcting the data types
# 
# a.Which all variables need to be brought to numeroc types?

# In[24]:


df4.dtypes


# Following variables need to be brought to numeric types.
# 
# *Reviews
# 
# *Installs
# 
# *Price

# In[25]:


df5=df4.copy(deep=True)


# b.Price variable-remove$sign and convert to float.

# In[27]:


df5['Price']=df5['Price'].str.replace('$','')


# In[28]:


df5.loc[[4453,7333]]


# $ sign operator from price is removed

# In[44]:


df5['Price']=df5['Price'].astype(float)


# In[45]:


df5['Price'].dtypes


# price variable is now a float type.

# c.Installs - remove ',' and '+' sign, convert to integer.

# In[31]:


df5['Installs']=df5['Installs'].str.replace('+','')
df5['Installs']=df5['Installs'].str.replace(',','')


# In[32]:


df5.head()


# In[33]:


df5['Installs']=df5['Installs'].astype(int)


# In[34]:


df5['Installs'].dtypes


# signs are removed and converted to integer types.

# d.convert all other identified coloumns to numeric.

# In[37]:


df5.dtypes


# Reviews are converted to be numeric data types.

# In[39]:


df5['Reviews']=df5['Reviews'].astype(int)


# In[40]:


df5['Reviews'].dtypes


# In[41]:


df5.dtypes


# As before indicated variables are now numeric types
# 
# *Reviews
# 
# *Installs
# 
# *Price

# # 3.Sanity checks - check for the following and handle accordingly.
# 
# a. Avg.rating should be between 1 and 5, as only these values are allowed in play store.
# 
# 1.Are they are such records?Drip if so.

# In[46]:


check1=df5['Rating'] > 5


# In[47]:


check1.any()


# In[48]:


check2=df5['Rating'] < 1


# In[49]:


check2.any()


# No such records are founded.

# b.Reviews should not be  more than installs as only those who installed can Review the app. 
# 
# 1.Are they such records?drop if so.

# In[54]:


dfcheck=pd.DataFrame()
dfcheck=df5[df5.Reviews > df5.Installs]


# In[55]:


dfcheck.shape


# In[57]:


dfcheck.head(7)


# In this we have identified 7 invalid records.

# In[58]:


df6=df5.copy(deep=0)
df6.drop(df5[df5.Reviews > df5.Installs].index,inplace = True)


# In[60]:


dfcheck1 = df6[df6.Reviews >df6.Installs]
dfcheck1.shape


# All invalid records are dropped.

# # 4.Identify and handle outliers.
# 
# a.Price column
# 
# 1.Make suitable plot to identify outliers in place.

# In[67]:


def plot_box(df,c1):
    df.boxplot(column=[c1])
    plt.grid(False)
    plt.show()


# In[68]:


plot_box(df6,"Price")


# It indicates they have many outliers.

# In[69]:


def outliers(df,c1):
    q1=df[c1].quantile(0.25)
    q3=df[c1].quantile(0.75)
    iqr=q3-q1
    lower_bound=q1-1.5*iqr
    upper_bound=q3+1.5*iqr
    
    ls=df.index[(df[c1]<lower_bound) | (df[c1]>upper_bound)]
    return ls


# In[70]:


indexes=outliers(df6,"Price")


# In[71]:


indexes


# In[72]:


len(indexes)


# There are totally 538 outliers are founded in Price.
# 
# 2.Do you expect apps on the play store to cost $200?Check out these cases.

# In[73]:


df6.loc[df6['Price'] > 200]


# As per above statements the costs of apps represents $200.
# 
# 3.After dropping the useless records, make the suitable plot again to identify outliers.

# In[74]:


def remove(df,ls):
    ls=sorted(set(ls))
    df=df.drop(ls)
    return df


# In[75]:


dfcleaned=remove(df6,indexes)


# In[76]:


print(df6.shape,
dfcleaned.shape)


# In[77]:


plot_box(dfcleaned,"Price")


# 4.Limit data to records with price <$30.

# In[78]:


dflimit=df6[df6['Price']<30]


# In[81]:


print(df6.shape,
      dflimit.shape)


# b.Reviews column.
# 
# 1.Make suitable plot.

# In[86]:


total=df6.groupby('Genres')['Reviews'].sum().sort_values()
plt.subplots(figsize=(15,8))
total.head(10).plot(kind='pie',fontsize=14)

print(total.sort_values(ascending=False))
plt.show()


# Top 10 reviewd apps by Genre
# 
# 2.Limit data to apps with < 1Million reviews.

# In[90]:


dflim=df6[df6['Reviews']<1000000]
dflim=dflimit.sort_values(["Reviews"],ascending=False)
dflim.head()


# In[91]:


print(df6.shape,dflim.shape)


# c.Installs
# 
# 1.What is the 95th percentile of the installs?

# In[92]:


print("95th percentile of the installs:\n",df6.Installs.quantile(0.95))


# 2.Drop records having a value more than the 95th percentile.

# In[94]:


df6[df6['Reviews'] > 100000000.0]


# There are no values greater than 95th percentile.

# In[95]:


indices=df6[df6['Reviews'] > 100000000.0].index
df6.drop(indices,inplace = True)


# # Data analysis to answer business questions
# 
# 5.What is the distribution of ratings like?(use seaborn)More skewed towards higher/lower values?

# In[96]:


sns.distplot(df6['Rating'])
plt.show()


# From the above representing chart most of the rating lies between 4 and 5.

# b.what is the implication of this in your analysis?
# 
# Real life distributions are usually skewed. If there are too much skewness in the data, then many statistical model don’t work. So in skewed data, the tail region may act as an outlier for the statistical model and we know that outliers adversely affect the model’s performance especially regression-based models. So there is a necessity to transform the skewed data to close enough to a Gaussian distribution or Normal distribution. This will allow us to try more number of statistical model.
# 
# Conclusion: If we have a skewed data then it may harm our results. So, in order to use a skewed data we have to apply a log transformation over the whole set of values to discover patterns in the data and make it usable for the statistical model.

# # 6.What are the top content rating values?
# 

# In[98]:


print("top Content Rating values :\n",df6['Content Rating'].value_counts())


# In[99]:


Adult_rating = df[df['Content Rating'] == 'Adults only 18+'].index.to_list()
unrated =df[df['Content Rating'] == 'Unrated'].index.to_list()
df.drop(Adult_rating, inplace = True)
df.drop(unrated, inplace = True)
df['Content Rating'].value_counts()


# From the above values in the content rating adults+18 and unrated has only a few records and it has been removed/droped.

# In[180]:


import plotly.graph_objects as go

fig = go.Figure(go.Pie(
    name = "",
    values = [7414,1083,461,397],
    labels = ['Everyone','Teen','Mature 17+','Everyone 10+'],
))
fig.show()


# # 7. Effect of size on rating.
# 
# a. Make a joinplot to understand the effect of size on rating.

# In[113]:


sns.jointplot(x=df6['Size'],y=df6['Rating'],data=df6,kind='hex')
plt.show()


# b. Do you see any patterns?
#    
#    The most of the data is in between Rating 3.5-5.0 and size 0-40000. and data is dense on rating 4.5 and little bellow and      size of 20000
# 
# c. How do you explain the pattern?
#    
#    Apps that has size of 20mb are most rated and apps with size less than 20mb are not much rated also it gets even worse after 20mb as size increases ratings decreases

# # 8. Effect of price on rating.
# 
# a. Make a jointplot (with regression line).

# In[129]:


sns.jointplot(x ="Rating" , y = "Price" ,data = df6)
plt.show()


# b. What pattern do you see?
#    most rated apps are under $50
# 
# c. How do you explain the pattern?
#    Most expensive apps don't get much rating

# d. Replot the data, this time with only records with price >0.

# In[143]:


Price_greaterthan_zero = df6[df6['Price'] > 0]
sns.jointplot(x ="Price" , y = "Rating" ,data = Price_greaterthan_zero, kind = "reg" )
plt.show()


# In[145]:


sns.lmplot(x='Price', y='Rating', hue ='Content Rating', data=df6)
plt.show()


# # 9. Look at all the numeric interactions together –
#  
#  a. Make a pairplort with the colulmns - 'Reviews', 'Size', 'Rating', 'Price'.

# In[148]:


sns.pairplot(df6,vars=['Reviews','Size', 'Rating', 'Price'])
plt.show()


# # 10.Rating vs. content rating.
# 
# a. Make a bar plot displaying the rating for each content rating.

# In[149]:


df6.groupby(['Content Rating'])['Rating'].count().plot.bar(color="green")
plt.ylabel('Rating')
plt.show()


# b. Which metric would you use? Mean? Median? Some other quantile?

# In[150]:


sns.distplot(df6['Rating'],bins=5)
plt.show()


# In[155]:


plt.boxplot(df6['Rating'])
plt.show()


# Mean
# 
# The distribution of data is left skewed and has outliers.The mean is better than the median because it isn’t influenced by Outliers.

# In[156]:


ax=df6['Rating'].groupby(df6['Content Rating']).mean().plot(kind = 'bar')
ax.set(xlabel ='Rating of content', ylabel = 'Average of Ratings')
plt.show()


# # 11. Content rating vs. size vs. rating – 3 variables at a time
#   
#   a. Create 5 buckets (20% records in each) based on Size.

# In[166]:


bins=[0, 4600, 12000, 21516, 32000, 100000]
df['Size_Buckets'] = pd.cut(df['Size'], bins, labels=['VERY LOW','LOW','MED','HIGH','VERY HIGH'])
pd.pivot_table(df, values='Rating', index='Size_Buckets', columns='Content Rating')


# b. By Content Rating vs. Size buckets, get the rating (20th percentile) for each combination.

# In[171]:


df.Size.quantile([0.2, 0.4,0.6,0.8])


# In[172]:


df.Rating.quantile([0.2, 0.4,0.6,0.8])


# c. Make a heatmap of this
# 
# i. Annotated
# 
# ii. Greens color map

# In[175]:


Size_Buckets =pd.pivot_table(df6, values='Rating', index='Size_Buckets', columns='Content Rating', 
                     aggfunc=lambda x:np.quantile(x,0.2))
Size_Buckets


# In[178]:


sns.heatmap(Size_Buckets, annot = True,linewidth=0.5)
plt.show()


# In[179]:


sns.heatmap(Size_Buckets, annot=True,linewidth=0.5, cmap='Greens')
plt.show()


# d. What’s your inference? Are lighter apps preferred in all categories? Heavier? Some?
# 
#    As we can see last two rows have 4 and more ratings except two spots and first two rows have 4 and below ratings except two spots therefore we can say that Heavier apps preferred in all categories.
