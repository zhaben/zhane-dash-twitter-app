#!/usr/bin/env python
# coding: utf-8

# # Comparing Hashtag Success: #METOO, #BLM, #MAGA

# # Accessing tweets

# In[1]:


import os
import tweepy as tw
import pandas as pd

access_token = '1279128353136574465-yGKBss8obvnK0LLT3nV6viICmDwbQB'
access_secret = 'wrVAWl884P4t0PwU9JXloWRzshdhGZEnRcayoxSW8xE75'
consumer_key = 'AHZyv0lHpvS2C30F0lPtJI7Yl'
consumer_secret = 'wvkav6VaaTTn9ekE2C56jERxMITyasf5NZLPmPIsANacfgnj9u'

auth = tw.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)
api = tw.API(auth, wait_on_rate_limit=True, timeout=5)


# In[2]:


# api.update_status("My first tweet!")


# In[3]:


search_words = "#wildfires"
date_since = "2020-09-01"

# collect tweets using cursor method, returns a cursor object
tweets = tw.Cursor(api.search, q=search_words, long="en", since=date_since).items(5)
tweets


# In[4]:


# iterate over attributes in object to get incormation about each tweet
for tweet in tweets:
    print(tweet.text)


# In[5]:


# collect a list of tweets using a list comprehension
x = [tweet.text for tweet in tweets]


# In[6]:


print(x)


# In[7]:


# exclude retweets from search
new_search = search_words +  "-filter:retweets"
new_search


# In[8]:


tweets = tw.Cursor(api.search,
                       q=new_search,
                       lang="en",
                       since=date_since).items(5)


# In[9]:


[tweet.text for tweet in tweets]


# In[10]:


# disp display all attributes for each tweet
#tweet


# In[11]:


# access attributes for each tweet
users_locs = [[tweet.user.screen_name, tweet.user.location] for tweet in tweets]
users_locs


# In[12]:


# create dataframe from a list of tweet data

tweet_text = pd.DataFrame(data=users_locs, 
                    columns=['user', "location"])
tweet_text


# In[13]:


new_search = "climate+change -filter:retweets"

tweets = tw.Cursor(api.search,
                   q=new_search,
                   lang="en",
                   since='2018-04-23').items(1000)

all_tweets = [tweet.text for tweet in tweets]
all_tweets[:5]


# In[14]:


new_search = "#metoo -filter:retweets"

tweets = tw.Cursor(api.search,
                   q=new_search,
                   lang="en",
                   since='2018-04-23').items(20)

metoo_tweets = [tweet.text for tweet in tweets]
metoo_tweets[:5]


# In[15]:


new_search = "#blm -filter:retweets"

tweets = tw.Cursor(api.search,
                   q=new_search,
                   lang="en",
                   since='2018-04-23').items(20)

blm_tweets = [tweet.text for tweet in tweets]
blm_tweets[:5]


# In[16]:


new_search = "#maga -filter:retweets"

tweets = tw.Cursor(api.search,
                   q=new_search,
                   lang="en",
                   since='2018-04-23').items(20)

maga_tweets = [tweet.text for tweet in tweets]
maga_tweets[:5]


# ## Sentiment Analysis Using NLP

# In[23]:


#!pip install afinn
from afinn import Afinn
af = Afinn()     #Instantiates an Afinn object


# In[24]:


# compute sentiment scores and labels
metoo_sentiment_scores = [af.score(tweet) for tweet in metoo_tweets]
metoo_sentiment_category = ['positive' if score > 0 
                          else 'negative' if score < 0 
                              else 'neutral' 
                                  for score in metoo_sentiment_scores]


blm_sentiment_scores = [af.score(tweet) for tweet in blm_tweets]
blm_sentiment_category = ['positive' if score > 0 
                          else 'negative' if score < 0 
                              else 'neutral' 
                                  for score in blm_sentiment_scores]

maga_sentiment_scores = [af.score(tweet) for tweet in maga_tweets]
maga_sentiment_category = ['positive' if score > 0 
                          else 'negative' if score < 0 
                              else 'neutral' 
                                  for score in maga_sentiment_scores]


# In[152]:


# categorize and join all tweet data
df_metoo = pd.DataFrame(metoo_tweets)
df_metoo['hashtag'] = '#metoo'
df_metoo['sentiment_score'] = metoo_sentiment_scores
df_metoo['sentiment_category'] = metoo_sentiment_category

         
df_blm = pd.DataFrame(blm_tweets)
df_blm['hashtag'] = '#blm'
df_blm['sentiment_score'] = blm_sentiment_scores
df_blm['sentiment_category'] = blm_sentiment_category
         
df_maga = pd.DataFrame(maga_tweets)
df_maga['hashtag'] = '#maga'
df_maga['sentiment_score'] = maga_sentiment_scores
df_maga['sentiment_category'] = maga_sentiment_category
         
df_all = pd.concat([df_metoo, df_blm, df_maga])
df_all.rename(columns={ df_all.columns[0]: "tweet" }, inplace = True)
df_all.head()


# In[26]:


#sentiment statistics per hashtag
#df = pd.DataFrame([list(df_all['hashtag']), sentiment_scores, sentiment_category]).T
#df.columns = ['hashtag', 'sentiment_score', 'sentiment_category']
df = df_all
df['sentiment_score'] = df.sentiment_score.astype('float')
df.groupby(by=['hashtag']).describe()


# Average sentiment is lower in blm and higher in metoo.

# In[27]:


# spread of sentiment polarity -- much higher in blm and maga, metoo has a lot more negative polarity
import matplotlib.pyplot as plt
import seaborn as sns

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
sp = sns.stripplot(x='hashtag', y="sentiment_score", 
                   hue='hashtag', data=df, ax=ax1)
bp = sns.boxplot(x='hashtag', y="sentiment_score", 
                 hue='hashtag', data=df, palette="Set2", ax=ax2)
t = f.suptitle('Visualizing Hashtag Sentiment', fontsize=14)
plt.show()


# In[28]:


# frequency of sentiment labels
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
sp = sns.stripplot(x='hashtag', y="sentiment_score", 
                   hue='hashtag', data=df, ax=ax1)
bp = sns.boxplot(x='hashtag', y="sentiment_score", 
                 hue='hashtag', data=df, palette="Set2", ax=ax2)
t = f.suptitle('Visualizing Hashtag Sentiment', fontsize=14)


# In[29]:


dp = sns.displot(x="hashtag", data=df, hue = 'sentiment_category', multiple="dodge", shrink=.8
                )


# metoo has the lowest number of positive sentiments, blm has the highest
# blm has the lowest number of neutral sentiments
# maga has the lowest number of negative sentiments
# what were the most positive and negative reviews about?

# In[30]:


df[(df['hashtag'] == "#blm") & (df.sentiment_score == max(df.sentiment_score))]
#the most positive tweet was about naomi osaka.


# In[31]:


df[(df['hashtag'] == "#blm") & (df.sentiment_score == min((df['hashtag'] == "#blm")))]


# In[32]:


df[(df['hashtag'] == "#maga") & (df.sentiment_score == min((df['hashtag'] == "#maga")))]


# ##  Visualization
# Pie charts

# In[149]:


#need to create a function to do this for each hashtag
import plotly.express as px

df_pie = df[df['hashtag'] == '#metoo']

#breaking positives, negatives, and neutrals in separate dataframes, 
#selecting sentiment scores, 
#ounting how many scores in each dataframe
#making it a type string so that it can go into px.pie

pos_num = df_pie[df_pie['sentiment_category'] == 'positive']['sentiment_score'].count().astype(str)
neg_num = df_pie[df_pie['sentiment_category'] == 'negative']['sentiment_score'].count().astype(str)
neu_num = df_pie[df_pie['sentiment_category'] == 'neutral']['sentiment_score'].count().astype(str)

scores = [pos_num, neg_num, neu_num]
print(scores)
fig = px.pie(df_pie, values=[pos_num, neg_num, neu_num], names= ['Positive', 'Negative', 'Neutral'], title='Me Too Sentimentient Categories')
fig.show()


# In[150]:


df_pie = df[df['hashtag'] == '#blm']

pos_num = df_pie[df_pie['sentiment_category'] == 'positive']['sentiment_score'].count().astype(str)
neg_num = df_pie[df_pie['sentiment_category'] == 'negative']['sentiment_score'].count().astype(str)
neu_num = df_pie[df_pie['sentiment_category'] == 'neutral']['sentiment_score'].count().astype(str)

scores = [pos_num, neg_num, neu_num]
print(scores)
fig = px.pie(df_pie, values=[pos_num, neg_num, neu_num], names= ['Positive', 'Negative', 'Neutral'], title='BLM Sentimentient Categories')
fig.show()


# In[151]:


df_pie = df[df['hashtag'] == '#maga']

pos_num = df_pie[df_pie['sentiment_category'] == 'positive']['sentiment_score'].count().astype(str)
neg_num = df_pie[df_pie['sentiment_category'] == 'negative']['sentiment_score'].count().astype(str)
neu_num = df_pie[df_pie['sentiment_category'] == 'neutral']['sentiment_score'].count().astype(str)

scores = [pos_num, neg_num, neu_num]
print(scores)
fig = px.pie(df_pie, values=[pos_num, neg_num, neu_num], names= ['Positive', 'Negative', 'Neutral'], title='MAGA Sentimentient Categories')
fig.show()


# # Launching Visualization to Dash

# In[ ]:


import pandas as pd
import plotly.express as px  # (version 4.7.0)
import plotly.graph_objects as go

import dash  # (version 1.12.0) pip install dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

app = dash.Dash(__name__)

# ------------------------------------------------------------------------------
# Import and clean data (importing csv into pandas)

#df = pd.read_csv("AB_NYC_2019.csv")
#df = df.groupby(['name', 'neighbourhood_group', 'neighbourhood', 'room_type', 'latitude', 'longitude'])[['minimum_nights']].mean()
#df.reset_index(inplace=True)
print(df[:5])

# ------------------------------------------------------------------------------
# App layout
app.layout = html.Div([


        html.H1("Twitter Hashtag Dashboard", style={'text-align': 'center'}),
        html.Br(),
        html.H4('Hashtag Pie Chart Comparison:'),
#         dcc.Graph(
#                             id='pie-chart',
#                             figure={
#                                 'data': [
#                                     go.Pie(
#                                         labels=['Positives', 'Negatives', 'Neutrals'], 
#                                         values=[pos_num, neg_num, neu_num],
#                                         name="View Metrics",
#                                         marker_colors=['rgba(184, 247, 212, 0.6)','rgba(255, 50, 50, 0.6)','rgba(131, 90, 241, 0.6)'],
#                                         textinfo='value',
#                                         hole=.65)
#                                 ]

#                             }
#                         )
                                     

html.Br(),
      dcc.Graph(id='pie-chart', figure={
                                'data': [
                                    go.Pie(
                                        labels=['Positives', 'Negatives', 'Neutrals'], 
                                        values=[pos_num, neg_num, neu_num],
                                        name="View Metrics",
                                        marker_colors=['rgba(184, 247, 212, 0.6)','rgba(255, 50, 50, 0.6)','rgba(131, 90, 241, 0.6)'],
                                        textinfo='value',
                                        hole=.65)
                                ]

                            })

])
# ------------------------------------------------------------------------------
# Connect the Plotly graphs with Dash Components
@app.callback(
    Output(component_id='pie-chart', component_property='figure'),
)
        
def update_graph(figure):
 

    # Plotly Express
    fig = px.pie(df_pie, values=[pos_num, neg_num, neu_num], 
                 names= ['Positive', 'Negative', 'Neutral'], title='MAGA Sentimentient Categories')
    
    return fig

# ------------------------------------------------------------------------------
if __name__ == '__main__':
    app.run_server(debug=False)




