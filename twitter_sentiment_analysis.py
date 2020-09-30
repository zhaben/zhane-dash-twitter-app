#!/usr/bin/env python
# coding: utf-8

# # How do people feel when tweeting #METOO, #BLM, and #MAGA?

# # Accessing tweets

# In[21]:


import os
import tweepy as tw
import pandas as pd

import plotly
import plotly.express as px  # (version 4.7.0)
import plotly.graph_objects as go

import dash  # (version 1.12.0) pip install dash
import dash_core_components as dcc
import dash_html_components as html

access_token = '1279128353136574465-yGKBss8obvnK0LLT3nV6viICmDwbQB'
access_secret = 'wrVAWl884P4t0PwU9JXloWRzshdhGZEnRcayoxSW8xE75'
consumer_key = 'AHZyv0lHpvS2C30F0lPtJI7Yl'
consumer_secret = 'wvkav6VaaTTn9ekE2C56jERxMITyasf5NZLPmPIsANacfgnj9u'

auth = tw.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)
api = tw.API(auth, wait_on_rate_limit=True)


# In[22]:


new_search = "#metoo -filter:retweets"

tweets = tw.Cursor(api.search,
                   q=new_search,
                   lang="en",
                   since='2018-04-23').items(100)

metoo_tweets = [tweet.text for tweet in tweets]
metoo_tweets[:5]


# In[23]:


new_search = "#blm -filter:retweets"

tweets = tw.Cursor(api.search,
                   q=new_search,
                   lang="en",
                   since='2018-04-23').items(100)

blm_tweets = [tweet.text for tweet in tweets]
blm_tweets[:5]


# In[24]:


new_search = "#maga -filter:retweets"

tweets = tw.Cursor(api.search,
                   q=new_search,
                   lang="en",
                   since='2018-04-23').items(100)

maga_tweets = [tweet.text for tweet in tweets]
maga_tweets[:5]


# ## Sentiment Analysis Using NLP

# In[25]:


#!pip install afinn
from afinn import Afinn
af = Afinn()     #Instantiates an Afinn object


# In[26]:


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


# In[27]:


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


# In[29]:


#sentiment statistics per hashtag
df = df_all
df['sentiment_score'] = df.sentiment_score.astype('float')
df.groupby(by=['hashtag']).describe()


# Average sentiment is lower in blm and higher in metoo.

# In[30]:


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


# In[31]:


# frequency of sentiment labels
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
sp = sns.stripplot(x='hashtag', y="sentiment_score", 
                   hue='hashtag', data=df, ax=ax1)
bp = sns.boxplot(x='hashtag', y="sentiment_score", 
                 hue='hashtag', data=df, palette="Set2", ax=ax2)
t = f.suptitle('Visualizing Hashtag Sentiment', fontsize=14)


# In[32]:


dp = sns.displot(x="hashtag", data=df, hue = 'sentiment_category', multiple="dodge", shrink=.8
                )


# metoo has the lowest number of positive sentiments, blm has the highest
# blm has the lowest number of neutral sentiments
# maga has the lowest number of negative sentiments
# what were the most positive and negative reviews about?

# In[33]:


df[(df['hashtag'] == "#blm") & (df.sentiment_score == max(df.sentiment_score))]
#the most positive tweet was about naomi osaka.


# In[34]:


df[(df['hashtag'] == "#blm") & (df.sentiment_score == min((df['hashtag'] == "#blm")))]


# In[35]:


df[(df['hashtag'] == "#maga") & (df.sentiment_score == min((df['hashtag'] == "#maga")))]


# ##  Visualization
# Pie charts

# In[36]:


#need to create a function to do this for each hashtag
import plotly.express as px

df_pie_metoo = df[df['hashtag'] == '#metoo']

#breaking positives, negatives, and neutrals in separate dataframes, 
#selecting sentiment scores, 
#ounting how many scores in each dataframe
#making it a type string so that it can go into px.pie

pos_num_metoo = df_pie_metoo[df_pie_metoo['sentiment_category'] == 'positive']['sentiment_score'].count().astype(str)
neg_num_metoo = df_pie_metoo[df_pie_metoo['sentiment_category'] == 'negative']['sentiment_score'].count().astype(str)
neu_num_metoo = df_pie_metoo[df_pie_metoo['sentiment_category'] == 'neutral']['sentiment_score'].count().astype(str)

scores = [pos_num_metoo, neg_num_metoo, neu_num_metoo]
print(scores)
fig = px.pie(df_pie_metoo, values=[pos_num_metoo, neg_num_metoo, neu_num_metoo], names= ['Positive', 'Negative', 'Neutral'], title='Me Too Sentimentient Categories')
fig.show()


# In[37]:


df_pie_blm = df[df['hashtag'] == '#blm']

pos_num_blm = df_pie_blm[df_pie_blm['sentiment_category'] == 'positive']['sentiment_score'].count().astype(str)
neg_num_blm = df_pie_blm[df_pie_blm['sentiment_category'] == 'negative']['sentiment_score'].count().astype(str)
neu_num_blm = df_pie_blm[df_pie_blm['sentiment_category'] == 'neutral']['sentiment_score'].count().astype(str)

scores = [pos_num_blm, neg_num_blm, neu_num_blm]
print(scores)
fig = px.pie(df_pie_blm, values=[pos_num_blm, neg_num_blm, neu_num_blm], names= ['Positive', 'Negative', 'Neutral'], title='BLM Sentimentient Categories')
fig.show()


# In[38]:


df_pie_maga = df[df['hashtag'] == '#maga']

pos_num_maga = df_pie_maga[df_pie_maga['sentiment_category'] == 'positive']['sentiment_score'].count().astype(str)
neg_num_maga = df_pie_maga[df_pie_maga['sentiment_category'] == 'negative']['sentiment_score'].count().astype(str)
neu_num_maga = df_pie_maga[df_pie_maga['sentiment_category'] == 'neutral']['sentiment_score'].count().astype(str)

scores = [pos_num_maga, neg_num_maga, neu_num_maga]
print(scores)
fig = px.pie(df_pie_maga, values=[pos_num_maga, neg_num_maga, neu_num_maga], names= ['Positive', 'Negative', 'Neutral'], title='MAGA Sentimentient Categories')
fig.show()

# # Launching Visualization to Dash

# In[39]:


import pandas as pd
import plotly.express as px  # (version 4.7.0)
import plotly.graph_objects as go

import dash
import dash_html_components as html
import dash_core_components as dcc
import plotly.graph_objects as go

from dash.dependencies import Input, Output

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

app.layout = html.Div([
    html.H1("Twitter Hashtag Sentiment Tracker", style={'text-align': 'center'}),
    html.Br(),
    dcc.Tabs(id='tabs', value='tab-1', 
             children=[
                dcc.Tab(label='1', value='tab-1'),
                dcc.Tab(label='2', value='tab-2'),
    ]),
    html.Div(id='tabs-content')
])


@app.callback(Output('tabs-content', 'children'),
        [Input('tabs', 'value')])

def render_content(tab):
    if tab == 'tab-1':
        return html.Div([
            html.H3('Sentiment Scores'),
                dcc.Graph(
                    figure={
                            'data': [
                            {'x': ['metoo', 'blm', 'maga'], 'y': [neu_num_metoo, neu_num_blm, neu_num_maga],
                                'type': 'bar', 'name': 'Neutral', 'marker': {"color": 'rgba(131, 90, 241, 0.6)'}},
                            {'x': ['metoo', 'blm', 'maga'], 'y': [pos_num_metoo, pos_num_blm, pos_num_maga],
                             'type': 'bar', 'name': 'Positive', 'marker': {"color": 'rgba(184, 247, 212, 0.6)'}},
                            {'x': ['metoo', 'blm', 'maga'], 'y': [neg_num_metoo, neg_num_blm, neg_num_maga],
                             'type': 'bar', 'name': 'Negative', 'marker': {"color": 'rgba(255, 50, 50, 0.6)'}}
                                    ]
                            })
                        ])

    elif tab == 'tab-2':        
        return html.Div([
            html.H3('#metoo'),
                dcc.Graph(figure={
                                'data': [
                                        go.Pie(
                                            labels=['Positives', 'Negatives', 'Neutrals'], 
                                            values=[pos_num_metoo, neg_num_metoo, neu_num_metoo],
                                            name="View Metrics",
                                            marker_colors=['rgba(184, 247, 212, 0.6)','rgba(255, 50, 50, 0.6)','rgba(131, 90, 241, 0.6)'],
                                            textinfo='value',
                                            hole=.65)
                                        ]

                                }),
            html.H3('#blm'),
                dcc.Graph(figure={
                                'data': [
                                        go.Pie(
                                            labels=['Positives', 'Negatives', 'Neutrals'], 
                                            values=[pos_num_blm, neg_num_blm, neu_num_blm],
                                            name="View Metrics",
                                            marker_colors=['rgba(184, 247, 212, 0.6)','rgba(255, 50, 50, 0.6)','rgba(131, 90, 241, 0.6)'],
                                            textinfo='value',
                                            hole=.65)
                                        ]

                                }),
            html.H3('#maga'),
                dcc.Graph(figure={
                                'data': [
                                        go.Pie(
                                            labels=['Positives', 'Negatives', 'Neutrals'], 
                                            values=[pos_num_maga, neg_num_maga, neu_num_maga],
                                            name="View Metrics",
                                            marker_colors=['rgba(184, 247, 212, 0.6)','rgba(255, 50, 50, 0.6)','rgba(131, 90, 241, 0.6)'],
                                            textinfo='value',
                                            hole=.65)
                                        ]

                                })
        
                        ])
  
  
if __name__ == '__main__':
    app.run_server(debug=False)
