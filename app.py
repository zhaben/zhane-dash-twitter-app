#!/usr/bin/env python
# coding: utf-8

# # How do people feel when tweeting #METOO, #BLM, and #MAGA?

# ## Accessing tweets from Twitter's API

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
api = tw.API(auth, wait_on_rate_limit=True)


# ## Extracting timestamps from tweets, saving data to a csv

# In[2]:


import tweepy
import csv
# Open/create a file to append data to
csvFile = open('result-1.csv', 'a')

#Use csv writer
csvWriter = csv.writer(csvFile)

new_search = "#metoo -filter:retweets"

for tweet in tweepy.Cursor(api.search,
                           q = new_search,
                           since = "2020-08-30",
                           until = "2020-09-30",
                           lang = "en").items(100):

    # Write a row to the CSV file. I use encode UTF-8
    csvWriter.writerow([tweet.created_at, tweet.text.encode('utf-8'), tweet.coordinates, new_search])
#    print(tweet.created_at, tweet.text, tweet.coordinates, new_search)
#print(dir(tweet))
csvFile.close()


# In[3]:


csvFile = open('result-2.csv', 'a')

#Use csv writer
csvWriter = csv.writer(csvFile)

new_search = "#blm -filter:retweets"

for tweet in tweepy.Cursor(api.search,
                           q = new_search,
                           since = "2020-08-30",
                           until = "2020-09-30",
                           lang = "en").items(100):

    # Write a row to the CSV file. I use encode UTF-8
    csvWriter.writerow([tweet.created_at, tweet.text.encode('utf-8'), tweet.coordinates, new_search])
#    print(tweet.created_at, tweet.text, tweet.coordinates, new_search)
#print(dir(tweet))
csvFile.close()


# In[4]:


csvFile = open('result-3.csv', 'a')

#Use csv writer
csvWriter = csv.writer(csvFile)

new_search = "#maga -filter:retweets"

for tweet in tweepy.Cursor(api.search,
                           q = new_search,
                           since = "2020-08-30",
                           until = "2020-09-30",
                           lang = "en").items(100):

    # Write a row to the CSV file. I use encode UTF-8
    csvWriter.writerow([tweet.created_at, tweet.text.encode('utf-8'), tweet.coordinates, new_search])
#    print(tweet.created_at, tweet.text, tweet.coordinates, new_search)
#print(dir(tweet))
csvFile.close()


# In[5]:


#Concatinating all the results into one dataframe
w = pd.read_csv("result-1.csv", header = None, names=["Time", "Tweet", "Location", "Hashtag"])
x = pd.read_csv("result-2.csv", header = None, names=["Time", "Tweet", "Location", "Hashtag"])
y = pd.read_csv("result-3.csv", header = None, names=["Time", "Tweet", "Location", "Hashtag"])
z = pd.concat([w, x, y])


# ## Sentiment Analysis using NLP
# ### Categorizing each tweet as positive, negative, or neutral

# In[6]:


#!pip install afinn
from afinn import Afinn
af = Afinn()     #Instantiates an Afinn object


# In[7]:


#Assigning a sentiment score and category to the results
z_sentiment_scores = [af.score(tweet) for tweet in z['Tweet']]
z_sentiment_category = ['positive' if score > 0 
                          else 'negative' if score < 0 
                              else 'neutral' 
                                  for score in z_sentiment_scores]
#Adding sentiment scores and categories to dataframe
z['sentiment_score'] = z_sentiment_scores
z['sentiment_category'] = z_sentiment_category
z.head(3)


# In[8]:


#positive tweets
z[(z['Hashtag'] == "#metoo -filter:retweets") & (z.sentiment_category == 'positive')][:3]


# In[9]:


#negative tweets
x = z[z['Hashtag'] == '#blm -filter:retweets']
x[x['sentiment_score'] == min(x['sentiment_score'])]


# In[10]:


#worst tweet
z[z.sentiment_score == min(z.sentiment_score)]


# In[11]:


#best tweet
z[z.sentiment_score == max(z.sentiment_score)]


# In[12]:


#neutral tweets
z[(z['Hashtag'] == "#metoo -filter:retweets") & (z.sentiment_category == 'neutral')][:3]


# # Visualizations
# ## Line Charts 
# ### Plotting Sentiment Scores over Time

# In[13]:


#Line chart of sentiment scores over time for each hashtag
import plotly.graph_objects as go
import numpy as np

x = z['Time']
y = z['sentiment_score']

fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=z.query("Hashtag=='#metoo -filter:retweets'")['sentiment_score'], name="metoo",
                    hoverinfo='text+name',     
                    line_shape='linear'))
fig.add_trace(go.Scatter(x=x, y=z.query("Hashtag=='#blm -filter:retweets'")['sentiment_score'], name="blm",
                     hoverinfo='text+name',
                     line_shape='linear'))
fig.add_trace(go.Scatter(x=x, y=z.query("Hashtag=='#maga -filter:retweets'")['sentiment_score'], name="blm",
                     hoverinfo='text+name',
                     line_shape='linear'))

fig.update_traces(hoverinfo='text+name', mode='lines+markers')
fig.update_layout(legend=dict(y=0.5, traceorder='reversed', font_size=16))

fig.show()


# ## Describing Sentiments Statistically per Hashtag w/ Comparisons

# In[14]:


#sentiment statistics per hashtag
z.groupby(by=['Hashtag']).describe()
#z.describe()


# In[16]:


# spread of sentiment polarity
import matplotlib.pyplot as plt
import seaborn as sns

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
sp = sns.stripplot(x='Hashtag', y="sentiment_score", 
                   hue='Hashtag', data=z, ax=ax1)
bp = sns.boxplot(x='Hashtag', y="sentiment_score", 
                 hue='Hashtag', data=z, palette="Set2", ax=ax2)
t = f.suptitle('Visualizing Hashtag Sentiment', fontsize=14)
plt.show()


# In[17]:


dp = sns.displot(x="Hashtag", data=z, hue = 'sentiment_category', multiple="dodge", shrink=.8)


# ## Pie Charts

# In[19]:


import plotly.express as px
#function to create a pie chart for each hashtag

#breaking positives, negatives, and neutrals in separate dataframes, 
#selecting sentiment scores, 
#counting how many scores in each dataframe
#making it a type string so that it can go into px.pie

def pie_chart(i):
    
    df = z[z['Hashtag'] == '#'+ str(i) + ' -filter:retweets']
    
    pos_num = df[df['sentiment_category'] == 'positive']['sentiment_score'].count().astype(str)
    neg_num = df[df['sentiment_category'] == 'negative']['sentiment_score'].count().astype(str)
    neu_num = df[df['sentiment_category'] == 'neutral']['sentiment_score'].count().astype(str)

    counts = [pos_num, neg_num, neu_num]
    print(counts)
    fig = px.pie(df, values=counts, names= ['Positive', 'Negative', 'Neutral'], title='#'+str(i)+' Sentimentient Categories')
    return fig.show()
    
print(pie_chart('metoo'))
print(pie_chart('blm'))
print(pie_chart('maga'))


# # Launching Visualization to Dash

# In[ ]:


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

######

#Variables for Dash
df_pie_metoo = z[z['Hashtag'] == '#metoo -filter:retweets']
pos_num_metoo = df_pie_metoo[df_pie_metoo['sentiment_category'] == 'positive']['sentiment_score'].count().astype(str)
neg_num_metoo = df_pie_metoo[df_pie_metoo['sentiment_category'] == 'negative']['sentiment_score'].count().astype(str)
neu_num_metoo = df_pie_metoo[df_pie_metoo['sentiment_category'] == 'neutral']['sentiment_score'].count().astype(str)

df_pie_blm = z[z['Hashtag'] == '#blm -filter:retweets']
pos_num_blm = df_pie_blm[df_pie_blm['sentiment_category'] == 'positive']['sentiment_score'].count().astype(str)
neg_num_blm = df_pie_blm[df_pie_blm['sentiment_category'] == 'negative']['sentiment_score'].count().astype(str)
neu_num_blm = df_pie_blm[df_pie_blm['sentiment_category'] == 'neutral']['sentiment_score'].count().astype(str)

df_pie_maga = z[z['Hashtag'] == '#maga -filter:retweets']
pos_num_maga = df_pie_maga[df_pie_maga['sentiment_category'] == 'positive']['sentiment_score'].count().astype(str)
neg_num_maga = df_pie_maga[df_pie_maga['sentiment_category'] == 'negative']['sentiment_score'].count().astype(str)
neu_num_maga = df_pie_maga[df_pie_maga['sentiment_category'] == 'neutral']['sentiment_score'].count().astype(str)

#####

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
                            }),
                dcc.Graph(
                    
                    figure={
                            'data': [
                                        go.Scatter(x=z['Time'], 
                                            y=z.query("Hashtag=='#metoo -filter:retweets'")['sentiment_score'], 
                                            name="metoo",
                                            hoverinfo='text+name',     
                                            line_shape='linear',
                                            line_color='red'
                                                  ),
                                            
                                        go.Scatter(x=z['Time'], 
                                            y=z.query("Hashtag=='#blm -filter:retweets'")['sentiment_score'], 
                                            name="blm",
                                            hoverinfo='text+name',
                                            line_shape='linear',
                                            line_color='blue'
                                                  ),
                                            
                                        go.Scatter(x=z['Time'], 
                                            y=z.query("Hashtag=='#maga -filter:retweets'")['sentiment_score'], 
                                            name="maga",
                                            hoverinfo='text+name',
                                            line_shape='linear',
                                            line_color='green'
                                                  )                                        
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

