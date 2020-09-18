#!/usr/bin/env python
# coding: utf-8

import os
import tweepy as tw
import pandas as pd
import plotly.express as px

access_token = '1279128353136574465-yGKBss8obvnK0LLT3nV6viICmDwbQB'
access_secret = 'wrVAWl884P4t0PwU9JXloWRzshdhGZEnRcayoxSW8xE75'
consumer_key = 'AHZyv0lHpvS2C30F0lPtJI7Yl'
consumer_secret = 'wvkav6VaaTTn9ekE2C56jERxMITyasf5NZLPmPIsANacfgnj9u'

auth = tw.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)
api = tw.API(auth, wait_on_rate_limit=True, timeout=5)




new_search = "#metoo -filter:retweets"

tweets = tw.Cursor(api.search,
                   q=new_search,
                   lang="en",
                   since='2018-04-23').items(20)

metoo_tweets = [tweet.text for tweet in tweets]
metoo_tweets[:5]


new_search = "#blm -filter:retweets"

tweets = tw.Cursor(api.search,
                   q=new_search,
                   lang="en",
                   since='2018-04-23').items(20)

blm_tweets = [tweet.text for tweet in tweets]
blm_tweets[:5]




new_search = "#maga -filter:retweets"

tweets = tw.Cursor(api.search,
                   q=new_search,
                   lang="en",
                   since='2018-04-23').items(20)

maga_tweets = [tweet.text for tweet in tweets]
maga_tweets[:5]



#!pip install afinn
from afinn import Afinn
af = Afinn()     #Instantiates an Afinn object




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


df = df_all
df['sentiment_score'] = df.sentiment_score.astype('float')



df_pie = df[df['hashtag'] == '#maga']

pos_num = df_pie[df_pie['sentiment_category'] == 'positive']['sentiment_score'].count().astype(str)
neg_num = df_pie[df_pie['sentiment_category'] == 'negative']['sentiment_score'].count().astype(str)
neu_num = df_pie[df_pie['sentiment_category'] == 'neutral']['sentiment_score'].count().astype(str)

scores = [pos_num, neg_num, neu_num]
print(scores)
fig = px.pie(df_pie, values=[pos_num, neg_num, neu_num], names= ['Positive', 'Negative', 'Neutral'], title='MAGA Sentimentient Categories')
fig.show()


import pandas as pd
import plotly.express as px  # (version 4.7.0)
import plotly.graph_objects as go

import dash  # (version 1.12.0) pip install dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

app = dash.Dash(__name__)
server = app.server

# ------------------------------------------------------------------------------
# Import and clean data (importing csv into pandas)

#df = pd.read_csv("AB_NYC_2019.csv")
#df = df.groupby(['name', 'neighbourhood_group', 'neighbourhood', 'room_type', 'latitude', 'longitude'])[['minimum_nights']].mean()
#df.reset_index(inplace=True)
print(df[:5])


# ------------------------------------------------------------------------------
# App layout
app.layout = html.Div([
    html.Div([
        html.Label(['Twitter Hashtag Sentiment Scores']),
        dcc.Dropdown(
            id='my_dropdown',
            options=[
                     {'label': 'hashtag', 'value': 'hashtag'}
            ],
            value='hashtag',
            multi=False,
            clearable=False,
            style={"width": "50%"}
        ),
    ]),

    html.Div([
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
        
    ]),

])

#---------------------------------------------------------------
@app.callback(
    Output(component_id='the_graph', component_property='figure'),
    [Input(component_id='my_dropdown', component_property='value')]
)

def update_graph(figure):

    fig = px.pie(df_pie, values=[pos_num, neg_num, neu_num], 
                 names= ['Positive', 'Negative', 'Neutral'], title='MAGA Sentimentient Categories')
    
    return fig



if __name__ == '__main__':
    app.run_server(debug=False)
