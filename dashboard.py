import dash
from dash import dcc, html, Input, Output
import pandas as pd
import plotly.express as px

# ⚠️ Carica il tuo dataset
clean_tweet = pd.read_csv("clean_tweet.csv")
original_tweet = pd.read_csv("tweet_originali.csv")
original_tweet = original_tweet.merge(
    clean_tweet[['tweet_id', 'sentiment_gold']],
    on='tweet_id',
    how='left'
)
tfidf_df = pd.read_csv("tf_idf_scores.csv", sep=",")
shap_df = pd.read_csv("shap_values_flat.csv")

film_colors = {
    'deadpool': '#2b7cff',
    'suicidesquad': '#ff4d4d'
}

sentiment_labels = {
    -1: 'Negativo',
     0: 'Neutro',
     1: 'Positivo'
}

# Inizializzazione app Dash
#app = dash.Dash(__name__)
app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "Dashboard Sentiment Analysis"

# Layout della dashboard
app.layout = html.Div([
    html.H1("Sentiment Analysis: Deadpool vs Suicide Squad", style={
        'textAlign': 'center',
        'color': '#2b7cff',
        'marginTop': '30px',
        'marginBottom': '10px'
    }),

    html.Div([
        html.Label("Seleziona Film:", style={'textAlign': 'center', 'fontWeight': 'bold'}),
        dcc.Dropdown(
            id='film-dropdown',
            options=[
                {'label': 'Deadpool', 'value': 'deadpool'},
                {'label': 'Suicide Squad', 'value': 'suicidesquad'},
                {'label': 'Entrambi', 'value': 'both'}
            ],
            value='both',
            clearable=False,
            style={'width': '300px', 'margin': 'auto'}
        )
    ], style={'textAlign': 'center', 'marginTop': '10px', 'marginBottom': '30px'}),

    html.Div([
        dcc.Tabs(
            id='tabs',
            value='tab-overview',
            children=[
                dcc.Tab(label='Panoramica generale', value='tab-overview'),
                dcc.Tab(label='Lessico e frequenze', value='tab-wordcloud'),
                dcc.Tab(label='Explainability & Performance', value='tab-extra'),
            ],
            style={'width': '80%', 'margin': 'auto', 'fontWeight': 'bold', 'fontSize': '17px'},
            colors={
                "border": "white",
                "primary": "#2b7cff",
                "background": "#f9f9f9"
            }
        )
        ,
        html.Div(id='tabs-content', style={'padding': '30px'})
    ]),
    #################################
         ],
        style={
            'fontFamily': 'Arial, sans-serif',
            'backgroundColor': '#ffffff'})


# Callback per aggiornare i grafici e il tweet
@app.callback(
    Output('shap-both-chart', 'figure'),
    Input('sentiment-selector', 'value')
)
def update_shap_both(sentiment_value):
    df = shap_df[shap_df['sentiment'] == sentiment_value]

    # Qui mettiamo l'aggregazione
    df_agg = df.groupby(['word', 'film', 'sentiment']).agg({'shap_value': 'mean'}).reset_index()

    # Ora usiamo df_agg invece di df
    # Prendi top 15 parole per importanza assoluta SHAP
    df_top = df_agg.loc[df_agg['shap_value'].abs().sort_values(ascending=False).index].head(15)

    # Ordina parole per valore SHAP
    df_top = df_top.sort_values('shap_value')

    ordered_words = df_top['word'].tolist()

    fig = px.scatter(
        df_top,
        x='word',
        y='shap_value',
        size=df_top['shap_value'].abs(),
        color='film',
        color_discrete_map=film_colors,
        title=f"Top 15 parole influenti ({sentiment_labels[sentiment_value]})",
        labels={'shap_value': 'Valore SHAP', 'word': 'Parola'},
        hover_name='word',
        category_orders={'word': ordered_words}
    )
    fig.update_layout(
        xaxis_title='Parola',
        yaxis_title='Valore SHAP',
        xaxis_tickangle=45,
        height=600
    )

    return fig

@app.callback(
    Output('shap-single-chart', 'figure'),
    [Input('sentiment-selector', 'value'),
     Input('film-dropdown', 'value')]
)
def update_shap_single(sentiment_value, selected_film):
    # Filtra solo per il film selezionato e sentiment
    df = shap_df[(shap_df['sentiment'] == sentiment_value) & (shap_df['film'] == selected_film)]

    # Aggrega
    df_agg = df.groupby(['word', 'film', 'sentiment']).agg({'shap_value': 'mean'}).reset_index()

    # Prendi top 15 parole
    df_top = df_agg.loc[df_agg['shap_value'].abs().sort_values(ascending=False).index].head(15)

    df_top = df_top.sort_values('shap_value')

    ordered_words = df_top['word'].tolist()

    fig = px.scatter(
        df_top,
        x='word',
        y='shap_value',
        size=df_top['shap_value'].abs(),
        color_discrete_sequence=['#2b7cff'] if selected_film == 'deadpool' else ['#ff4d4d'],
        title=f"Top 15 parole influenti ({selected_film.capitalize()}, {sentiment_labels[sentiment_value]})",
        labels={'shap_value': 'Valore SHAP', 'word': 'Parola'},
        hover_name='word',
        category_orders={'word': ordered_words}
    )
    fig.update_layout(
        xaxis_title='Parola',
        yaxis_title='Valore SHAP',
        xaxis_tickangle=45,
        height=600
    )
    return fig



@app.callback(
    Output('tabs-content', 'children'),
    [Input('film-dropdown', 'value'),
     Input('tabs', 'value')]
)
def update_tab_content(selected_film, selected_tab):
    # Filtro dataset
    if selected_film == 'both':
        filtered_df = clean_tweet[clean_tweet['sentiment_gold'].isin([-1, 0, 1])]
        filtered_df_originali = original_tweet
    else:
        filtered_df = clean_tweet[(clean_tweet['keyword'] == selected_film) & (clean_tweet['sentiment_gold'].isin([-1, 0, 1]))]
        filtered_df_originali = original_tweet[original_tweet['keyword'] == selected_film]

    ### === TAB 1: Panoramica generale === ###
    if selected_tab == 'tab-overview':
        # Mappa sentiment in stringhe
        filtered_df['sentiment_label_str'] = filtered_df['sentiment_gold'].map({
            -1: 'Negativo (-1)',
             0: 'Neutro (0)',
             1: 'Positivo (+1)'
        })
        # Conta tweet totali in base alla selezione
        tweet_count = len(filtered_df)

        if selected_film == 'both':
            info_box = html.P(f"Numero totale di tweet analizzati: {tweet_count}", style={'fontWeight': 'bold'})
        elif selected_film == 'deadpool':
            info_box = html.P(f"Numero di tweet su Deadpool: {tweet_count}",
                              style={'fontWeight': 'bold', 'color': 'red'})
        elif selected_film == 'suicidesquad':
            info_box = html.P(f"Numero di tweet su Suicide Squad: {tweet_count}",
                              style={'fontWeight': 'bold', 'color': 'purple'})

        # Frase insight per film singolo
        if selected_film in ['deadpool', 'suicidesquad']:
            film_df = filtered_df.copy()
            total = len(film_df)

            perc_pos = round((film_df['sentiment_gold'] == 1).sum() / total * 100, 1) if total > 0 else 0
            perc_neg = round((film_df['sentiment_gold'] == -1).sum() / total * 100, 1) if total > 0 else 0
            perc_neu = round((film_df['sentiment_gold'] == 0).sum() / total * 100, 1) if total > 0 else 0

            insight_single = html.P(
                f"Il {perc_pos}% dei tweet su {selected_film.capitalize()} è positivo, "
                f"{perc_neg}% negativo e {perc_neu}% neutro.",
                style={'fontWeight': 'bold', 'fontSize': '16px', 'color': '#444', 'textAlign': 'center'}
            )
        else:
            insight_single = None  # Nessuna frase se confronto (cioè selected_film == 'both')

        # Dati per grafico a torta
        pie_df = filtered_df['sentiment_gold'].value_counts().reindex([-1, 0, 1], fill_value=0).reset_index()
        pie_df.columns = ['sentiment_gold', 'count']
        pie_df['label'] = pie_df['sentiment_gold'].map({
            -1: 'Negativo (-1)',
            0: 'Neutro (0)',
            1: 'Positivo (+1)'
        })
        fig_pie = px.pie(
            pie_df,
            names='label',
            values='count',
            title=f"Distribuzione percentuale dei sentiment per {selected_film.capitalize() if selected_film != 'both' else 'entrambi i film'}",
            color='label',
            color_discrete_map={
                'Negativo (-1)': 'red',
                'Neutro (0)': 'gray',
                'Positivo (+1)': 'green'
            }
        )
        fig_pie.update_traces(textinfo='percent+label')



        # Conta tweet per categoria (già come stringa)
        sentiment_counts = filtered_df['sentiment_label_str'].value_counts().reindex(
            ['Negativo (-1)', 'Neutro (0)', 'Positivo (+1)'], fill_value=0
        ).reset_index()
        sentiment_counts.columns = ['sentiment_label_str', 'count']

        fig_sentiment = px.bar(
            sentiment_counts,
            x='sentiment_label_str',
            y='count',
            color='sentiment_label_str',
            color_discrete_map={
                'Negativo (-1)': 'red',
                'Neutro (0)': 'gray',
                'Positivo (+1)': 'green'
            },
            title='Distribuzione dei Tweet per Sentiment'
        )
        fig_sentiment.update_layout(bargap=0.2)

        # Tweet campione
        #tweet_sample = filtered_df_originali.sample(1)['tweet_text'].values[0]
        tweet_row = filtered_df_originali.sample(1)
        tweet_text = tweet_row['tweet_text'].values[0]
        tweet_sentiment = tweet_row['sentiment_gold'].values[0]

        sentiment_label = {-1: "Negativo", 0: "Neutro", 1: "Positivo"}.get(tweet_sentiment, "N/A")

        # Calcolo distribuzione tweet per film e sentiment
        comparison_df = clean_tweet[clean_tweet['sentiment_gold'].isin([-1, 0, 1])]
        comparison_df = comparison_df[comparison_df['keyword'].isin(['deadpool', 'suicidesquad'])]

        comparison_df['sentiment_str'] = comparison_df['sentiment_gold'].map({
            -1: 'Negativo (-1)',
            0: 'Neutro (0)',
            1: 'Positivo (+1)'
        })

        grouped = comparison_df.groupby(['sentiment_str', 'keyword']).size().reset_index(name='count')

        fig_comparison = px.bar(
            grouped,
            x='sentiment_str',
            y='count',
            color='keyword',
            barmode='group',
            title="Confronto sentiment tra Deadpool e Suicide Squad",
            color_discrete_map={
                'deadpool': 'red',
                'suicidesquad': 'purple'
            }
        )

        # Percentuali di sentiment per ciascun film
        pivot_df = grouped.pivot(index='sentiment_str', columns='keyword', values='count').fillna(0)

        # Calcola percentuali per ogni film
        deadpool_total = pivot_df['deadpool'].sum()
        ss_total = pivot_df['suicidesquad'].sum()

        # Estrai % positive
        pos_deadpool = round((pivot_df.loc['Positivo (+1)', 'deadpool'] / deadpool_total) * 100,
                             1) if deadpool_total > 0 else 0
        pos_ss = round((pivot_df.loc['Positivo (+1)', 'suicidesquad'] / ss_total) * 100, 1) if ss_total > 0 else 0

        # Estrai % negative
        neg_deadpool = round((pivot_df.loc['Negativo (-1)', 'deadpool'] / deadpool_total) * 100,
                             1) if deadpool_total > 0 else 0
        neg_ss = round((pivot_df.loc['Negativo (-1)', 'suicidesquad'] / ss_total) * 100, 1) if ss_total > 0 else 0

        if pos_deadpool > pos_ss:
            insight = f"Deadpool ha una percentuale più alta di tweet positivi ({pos_deadpool}%) rispetto a Suicide Squad ({pos_ss}%)."
        else:
            insight = f"Suicide Squad ha una percentuale più alta di tweet positivi ({pos_ss}%) rispetto a Deadpool ({pos_deadpool}%)."

        if neg_deadpool > neg_ss:
            insight2 = f"Inoltre, Deadpool ha ricevuto più tweet negativi ({neg_deadpool}%) rispetto a Suicide Squad ({neg_ss}%)."
        else:
            insight2 = f"Inoltre, Suicide Squad ha ricevuto più tweet negativi ({neg_ss}%) rispetto a Deadpool ({neg_deadpool}%)."


        return html.Div([
            info_box,
            html.Div([
                dcc.Graph(figure=fig_sentiment)
            ], style={'width': '70%', 'margin': 'auto'}),

            html.Div([
                dcc.Graph(figure=fig_pie)
            ], style={'width': '50%', 'margin': 'auto', 'paddingTop': '10px'}),


            html.Div(insight_single) if insight_single else html.Div(),

            html.Div([
                dcc.Graph(figure=fig_comparison)
            ], style={'width': '70%', 'margin': 'auto'}),

            html.Div([
                html.P(insight, style={'fontWeight': 'bold', 'fontSize': '16px', 'color': 'green'}),
                html.P(insight2, style={'fontWeight': 'bold', 'fontSize': '16px', 'color': 'crimson'})
            ], style={'width': '70%', 'margin': 'auto', 'paddingTop': '10px', 'textAlign': 'center'}),

            html.H3("Tweet Campione"),
            html.Div([
                html.P(tweet_text, style={'marginBottom': '10px'}),
                html.P(f"Sentiment: {sentiment_label}", style={'fontWeight': 'bold'})
            ], style={
                'backgroundColor': '#f9f9f9',
                'padding': '15px',
                'border': '1px solid #ddd',
                'borderRadius': '5px'
            })
        ])

    ### === TAB 3: Wordcloud & Emoji === ###
    elif selected_tab == 'tab-wordcloud':
        # Calcolo varietà lessicale per ciascun sentiment (solo se è disponibile 'clean_text')
        if 'tweet_text' in filtered_df.columns:
            lexical_variety = filtered_df.groupby('sentiment_gold')['tweet_text'].apply(
                lambda texts: texts.apply(lambda t: len(set(t.split()))).mean()
            ).reindex([-1, 0, 1]).reset_index()

            lexical_variety.columns = ['sentiment', 'unique_word_avg']
            sentiment_map = {-1: 'Negativo', 0: 'Neutro', 1: 'Positivo'}
            lexical_variety['sentiment'] = lexical_variety['sentiment'].map(sentiment_map)

            fig_lexical = px.bar(
                lexical_variety,
                x='sentiment',
                y='unique_word_avg',
                color='sentiment',
                color_discrete_map={'Negativo': 'red', 'Neutro': 'gray', 'Positivo': 'green'},
                title='Varietà Lessicale Media per Sentiment',
                labels={'unique_word_avg': 'Parole Uniche (media)'}
            )
            fig_lexical.update_layout(showlegend=False)
        else:
            fig_lexical = None

        if selected_film == 'both':
            # Filtro: entrambe i film, solo sentiment validi
            filtered_tfidf_both = tfidf_df[
                (tfidf_df['film'].isin(['deadpool', 'suicidesquad'])) &
                (tfidf_df['tfidf_score'] > 0)
                ]

            # Prendi top parole per ciascun film (es: 15 + 15)
            top_deadpool = filtered_tfidf_both[filtered_tfidf_both['film'] == 'deadpool'].nlargest(15, 'tfidf_score')
            top_suicidesquad = filtered_tfidf_both[filtered_tfidf_both['film'] == 'suicidesquad'].nlargest(15,
                                                                                                           'tfidf_score')

            top_both = pd.concat([top_deadpool, top_suicidesquad])

            # Crea bubble chart
            fig_bubble_both = px.scatter(
                top_both,
                x=top_both.index,
                y='tfidf_score',
                size='tfidf_score',
                color='film',
                hover_name='term',
                text='term',
                color_discrete_map={'deadpool': '#2ca02c', 'suicidesquad': '#1f77b4'},
                title="Confronto parole più rappresentative (TF-IDF)"
            )
            fig_bubble_both.update_traces(textposition='top center')
            fig_bubble_both.update_layout(
                xaxis=dict(showticklabels=False),
                yaxis_title='TF-IDF Score',
                showlegend=True,
                height=500
            )

            # Prepara dizionari per risultati multipli
            figs_by_sentiment = {}

            # Colori
            film_colors = {'deadpool': '#2ca02c', 'suicidesquad': '#1f77b4'}
            sentiment_labels = {-1: 'Negativo', 0: 'Neutro', 1: 'Positivo'}

            for sentiment_value in [-1, 0, 1]:
                filtered_sentiment = tfidf_df[
                    (tfidf_df['film'].isin(['deadpool', 'suicidesquad'])) &
                    (tfidf_df['sentiment'] == sentiment_value)
                    ]

                # Top 10 parole per film
                top_dp = filtered_sentiment[filtered_sentiment['film'] == 'deadpool'].nlargest(10, 'tfidf_score')
                top_ss = filtered_sentiment[filtered_sentiment['film'] == 'suicidesquad'].nlargest(10, 'tfidf_score')
                top_terms = pd.concat([top_dp, top_ss])

                # Bubble chart
                fig = px.scatter(
                    top_terms,
                    x=top_terms.index,
                    y='tfidf_score',
                    size='tfidf_score',
                    color='film',
                    hover_name='term',
                    text='term',
                    color_discrete_map=film_colors,
                    title=f"Parole {sentiment_labels[sentiment_value]} – Deadpool vs Suicide Squad"
                )
                fig.update_traces(textposition='top center')
                fig.update_layout(
                    xaxis=dict(showticklabels=False),
                    yaxis_title='TF-IDF Score',
                    showlegend=True,
                    height=450
                )

                figs_by_sentiment[sentiment_value] = fig

            return html.Div([
                html.H3("Wordcloud dei Tweet"),
                html.Div([
                    html.Img(src='/assets/wordcloudDP.png', style={'width': '48%', 'marginRight': '2%'}),
                    html.Img(src='/assets/wordcloudSS.png', style={'width': '48%'})
                ], style={'display': 'flex'}),

                html.H3("Hashtag più usati"),
                html.Div([
                    html.Img(src='/assets/hashtagDP.png', style={'width': '48%', 'marginRight': '2%'}),
                    html.Img(src='/assets/hashtagSS.png', style={'width': '48%'})
                ], style={'display': 'flex'}),

                html.H3("Analisi delle Emoji"),
                html.Div([
                    html.Img(src='/assets/positive_emoji.png', style={'width': '30%', 'marginRight': '5%'}),
                    html.Img(src='/assets/neutral_emoji.png', style={'width': '30%', 'marginRight': '5%'}),
                    html.Img(src='/assets/negative_emoji.png', style={'width': '30%'})
                ], style={'display': 'flex'}),

                html.H3("Confronto per Sentiment tra i due film"),

                html.Div([
                    html.H4("Sentiment Positivo", style={'textAlign': 'center'}),
                    dcc.Graph(figure=figs_by_sentiment[1]),

                    html.H4("Sentiment Neutro", style={'textAlign': 'center'}),
                    dcc.Graph(figure=figs_by_sentiment[0]),

                    html.H4("Sentiment Negativo", style={'textAlign': 'center'}),
                    dcc.Graph(figure=figs_by_sentiment[-1]),
                ])

            ])

        elif selected_film == 'deadpool':
            # Filtra per Deadpool
            filtered_tfidf = tfidf_df[(tfidf_df['film'] == 'deadpool') & (tfidf_df['tfidf_score'] > 0)]

            top_terms = filtered_tfidf.sort_values(by='tfidf_score', ascending=False).head(40)

            # Mappa sentiment in etichette e colori
            sentiment_map = {-1: 'Negativo', 0: 'Neutro', 1: 'Positivo'}
            color_map = {'Negativo': 'red', 'Neutro': 'gray', 'Positivo': 'green'}
            top_terms['sentiment_label'] = top_terms['sentiment'].map(sentiment_map)

            fig_bubble = px.scatter(
                top_terms,
                x=top_terms.index,
                y='tfidf_score',
                size='tfidf_score',
                color='sentiment_label',
                hover_name='term',
                text='term',
                color_discrete_map=color_map,
                title='Parole con TF-IDF più alto – Deadpool'
            )
            fig_bubble.update_traces(textposition='top center')
            fig_bubble.update_layout(
                xaxis=dict(showticklabels=False),
                yaxis_title='TF-IDF Score',
                showlegend=True,
                height=500
            )
            return html.Div([
                html.H3("Wordcloud e Hashtag - Deadpool"),
                html.Div([
                    html.Img(src='/assets/wordcloudDP.png', style={'width': '48%', 'marginRight': '4%'}),
                    html.Img(src='/assets/hashtagDP.png', style={'width': '48%'})
                ], style={'display': 'flex', 'justifyContent': 'center'}),

                html.H3("Analisi delle Emoji - Deadpool"),
                html.Div([
                    html.Img(src='/assets/positive_emoji.png', style={'width': '30%', 'marginRight': '5%'}),
                    html.Img(src='/assets/neutral_emoji.png', style={'width': '30%', 'marginRight': '5%'}),
                    html.Img(src='/assets/negative_emoji.png', style={'width': '30%'})
                ], style={'display': 'flex'}),

                html.H3("Termini rappresentativi per Sentiment - Deadpool"),
                
                dcc.Graph(figure=fig_bubble),
                html.H3("Wordcloud per Sentiment - Deadpool", style={'textAlign': 'center', 'marginTop': '40px'}),
                html.Div([
                    html.Div([
                        html.Img(src='/assets/wordcloudpositiveDP.png', style={'width': '100%'}),
                        html.P("Positivo", style={'textAlign': 'center', 'fontWeight': 'bold', 'color': 'green'})
                    ], style={'width': '32%', 'marginRight': '2%'}),
                    html.Div([
                        html.Img(src='/assets/wordcloudneutralDP.png', style={'width': '100%'}),
                        html.P("Neutro", style={'textAlign': 'center', 'fontWeight': 'bold', 'color': 'gray'})
                    ], style={'width': '32%', 'marginRight': '2%'}),
                    html.Div([
                        html.Img(src='/assets/wordcloudnegativeDP.png', style={'width': '100%'}),
                        html.P("Negativo", style={'textAlign': 'center', 'fontWeight': 'bold', 'color': 'red'})
                    ], style={'width': '32%'})
                ], style={'display': 'flex', 'justifyContent': 'space-between', 'marginBottom': '40px'}),

            ])

        elif selected_film == 'suicidesquad':
            # Filtra per Suicide Squad
            filtered_tfidf = tfidf_df[(tfidf_df['film'] == 'suicidesquad') & (tfidf_df['tfidf_score'] > 0)]
            top_terms = filtered_tfidf.sort_values(by='tfidf_score', ascending=False).head(40)

            # Mappa sentiment in etichette e colori
            sentiment_map = {-1: 'Negativo', 0: 'Neutro', 1: 'Positivo'}
            color_map = {'Negativo': 'red', 'Neutro': 'gray', 'Positivo': 'green'}
            top_terms['sentiment_label'] = top_terms['sentiment'].map(sentiment_map)

            fig_bubble = px.scatter(
                top_terms,
                x=top_terms.index,
                y='tfidf_score',
                size='tfidf_score',
                color='sentiment_label',
                hover_name='term',
                text='term',
                color_discrete_map=color_map,
                title='Parole con TF-IDF più alto – Suicide Squad'
            )
            fig_bubble.update_traces(textposition='top center')
            fig_bubble.update_layout(
                xaxis=dict(showticklabels=False),
                yaxis_title='TF-IDF Score',
                showlegend=True,
                height=500
            )
            return html.Div([
                html.H3("Wordcloud e hashtag - Suicide Squad"),
                html.Div([
                    html.Img(src='/assets/wordcloudSS.png', style={'width': '48%', 'marginRight': '4%'}),
                    html.Img(src='/assets/hashtagSS.png', style={'width': '48%'})
                ], style={'display': 'flex', 'justifyContent': 'center'}),

                html.H3("Analisi delle Emoji - Suicide Squad"),
                html.Div([
                    html.Img(src='/assets/positive_emoji.png', style={'width': '30%', 'marginRight': '5%'}),
                    html.Img(src='/assets/neutral_emoji.png', style={'width': '30%', 'marginRight': '5%'}),
                    html.Img(src='/assets/negative_emoji.png', style={'width': '30%'})
                ], style={'display': 'flex'}),

                html.H3("Termini rappresentativi per Sentiment - Suicide Squad"),
                dcc.Graph(figure=fig_bubble),

                html.H3("Wordcloud per Sentiment - Deadpool", style={'textAlign': 'center', 'marginTop': '40px'}),
                html.Div([
                    html.Div([
                        html.Img(src='/assets/wordcloudpositiveSS.png', style={'width': '100%'}),
                        html.P("Positivo", style={'textAlign': 'center', 'fontWeight': 'bold', 'color': 'green'})
                    ], style={'width': '32%', 'marginRight': '2%'}),
                    html.Div([
                        html.Img(src='/assets/wordcloudneutralSS.png', style={'width': '100%'}),
                        html.P("Neutro", style={'textAlign': 'center', 'fontWeight': 'bold', 'color': 'gray'})
                    ], style={'width': '32%', 'marginRight': '2%'}),
                    html.Div([
                        html.Img(src='/assets/wordcloudnegativeSS.png', style={'width': '100%'}),
                        html.P("Negativo", style={'textAlign': 'center', 'fontWeight': 'bold', 'color': 'red'})
                    ], style={'width': '32%'})
                ], style={'display': 'flex', 'justifyContent': 'space-between', 'marginBottom': '40px'}),

            ])



    ### === TAB 4: Extra === ###
    elif selected_tab == 'tab-extra':

        tweet_samples = []

        # Mostra i tweet campione solo se il film è uno specifico
        if selected_film in ['deadpool', 'suicidesquad']:
            sentimenti = [-1, 0, 1]
            colori = {-1: 'red', 0: 'gray', 1: 'green'}
            etichette = {-1: 'Negativo ', 0: 'Neutro ', 1: 'Positivo '}

            for s in sentimenti:
                df_sample = filtered_df[filtered_df['sentiment_gold'] == s]

                if df_sample.empty:
                    continue

                tweet = df_sample.sample(1).iloc[0]['tweet_text']

                tweet_div = html.Div([
                    html.H4(etichette[s], style={'color': colori[s]}),
                    html.P(tweet, style={
                        'borderLeft': f'5px solid {colori[s]}',
                        'paddingLeft': '10px',
                        'backgroundColor': '#f9f9f9',
                        'borderRadius': '5px',
                        'padding': '10px'
                    })
                ], style={'marginBottom': '20px'})

                tweet_samples.append(tweet_div)

        # SHAP dropdown
        dropdown = html.Div([
            html.H3("Spiegabilità: SHAP Overview", style={'marginTop': '40px'}),
            html.Label("Seleziona Sentiment:"),
            dcc.Dropdown(
                id='sentiment-selector',
                options=[
                    {'label': 'Positivo (+1)', 'value': 1},
                    {'label': 'Neutro (0)', 'value': 0},
                    {'label': 'Negativo (-1)', 'value': -1}
                ],
                value=1,
                clearable=False,
                style={'width': '200px'}
            ),
            html.Br()
        ])

        # Grafico diverso in base al film
        graph_component = (
            dcc.Graph(id='shap-both-chart')
            if selected_film == 'both'
            else dcc.Graph(id='shap-single-chart')
        )

        # ✅ Costruzione del layout finale (tweet solo se presenti)
        children = []

        if tweet_samples:
            children.append(
                html.Div([
                    html.H3("Tweet rappresentativi per ciascun Sentiment", style={'textAlign': 'center'}),
                    *tweet_samples
                ])
            )

        children.append(dropdown)
        children.append(graph_component)

        if selected_film == 'both':
            # Dati delle performance (puoi anche caricarli da CSV se preferisci)
            performance_data = pd.DataFrame({
                'film': ['deadpool', 'deadpool', 'deadpool', 'suicidesquad', 'suicidesquad', 'suicidesquad'],
                'modello': ['AFINN', 'Logistic Regression', 'BERT', 'AFINN', 'Logistic Regression', 'BERT'],
                'accuracy': [0.41, 0.70, 0.75, 0.39, 0.58, 0.65]
            })

            fig_perf = px.bar(
                performance_data,
                x='modello',
                y='accuracy',
                color='film',
                barmode='group',
                title='Confronto Performance dei Modelli',
                #text='accuracy',
                color_discrete_map={
                    'deadpool': '#2b7cff',
                    'suicidesquad': '#ff4d4d'
                }
            )
            #fig_perf.update_traces(texttemplate='%{text:.2%}', textposition='outside')
            fig_perf.update_layout(
                yaxis_tickformat='.0%',
                yaxis_title='Accuracy',
                xaxis_title='Modello',
                margin=dict(t=50, b=50),
                bargap = 0.5
            )

            children.append(
                html.Div([
                    html.H3("Performance dei modelli", style={'textAlign': 'center', 'marginTop': '60px'}),
                    dcc.Graph(figure=fig_perf)
                ])
            )

        elif selected_film in ['deadpool', 'suicidesquad']:
            performance_data = pd.DataFrame({
                'modello': ['AFINN', 'Logistic Regression', 'BERT'],
                'accuracy': [0.41, 0.70, 0.75] if selected_film == 'deadpool' else [0.39, 0.58, 0.65]
            })

            fig_perf = px.bar(
                performance_data,
                x='modello',
                y='accuracy',
                color='modello',
                title=f'Performance dei modelli – {selected_film.capitalize()}',
                color_discrete_map={
                    'AFINN': 'gray',
                    'Logistic Regression': '#2b7cff',
                    'BERT': '#00cc96'
                }
            )

            fig_perf.update_layout(
                yaxis_tickformat='.0%',
                yaxis_title='Accuracy',
                xaxis_title='Modello',
                showlegend=False,
                margin=dict(t=50, b=50),
                bargap=0.5
            )

            children.append(
                html.Div([
                    html.H3("Performance dei modelli", style={'textAlign': 'center', 'marginTop': '60px'}),
                    dcc.Graph(figure=fig_perf)
                ])
            )
            if selected_film=='deadpool':
                children.append(
                    html.Div([
                        html.H3("Matrice di Confusione - Deadpool",
                                style={'textAlign': 'center', 'marginTop': '40px'}),
                        html.Img(src='/assets/confmatrixDP.png',
                                 style={'width': '40%', 'display': 'block', 'margin': 'auto'})
                    ])
                )
            elif selected_film == 'suicidesquad':
                children.append(
                    html.Div([
                        html.H3("Matrice di Confusione - Suicide Squad",
                                style={'textAlign': 'center', 'marginTop': '40px'}),
                        html.Img(src='/assets/confmatrixSS.png',
                                 style={'width': '40%', 'display': 'block', 'margin': 'auto'})
                    ])
                )

        return html.Div(children)


# Esecuzione server
if __name__ == '__main__':
    app.run(debug=True)
