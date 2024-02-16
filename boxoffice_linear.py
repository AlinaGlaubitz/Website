# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.


from dash import Dash, dcc, html, callback_context
import plotly.express as px
import plotly.graph_objs as go
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.colors as mcolors
from datetime import date 
import holidays 
import pickle
from datetime import datetime, timedelta
from dash.dependencies import Input, Output
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def create_dash_application(flask_app):

    # Global variable declaration
    global pred_linear_covid, pred_linear_nocovid, actual_data, actual_dates, forecast_dates, df
    pred_linear_covid = None
    pred_linear_nocovid = None
    actual_data = None
    actual_dates=None
    forecast_dates=None
    df = None

    dash_app = Dash(__name__, server=flask_app, url_base_pathname='/dashboard/')
    
    colors = {
        'background': '#1f2c56',
        'text': '#FFFFFF'
    }

    # Data download from database
    df = pd.read_csv('static/csv_files/boxoffice_data.csv')

    # Drop daily and weekly change
    df.drop(columns=['DailyChange','WeeklyChange'],inplace=True)


    # 911 date was in the wrong format 
    date_wrong = df[df.Date == 'Sep 119 2001'].index.values[0]
    df.loc[date_wrong,'Date'] = 'Sep 11 2001'

    # Transform the dates into datetime format
    df['Date'] = pd.to_datetime(df['Date'], format='%b %d %Y')

    # Create new column for the year
    df['Year'] = df['Date'].dt.year

    # Create new column for the year
    df['Weekday_Plot'] = df['Weekday']

    # Convert integer values to float for interpolation
    df['DailyGross'] = df['DailyGross'].astype(float)
    df['NoReleases'] = df['NoReleases'].astype(float)

    # Sort by date
    df.sort_values(by='Date',ascending=False,inplace=True)


    # Create a new column 'DaysAsNo1'
    df['DaysAsNo1'] = df.groupby('No1Movie').cumcount() + 1

    # Reset the count when the movie changes
    df['DaysAsNo1'] = df.groupby((df['No1Movie'] != df['No1Movie'].shift(1)).cumsum()).cumcount() + 1

    df = pd.get_dummies(df, columns=['Weekday'],dtype='int')

    # Create list of holidays for the two largest contributors to box office
    holidays_china = holidays.China(years = range(2002,2025)).items() 
    holidays_us = holidays.UnitedStates(years = range(2002,2025)).items() 

    # Create column in pd dateformat
    df['DateTime'] =[pd.to_datetime(day).date() for day in df['Date']]

    # Sort by date
    df.sort_values(by='Date',ascending=True,inplace=True)

    # Dates for the forecast
    start_date = df[df.Weekday_Sunday==1].iloc[len(df[df.Weekday_Sunday==1])-1].Date
    print(start_date)
    forecast_dates = [start_date + timedelta(days=i) for i in range(7)]
    end_date = df.iloc[len(df)-1].Date
    # Create an empty list to store the dates
    actual_dates =  []
    actual_data = []

    # Iterate through the dates and add them to the list
    current_date = start_date
    while current_date <= end_date:
        actual_dates.append(current_date)
        actual_data.append(df[df['Date']==current_date].DailyGross.values[0])
        current_date += timedelta(days=1)

    # Initialize columns for holidays
    df['Holiday_US'] = 0
    df['Holiday_China'] = 0


    # One-hot encode holidays
    for holiday in holidays_us:
        df.loc[df['DateTime'] == holiday[0], 'Holiday_US'] = 1

    for holiday in holidays_china:
        df.loc[df['DateTime'] == holiday[0], 'Holiday_China'] = 1

    # 1. Create dataframe for next week's forecast
    # Initialize an empty list to store data
    num_days = 7
    data_list = []

    # Iterate through the next 'num_days' days
    for i in range(num_days):
        # Calculate the date for the current day
        current_date = start_date + pd.Timedelta(days=i)
        # Calculate sin/cos values for half-weekly and half-yearly cycles
        sin_half_week = np.sin(2 * np.pi * current_date.dayofweek / 3.5)
        cos_half_week = np.cos(2 * np.pi * current_date.dayofweek / 3.5)
        sin_half_year = np.sin(2 * np.pi * current_date.dayofyear / 365.25 * 2)
        cos_half_year = np.cos(2 * np.pi * current_date.dayofyear / 365.25 * 2)
        
        # Check if the current date is a holiday in the US or China
        is_holiday_us = 1 if current_date in holidays_us else 0
        is_holiday_china = 1 if current_date in holidays_china else 0
        
        # Append the data to the list
        data_list.append({
            'DayOfYear': current_date.dayofyear,
            'Weekday': current_date.strftime('%A'),
            'sin_half_week': sin_half_week,
            'cos_half_week': cos_half_week,
            'sin_half_year': sin_half_year,
            'cos_half_year': cos_half_year,
            'Holiday_US': is_holiday_us,
            'Holiday_China': is_holiday_china,
            'Year': current_date.year,
        })


    # Convert the list of dictionaries to a DataFrame
    test_df = pd.DataFrame(data_list)
    test_df = pd.get_dummies(test_df, columns=['Weekday'],dtype='int')

    # Create training data 
    train_df_covid = df[df.Date < start_date]
    train_df_covid = df[df.Date >= pd.to_datetime('Jan 1 2002')]

    # Initialize columns for holidays
    train_df_covid['Holiday_US'] = 0
    train_df_covid['Holiday_China'] = 0


    # One-hot encode holidays
    for holiday in holidays_us:
        train_df_covid.loc[df['Date'] == holiday[0], 'Holiday_US'] = 1

    for holiday in holidays_china:
        train_df_covid.loc[df['Date'] == holiday[0], 'Holiday_China'] = 1

    # Half-weekly cycle: Using sine and cosine transformations with a period of 3.5 days (half a week)
    train_df_covid['sin_half_week'] = np.sin(2 * np.pi * train_df_covid.Date.dt.dayofweek / 3.5)
    train_df_covid['cos_half_week'] = np.cos(2 * np.pi * train_df_covid.Date.dt.dayofweek / 3.5)

    # Half-yearly cycle: Using sine and cosine transformations with a period of 182.5 days (half a year)
    day_of_year = train_df_covid.DayOfYear
    train_df_covid['sin_half_year'] = np.sin(2 * np.pi * day_of_year / 365.25 * 2)
    train_df_covid['cos_half_year'] = np.cos(2 * np.pi * day_of_year / 365.25 * 2)

    # Create training data 
    start_2020 = train_df_covid[train_df_covid.Date == 'Jan 1 2020'].index.values[0]
    start_2022 = train_df_covid[train_df_covid.Date == 'Jan 1 2022'].index.values[0]
    if start_2020 < start_2022:
        train_df_nocovid = train_df_covid.drop(index=range(start_2020,start_2022))
    else:
        train_df_nocovid = train_df_covid.drop(index=range(start_2022-1,start_2020+1))


    # Building the SARIMAX model
    # Including the half-weekly cycle
    exog_features = ['Weekday_Monday', 'Weekday_Tuesday', 'Weekday_Wednesday', 'Weekday_Thursday', 
                    'Weekday_Friday', 'Weekday_Saturday', 'Weekday_Sunday', 'sin_half_week', 
                    'cos_half_week', 'sin_half_year', 
                    'cos_half_year','Holiday_US','Holiday_China']

    # Linear Regression - No COVID ------------------------------------------------------
    # Selecting features and target
    X_train = train_df_nocovid[['DayOfYear','Year','Weekday_Monday', 'Weekday_Tuesday', 'Weekday_Wednesday', 'Weekday_Thursday', 
                    'Weekday_Friday', 'Weekday_Saturday', 'Weekday_Sunday']]
    y_train = train_df_nocovid['DailyGross']

    # Model Training
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Save the model to a file
    with open('linear_nocovid.pkl', 'wb') as model_file:
        pickle.dump(model, model_file)

    # Getting the prediction data
    X_test = test_df[['DayOfYear','Year','Weekday_Monday', 'Weekday_Tuesday', 'Weekday_Wednesday', 'Weekday_Thursday', 
                    'Weekday_Friday', 'Weekday_Saturday', 'Weekday_Sunday']]

    # Load the model from the file
    with open('linear_nocovid.pkl', 'rb') as model_file:
        linear_nocovid = pickle.load(model_file)

    # Making predictions for the last week
    pred_linear_nocovid = linear_nocovid.predict(X_test)

    # Linear Regression - COVID ------------------------------------------------------
    X_train = train_df_covid[['DayOfYear','Year','Weekday_Monday', 'Weekday_Tuesday', 'Weekday_Wednesday', 'Weekday_Thursday', 
                    'Weekday_Friday', 'Weekday_Saturday', 'Weekday_Sunday']]
    y_train = train_df_covid['DailyGross']

    # Model Training
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Save the model to a file
    with open('linear_covid.pkl', 'wb') as model_file:
        pickle.dump(model, model_file)

    # Load the model from the file
    with open('linear_covid.pkl', 'rb') as model_file:
        linear_covid = pickle.load(model_file)

    # Making predictions for the last week
    pred_linear_covid = linear_covid.predict(X_test)

    predictions = pred_linear_covid

    # Calculate RMSE

    rmse = 0 if actual_data == [] else np.sqrt(mean_squared_error(actual_data,predictions[:len(actual_data)]))

    # Create traces
    trace1= go.Scatter(
        x=actual_dates,
        y=actual_data,
        mode='lines+markers',
        name='Actual Box Office',
        line=dict(color='fuchsia', width=4, dash='dash'),
        marker=dict(symbol='circle', size=15)
    )

    trace2 = go.Scatter(
        x=forecast_dates,
        y=predictions,  # Replace with actual column name
        mode='lines+markers',
        name='Predicted Box Office',
        line=dict(color='cornflowerblue', width=4),
        marker=dict(symbol='triangle-up', size=15)
    )


    # Layout settings
    layout_forecast = go.Layout(
        title='Forecast vs Actual',
        yaxis=dict(title='Box Office', range=[0, 120000000]),  # 0 to 100M
        showlegend=True
    )
    title = f"Box Office Forecast - Linear Regression with COVID data  (RMSE = {rmse})"#round(rmse,2)
    # Create the figures for your plots
    fig1 = go.Figure(data=[trace1, trace2], layout=layout_forecast)  # For SARIMAX model predictions vs actual
    fig1.update_layout(xaxis =  {'showgrid': False},
                            yaxis = {'showgrid': False})
    fig1.update_layout(
        yaxis=dict(range=[0, 120000000]),
        legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01),
        plot_bgcolor='rgba(0,0,0,0.2)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_family="Times New Roman",
        font_color="white",
        font_size = 20,
        title=title
        )

    df['Weekday'] = df['Date'].dt.weekday

    # Filter data for the last 52 weeks
    one_year_ago = datetime.now() - timedelta(weeks=52)
    filtered_df = df[df['Date'] > one_year_ago]

    # Calculate average box office for each weekday
    avg_box_office = filtered_df.groupby('Weekday')['DailyGross'].mean()

    # If needed, you can reorder the avg_box_office Series to have Sunday first
    avg_box_office = avg_box_office.reindex([6, 0, 1, 2, 3, 4, 5])

    # Current week's data
    current_week_box_office = [0 for k in range(7)]
    for k in range(len(actual_data)):
        current_week_box_office[k] = actual_data[k]

    # Plotting
    trace3 = go.Bar(
        x=['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'],
        y=avg_box_office,
        name='Average',
        marker=dict(color='cornflowerblue')
    )

    trace4 = go.Bar(
        x=['Sun','Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'],
        y=current_week_box_office,
        name='Current Week',
        marker=dict(color='fuchsia')
    )

    layout = go.Layout(
        title='Average Daily Box Office vs Current Week',
        yaxis=dict(title='Box Office'),
        barmode='group'
    )


    fig2 = go.Figure(data=[trace3, trace4], layout=layout)  # For average daily box office vs current week
    fig2.update_layout(legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01
    ))

    fig2.update_layout(
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    )

    trace5 = go.Scatter(
        x=df[df['Year']==2023].Date,
        y=df[df['Year']==2023].DailyGross,  # Replace with actual column name
        mode='lines+markers',
        name='Box Office in 2023',
        line=dict(color='cornflowerblue', width=1)
    )

    # Layout settings
    layout = go.Layout(
        title='Daily Box Office in the last year',
        yaxis=dict(title='Box Office', range=[0, 120000000]),  # 0 to 100M
        showlegend=True
    )

    fig3 = go.Figure(data=[trace5], layout=layout) 
    fig3.update_layout(legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01
    ))
    fig3.update_layout(
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    )


    # Calculate the sum of box office for the last week
    weekly_box_office = np.array(actual_data).sum()

    # Get the current No1Movie details
    current_no1_movie = df[-1:][['No1Movie', 'GrossOfNo1Movie']]

    # Calculate the sum of box office for the last week and compare to previous week
    #last_week_box_office_sum = df[-7:]['DailyGross'].sum()
    #previous_week_box_office_sum = df[-14:-7]['DailyGross'].sum()

    # Get the current No1Movie details
    current_no1_movie = df[-1:][['No1Movie', 'GrossOfNo1Movie', 'DaysAsNo1']]

    # Styling for the title and subtitle
    title_style = {
        'textAlign': 'center',
        'color': '#e944d6',
        'marginTop': '20px',
        'fontSize': '36px'
    }
    subtitle_style = {
        'textAlign': 'center',
        'color': '#e944d6',
        'fontSize': '24px'
    }

    # Styling for the card containers
    card_container_style = {
        'textAlign': 'center',
        'color': 'white',
        'backgroundColor': 'rgb(0,0,0,0.2)',
        'padding': '10px',
        'borderRadius': '5px',
        'margin': '10px'
    }

    # Update app layout
    dash_app.layout = html.Div(style={'backgroundColor':'#1f2c56'}, children=[
        html.Div(id='last-update-time', style={'textAlign': 'right', 'color': 'white', 'marginTop': '20px', 'marginRight': '20px'}),
        html.H1('Box Office Trends and Predictions - A Movie Dashboard with Daily Data and Weekly Forecast Updates', style=title_style),

    # Upper Section
        html.Div([
            # Upper Left - Main Forecast Plot
            # Upper Section of the App Layout
            html.Div([
                # Left Section with Main Forecast Plot and Dropdowns
                html.Div([
                    html.Div([
                        # Title for the button-container
                        html.H3('Select Model and Dataset for Forecast', style={'textAlign': 'center', 'color': 'white'}),

                        # Button-container
                        html.Div([
                            # Button for model selection
                            html.Div([
                                dcc.Dropdown(
                                    id='model-selector',
                                    options=[
                                        {'label': 'Linear Regression', 'value': 'LR'}
                                    ],
                                    value='LR'
                                )
                            ], style={'width': '45%', 'textAlign': 'center', 'display': 'inline-block'}),

                            # Button for dataset selection
                            html.Div([
                                dcc.Dropdown(
                                    id='dataset-selector',
                                    options=[
                                        {'label': 'Include COVID data', 'value': 'DS1'},
                                        {'label': 'Exclude COVID data', 'value': 'DS2'}
                                    ],
                                    value='DS1'
                                )
                            ], style={'width': '45%', 'textAlign': 'center', 'display': 'inline-block'}),
                        ], style={'width': '33%', 'margin': '0 auto', 'display': 'flex', 'justifyContent': 'space-around', 'backgroundColor': 'rgba(255, 255, 255, 0)'}),
                    
                    ], style={'backgroundColor': colors['background']}),



                    # Forecast Graph
                    dcc.Graph(id='forecast-graph', figure=fig1)
                ], style={'width': '100%', 'display': 'inline-block'}),
            ], style={'width': '75%', 'display': 'flex'}),


            # Upper Right - Weekly Box Office and No1 Movie Info
            html.Div([
                html.Div([
                    html.H3('Weekly Box Office', style={'color': 'white'}),
                    html.P(f"${weekly_box_office:,}",id='weekly-box-office', style={'color': '#e944d6', 'fontSize': '40px'}),
                ], style=card_container_style),
                html.Div([
                    html.H3('No1 Movie', style={'color': 'white'}),
                    html.P(current_no1_movie['No1Movie'], id='no1-movie-title', style={'color': '#e944d6', 'fontSize': '40px'}),
                    html.P(f"Daily Gross: ${current_no1_movie['GrossOfNo1Movie'].values[0]:,}", id='no1-movie-gross', style={'color': 'white', 'fontSize': '15px','marginTop': '-18px'}),
                    #html.P(f"Days as No1: {current_no1_movie['DaysAsNo1'].values[0]}", id='no1-movie-days', style={'color': 'white', 'fontSize': '15px'})
                ], style=card_container_style),    
            ], style={'width': '25%', 'display': 'inline-block', 'verticalAlign': 'top'})
        ], style={'display': 'flex'}),


        # Lower Section
        html.Div([
            # Lower Left - Daily Average Plot
            html.Div([dcc.Graph(id='daily-average-graph', figure=fig2)], style={'width': '50%', 'display': 'inline-block'}),


            html.Div([
            # Container for the dropdown and graph
            html.Div([
                # Dropdown for Year Selection
                dcc.Dropdown(
                    id='year-selector',
                    options=[{'label': str(year), 'value': year} for year in range(2002, 2024)],
                    value=2023,  # Default value
                    style={
                        'position': 'absolute',  # Positioning the dropdown
                        'top': '10px',  # Adjust the top position as needed
                        'right': '10px',  # Adjust the right position as needed
                        'width': '200px',  # Dropdown width
                        'zIndex': '999'  # Ensure dropdown is above the graph
                    }
                ),

                # Plot for Yearly Box Office Data
                dcc.Graph(
                    id='yearly-box-office-graph',
                    style={'width': '100%'}  # Ensure graph occupies the full container width
                )
            ], style={
                'position': 'relative',  # Relative positioning context for the container
                'width': '100%'  # Container width
            }),

            ], style={'width': '75%','display': 'flex', 'marginTop': '0px'}),
        ], style={'display': 'flex', 'marginTop': '-50px'}),

        # Regular code updates (once daily)
        dcc.Interval(
            id='interval-update-data',
            interval=1*3600*1000*24,  # in milliseconds, e.g., hourly updates
            n_intervals=0
        )
    ])


    @dash_app.callback(
        Output('yearly-box-office-graph', 'figure'),
        [Input('year-selector', 'value')]
    )
    def update_yearly_box_office(selected_year):
        filtered_df = df[df['Year'] == selected_year]
        trace = go.Scatter(
            x=filtered_df.Date,
            y=filtered_df.DailyGross,
            mode='lines+markers',
            name=f'Box Office in {selected_year}',
            line=dict(color='cornflowerblue', width=1)
        )
        layout = go.Layout(
            title=f'Daily Box Office in {selected_year}',
            yaxis=dict(title='Box Office', range=[0, 150000000]),
            showlegend=True
        )
        title = f'Daily Box Office in {selected_year}'
        fig = go.Figure(data=[trace], layout=layout)
        fig.update_layout(xaxis =  {'showgrid': False},
                            yaxis = {'showgrid': False})
        fig.update_layout(
            yaxis=dict(range=[0, 120000000]),
            legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01),
            plot_bgcolor='rgba(0,0,0,0.2)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_family="Times New Roman",
            font_color="white",
            font_size = 20,
            title=title
            )
        return fig



    @dash_app.callback(
        Output('forecast-graph', 'figure'),
        [Input('interval-update-data', 'n_intervals'),
        Input('model-selector', 'value'),
        Input('dataset-selector', 'value')]
    )
    def update_forecast_plot(n_intervals, selected_model, selected_dataset):
        global pred_linear_covid, pred_linear_nocovid, actual_data, actual_dates, forecast_dates, df
        # Callback context
        ctx = callback_context

        if ctx.triggered and ctx.triggered[0]['prop_id'].split('.')[0] == 'interval-update-data':
            # Data download
            df = pd.read_csv('static/csv_files/boxoffice_data.csv')

            # Convert box office to float values
            df['DailyGross'] = df['DailyGross'].astype(float)

            # 911 date was in the wrong format 
            date_wrong = df[df.Date == 'Sep 119 2001'].index.values[0]
            df.loc[date_wrong,'Date'] = 'Sep 11 2001'

            # Transform the dates into datetime format
            df['Date'] = pd.to_datetime(df['Date'], format='%b %d %Y')

            # Create new column for the year
            df['Year'] = df['Date'].dt.year

            # Sort by date
            df.sort_values(by='Date',ascending=True,inplace=True)

            # Create a new column 'DaysAsNo1'
            df['DaysAsNo1'] = df.groupby('No1Movie').cumcount() + 1

            # Reset the count when the movie changes
            df['DaysAsNo1'] = df.groupby((df['No1Movie'] != df['No1Movie'].shift(1)).cumsum()).cumcount() + 1

            if df.iloc[0].Weekday == 'Sunday':
                # Extract the date from the first row of df
                start_date = df.iloc[0]['Date']

                # Calculate holidays for the specified range of years
                holidays_china = dict(holidays.China(years=range(2002, 2025)).items())
                holidays_us = dict(holidays.UnitedStates(years=range(2002, 2025)).items())

                # Initialize an empty list to store data
                data_list = []

                # Number of days to include in test_df
                num_days = 7

                # Iterate through the next 'num_days' days
                for i in range(num_days):
                    # 1. Create dataframe for next week's forecast
                    # Calculate the date for the current day
                    current_date = start_date + pd.Timedelta(days=i)
                    
                    # Calculate sin/cos values for half-weekly and half-yearly cycles
                    sin_half_week = np.sin(2 * np.pi * current_date.dayofweek / 3.5)
                    cos_half_week = np.cos(2 * np.pi * current_date.dayofweek / 3.5)
                    sin_half_year = np.sin(2 * np.pi * current_date.dayofyear / 365.25 * 2)
                    cos_half_year = np.cos(2 * np.pi * current_date.dayofyear / 365.25 * 2)
                    
                    # Check if the current date is a holiday in the US or China
                    is_holiday_us = 1 if current_date in holidays_us else 0
                    is_holiday_china = 1 if current_date in holidays_china else 0
                    
                    # Append the data to the list
                    data_list.append({
                        'DayOfYear': current_date.dayofyear,
                        'Weekday': current_date.strftime('%A'),
                        'sin_half_week': sin_half_week,
                        'cos_half_week': cos_half_week,
                        'sin_half_year': sin_half_year,
                        'cos_half_year': cos_half_year,
                        'Holiday_US': is_holiday_us,
                        'Holiday_China': is_holiday_china,
                        'Year': current_date.year,
                    })

                # Convert the list of dictionaries to a DataFrame
                test_df = pd.DataFrame(data_list)

                test_df = pd.get_dummies(test_df, columns=['Weekday'],dtype='int')

                # 2. Generate predictions and traces for all four types of forecasts
                # Making predictions for the last week (including the exogenous variables)
                # Target variable
                # Create training data 
                train_df_covid = df[df.Date < start_date]
                train_df_covid = train_df_covid[df.Date >= pd.to_datetime('Jan 1 2002')]

                # Initialize columns for holidays
                train_df_covid['Holiday_US'] = 0
                train_df_covid['Holiday_China'] = 0


                # One-hot encode holidays
                for holiday in holidays_us:
                    train_df_covid.loc[df['Date'] == holiday[0], 'Holiday_US'] = 1

                for holiday in holidays_china:
                    train_df_covid.loc[df['Date'] == holiday[0], 'Holiday_China'] = 1

                # Half-weekly cycle: Using sine and cosine transformations with a period of 3.5 days (half a week)
                train_df_covid['sin_half_week'] = np.sin(2 * np.pi * train_df_covid.Date.dt.dayofweek / 3.5)
                train_df_covid['cos_half_week'] = np.cos(2 * np.pi * train_df_covid.Date.dt.dayofweek / 3.5)

                # Half-yearly cycle: Using sine and cosine transformations with a period of 182.5 days (half a year)
                day_of_year = train_df_covid.DayOfYear
                train_df_covid['sin_half_year'] = np.sin(2 * np.pi * day_of_year / 365.25 * 2)
                train_df_covid['cos_half_year'] = np.cos(2 * np.pi * day_of_year / 365.25 * 2)

                # Create training data 
                start_2020 = train_df_covid[train_df_covid.Date == 'Jan 1 2020'].index.values[0]
                start_2022 = train_df_covid[train_df_covid.Date == 'Jan 1 2022'].index.values[0]
                if start_2020 < start_2022:
                    train_df_nocovid = train_df_covid.drop(index=range(start_2020,start_2022))
                else:
                    train_df_nocovid = train_df_covid.drop(index=range(start_2022-1,start_2020+1))


                # Linear Regression - No COVID ------------------------------------------------------
                X_train = train_df_nocovid[['DayOfYear','Year','Weekday_Monday', 'Weekday_Tuesday', 'Weekday_Wednesday', 'Weekday_Thursday', 
                    'Weekday_Friday', 'Weekday_Saturday', 'Weekday_Sunday']]
                y_train = train_df_nocovid['DailyGross']

                # Model Training
                model = LinearRegression()
                model.fit(X_train, y_train)

                # Save the model to a file
                with open('linear_nocovid.pkl', 'wb') as model_file:
                    pickle.dump(model, model_file)

                # Load the model from the file
                with open('linear_nocovid.pkl', 'rb') as model_file:
                    linear_nocovid = pickle.load(model_file)

                # Data for predictions
                X_test = test_df[['DayOfYear', 'NoReleases', 'Year', 'Weekday_Friday',
                    'Weekday_Monday', 'Weekday_Saturday', 'Weekday_Sunday',
                    'Weekday_Thursday', 'Weekday_Tuesday', 'Weekday_Wednesday']]
                actual_data = []
                actual_dates = []
                forecast_dates = [df.iloc[0].Date + timedelta(days=i) for i in range(7)]

                # Making predictions for the last week
                pred_linear_nocovid = linear_nocovid.predict(X_test)

                # Linear Regression - COVID ------------------------------------------------------
                X_train = train_df_covid[['DayOfYear','Year','Weekday_Monday', 'Weekday_Tuesday', 'Weekday_Wednesday', 'Weekday_Thursday', 
                    'Weekday_Friday', 'Weekday_Saturday', 'Weekday_Sunday']]
                y_train = train_df_covid['DailyGross']

                # Model Training
                model = LinearRegression()
                model.fit(X_train, y_train)

                # Save the model to a file
                with open('linear_covid.pkl', 'wb') as model_file:
                    pickle.dump(model, model_file)

                # Load the model from the file
                with open('linear_covid.pkl', 'rb') as model_file:
                    linear_covid = pickle.load(model_file)
                # Making predictions for the last week
                pred_linear_covid = linear_covid.predict(X_test)

                print('New forecast generated')
            else:
                # Sort by date
                df.sort_values(by='Date',ascending=False,inplace=True)
                if (df.iloc[0].Date) not in (actual_dates):
                    actual_data.append(df.iloc[0].DailyGross)
                    actual_dates.append(df.iloc[0].Date)

                print('New data added')

            predictions = pred_linear_nocovid
        else:
            # Logic to update the plot based on the selected model and dataset
            if selected_model == 'LR':
                if selected_dataset == 'DS1':
                    # Use Linear Regression with Dataset 1 for predictions
                    predictions = pred_linear_covid  # Update with current X_test
                else:
                    # Use Linear Regression with Dataset 2 for predictions
                    predictions = pred_linear_nocovid  # Update with current X_test
        # Create new traces with the updated data
        trace1 = go.Scatter(
            x=actual_dates,  # or relevant Date data
            y=actual_data,
            mode='lines+markers',
            name='Actual Box Office',
            line=dict(color='fuchsia', width=4, dash='dash'),
            marker=dict(symbol='circle', size=15)
        )

        trace2 = go.Scatter(
            x=forecast_dates,  # or relevant Date data
            y=predictions,
            mode='lines+markers',
            name='Predicted Box Office',
            line=dict(color='cornflowerblue', width=4),
            marker=dict(symbol='triangle-up', size=15)
        )

        # Calculate RMSE
        rmse = 0 if actual_data == [] else np.sqrt(mean_squared_error(actual_data,predictions[:len(actual_data)]))

        # Determine the title based on selections
        title = f"Box Office Forecast - {'Linear Regression' if selected_model == 'LR' else selected_model} {'with' if selected_dataset == 'DS1' else 'without'} COVID data (RMSE = {round(rmse,2):,})" # 



        # Create and return the updated figure
        figure = go.Figure(data=[trace1, trace2], layout=layout_forecast)
        figure.update_layout(xaxis =  {'showgrid': False},
                            yaxis = {'showgrid': False})
        figure.update_layout(
            yaxis=dict(range=[0, 120000000]),
            legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01),
            plot_bgcolor='rgba(0,0,0,0.2)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_family="Times New Roman",
            font_color="white",
            font_size = 20,
            title=title
            )
        
        # Sort by date
        df.sort_values(by='Date',ascending=True,inplace=True)

        return figure


    # Callback for updating information
    @dash_app.callback(
        Output('daily-average-graph', 'figure'),
        [Input('interval-update-data', 'n_intervals')]
    )
    def update_daily_average_graph(n):
        global df, actual_data

        df['Weekday'] = df['Date'].dt.weekday

        # Filter data for the last 52 weeks
        one_year_ago = datetime.now() - timedelta(weeks=52)
        filtered_df = df[df['Date'] > one_year_ago]


        # Calculate average box office for each weekday
        avg_box_office = filtered_df.groupby('Weekday').DailyGross.mean()

        # If needed, you can reorder the avg_box_office Series to have Sunday first
        avg_box_office = avg_box_office.reindex([6, 0, 1, 2, 3, 4, 5])

        # Current week's data
        current_week_box_office = [0 for k in range(7)]
        for k in range(len(actual_data)):
            current_week_box_office[k] = actual_data[k]

        # Plotting
        trace3 = go.Bar(
            x=['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'],
            y=avg_box_office,
            name='Average',
            marker=dict(color='cornflowerblue')
        )

        trace4 = go.Bar(
            x=['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'],
            y=current_week_box_office,
            name='Current Week',
            marker=dict(color='fuchsia')
        )

        layout = go.Layout(
            title='Average Daily Box Office vs Current Week',
            yaxis=dict(title='Box Office'),
            barmode='group'
        )

        title = 'Average Daily Box Office (compared to current week)'
        fig2 = go.Figure(data=[trace3, trace4], layout=layout)  # For average daily box office vs current week
        fig2.update_layout(xaxis =  {'showgrid': False},
                            yaxis = {'showgrid': False})
        fig2.update_layout(
            legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01),
            plot_bgcolor='rgba(0,0,0,0.2)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_family="Times New Roman",
            font_color="white",
            font_size = 20,
            title=title
            )

        return fig2


    # Callback for updating information
    @dash_app.callback(
        [Output('weekly-box-office', 'children'),
        Output('no1-movie-title', 'children'),
        Output('no1-movie-gross', 'children')],
        [Input('interval-update-data', 'n_intervals')]
    )
    def update_box_office_info(n):
        global df
        # Your logic to calculate new values
        # For example:
        new_weekly_box_office = np.array(actual_data).sum()
        new_current_no1_movie = df[-1:][['No1Movie', 'GrossOfNo1Movie']].iloc[0]

        # Format the output strings
        weekly_box_office_str = f"$ {new_weekly_box_office}"
        no1_movie_title_str = new_current_no1_movie['No1Movie']
        no1_movie_gross = f"Daily Gross: $ {new_current_no1_movie['GrossOfNo1Movie']:,}"

        return weekly_box_office_str, no1_movie_title_str, no1_movie_gross

    # Callback for 'last-updated'
    @dash_app.callback(
        Output('last-update-time', 'children'),
        [Input('interval-update-data', 'n_intervals')]
    )
    def update_last_update_time(n):
        # Get the current time
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        # Return the formatted string
        return f"Last updated: {current_time}"

    return dash_app
