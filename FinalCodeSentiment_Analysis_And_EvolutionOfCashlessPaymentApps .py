#!/usr/bin/env python
# coding: utf-8

# In[20]:


import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk

nltk.download('vader_lexicon')
 
googlepay_df = pd.read_csv('googlePaySentiments.csv')
phonepe_df = pd.read_csv('PhonePeSentiments.csv')
paytm_df = pd.read_csv('PaytmSentiments.csv')


def display_data_info(df, name):
    print(f"\n{name} Data Info:")
    print(f"Size: {df.size}")
    print(f"Number of Rows: {df.shape[0]}")
    print(f"Number of Columns: {df.shape[1]}")
    
display_data_info(finance_data, "Finance Data")
display_data_info(googlepay_df, "Google Pay")
display_data_info(phonepe_df, "PhonePe")
display_data_info(paytm_df, "Paytm")



# In[ ]:





# In[21]:


#data pre-processing
def drop_columns_if_exist(df, columns):
    for column in columns:
        if column in df.columns:
            df.drop(column, axis=1, inplace=True)
            
drop_columns_if_exist(googlepay_df, ['isEdited', 'developerResponse'])
drop_columns_if_exist(phonepe_df, ['isEdited', 'developerResponse'])
drop_columns_if_exist(paytm_df, ['isEdited', 'developerResponse'])

# EDA 
def eda(df, name):
    print(f"\nEDA for {name}:")
    print(df.info())
    print(df.describe())
    print(df.head())

eda(finance_data, "Finance Data")



# In[22]:


eda(googlepay_df, "Google Pay")


# In[23]:


eda(phonepe_df, "PhonePe")


# In[24]:


eda(paytm_df, "Paytm")


# In[25]:


# Download VADER lexicon
nltk.download('vader_lexicon')

# Initialize VADER sentiment analyzer
sid = SentimentIntensityAnalyzer()

# Function to calculate sentiment score using VADER
def calculate_sentiment_score(text):
    scores = sid.polarity_scores(text)
    return scores['compound']

# Apply sentiment analysis to each dataset
googlepay_df['Sentiment_Score'] = googlepay_df['review'].apply(calculate_sentiment_score)
phonepe_df['Sentiment_Score'] = phonepe_df['review'].apply(calculate_sentiment_score)
paytm_df['Sentiment_Score'] = paytm_df['review'].apply(calculate_sentiment_score)

# Convert 'date' columns to datetime format
googlepay_df['date'] = pd.to_datetime(googlepay_df['date'], dayfirst=True)
phonepe_df['date'] = pd.to_datetime(phonepe_df['date'], dayfirst=True)
paytm_df['date'] = pd.to_datetime(paytm_df['date'], dayfirst=True)

# Calculate daily average sentiment scores
googlepay_daily = googlepay_df[['date', 'Sentiment_Score']].resample('D', on='date').mean().reset_index()
phonepe_daily = phonepe_df[['date', 'Sentiment_Score']].resample('D', on='date').mean().reset_index()
paytm_daily = paytm_df[['date', 'Sentiment_Score']].resample('D', on='date').mean().reset_index()

# Renaming columns 
googlepay_daily.rename(columns={'date': 'Date', 'Sentiment_Score': 'Sentiment_Score_GooglePay'}, inplace=True)
phonepe_daily.rename(columns={'date': 'Date', 'Sentiment_Score': 'Sentiment_Score_PhonePe'}, inplace=True)
paytm_daily.rename(columns={'date': 'Date', 'Sentiment_Score': 'Sentiment_Score_Paytm'}, inplace=True)

# Convert 'Date' column in finance_data to datetime format
finance_data['Date'] = pd.to_datetime(finance_data['Date'], dayfirst=True)

# Merge financial data with sentiment scores
finance_data = pd.merge(finance_data, googlepay_daily, on='Date', how='left')
finance_data = pd.merge(finance_data, phonepe_daily, on='Date', how='left')
finance_data = pd.merge(finance_data, paytm_daily, on='Date', how='left')


# Define date periods for classification
pre_covid_end = pd.Timestamp('2019-12-31')
during_covid_start = pd.Timestamp('2020-01-01')
during_covid_end = pd.Timestamp('2021-12-31')
post_covid_start = pd.Timestamp('2022-01-01')

# Function to classify sentiment based on score
def classify_sentiment(score):
    if score > 0.05:
        return 'Positive'
    elif score < -0.05:
        return 'Negative'
    else:
        return 'Neutral'

# Function to analyze sentiment for each period and app
def analyze_period_sentiment(data, period_name):
    sentiment_results = {}
    for period, df in data.items():
        sentiment_counts = df['Sentiment_Score'].apply(classify_sentiment).value_counts()
        sentiment_results[period] = sentiment_counts.to_dict()
    return {period_name: sentiment_results}

# Segment sentiment data for each app based on periods
googlepay_pre_covid = googlepay_df[googlepay_df['date'] <= pre_covid_end]
googlepay_during_covid = googlepay_df[(googlepay_df['date'] >= during_covid_start) & (googlepay_df['date'] <= during_covid_end)]
googlepay_post_covid = googlepay_df[googlepay_df['date'] >= post_covid_start]

phonepe_pre_covid = phonepe_df[phonepe_df['date'] <= pre_covid_end]
phonepe_during_covid = phonepe_df[(phonepe_df['date'] >= during_covid_start) & (phonepe_df['date'] <= during_covid_end)]
phonepe_post_covid = phonepe_df[phonepe_df['date'] >= post_covid_start]

paytm_pre_covid = paytm_df[paytm_df['date'] <= pre_covid_end]
paytm_during_covid = paytm_df[(paytm_df['date'] >= during_covid_start) & (paytm_df['date'] <= during_covid_end)]
paytm_post_covid = paytm_df[paytm_df['date'] >= post_covid_start]

# Analyze sentiment for each period and app
googlepay_sentiments = analyze_period_sentiment({'PRE COVID': googlepay_pre_covid, 'DURING COVID': googlepay_during_covid, 'POST COVID': googlepay_post_covid}, 'Google Pay')
phonepe_sentiments = analyze_period_sentiment({'PRE COVID': phonepe_pre_covid, 'DURING COVID': phonepe_during_covid, 'POST COVID': phonepe_post_covid}, 'PhonePe')
paytm_sentiments = analyze_period_sentiment({'PRE COVID': paytm_pre_covid, 'DURING COVID': paytm_during_covid, 'POST COVID': paytm_post_covid}, 'Paytm')

# Print sentiment analysis results
print("Google Pay Sentiments:")
print(googlepay_sentiments)
print("\nPhonePe Sentiments:")
print(phonepe_sentiments)
print("\nPaytm Sentiments:")
print(paytm_sentiments)



# In[26]:


# Function to generate word cloud
def generate_word_cloud(text, title, ax):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.set_title(title)
    ax.axis('off')

# Visualization
fig, axes = plt.subplots(3, 3, figsize=(15, 15))

# Google Pay Word Clouds
generate_word_cloud(' '.join(googlepay_pre_covid['review']), 'Google Pay Pre COVID', axes[0, 0])
generate_word_cloud(' '.join(googlepay_during_covid['review']), 'Google Pay During COVID', axes[0, 1])
generate_word_cloud(' '.join(googlepay_post_covid['review']), 'Google Pay Post COVID', axes[0, 2])

# PhonePe Word Clouds
generate_word_cloud(' '.join(phonepe_pre_covid['review']), 'PhonePe Pre COVID', axes[1, 0])
generate_word_cloud(' '.join(phonepe_during_covid['review']), 'PhonePe During COVID', axes[1, 1])
generate_word_cloud(' '.join(phonepe_post_covid['review']), 'PhonePe Post COVID', axes[1, 2])

# Paytm Word Clouds
generate_word_cloud(' '.join(paytm_pre_covid['review']), 'Paytm Pre COVID', axes[2, 0])
generate_word_cloud(' '.join(paytm_during_covid['review']), 'Paytm During COVID', axes[2, 1])
generate_word_cloud(' '.join(paytm_post_covid['review']), 'Paytm Post COVID', axes[2, 2])

plt.tight_layout()
plt.show()


# In[27]:


# Visualization 1
def plot_pie_chart(app_results, app_name, period, ax):
    sentiments = ['Positive', 'Neutral', 'Negative']
    values = [app_results.get(sentiment, 0) for sentiment in sentiments]
    ax.pie(values, labels=sentiments, autopct='%1.1f%%', startangle=140, colors=['lightgreen', 'skyblue', 'lightcoral'])
    ax.set_title(f'{app_name} Sentiment Analysis for {period}')
    ax.axis('equal')

fig, axes = plt.subplots(3, 3, figsize=(15, 15))
for i, (period, results) in enumerate(googlepay_sentiments['Google Pay'].items()):
    plot_pie_chart(results, 'Google Pay', period, axes[0, i])
for i, (period, results) in enumerate(phonepe_sentiments['PhonePe'].items()):
    plot_pie_chart(results, 'PhonePe', period, axes[1, i])
for i, (period, results) in enumerate(paytm_sentiments['Paytm'].items()):
    plot_pie_chart(results, 'Paytm', period, axes[2, i])
plt.tight_layout()
plt.show()



# In[28]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the merged data
finance_data = pd.read_csv('Merged_Finance_Data_with_Sentiment.csv')

# Convert 'Date' column to datetime format
finance_data['Date'] = pd.to_datetime(finance_data['Date'])

# Define date ranges for categorization
periods = {
    'PRE COVID': ((2017, 1, 1), (2019, 12, 31)),
    'DURING COVID': ((2020, 1, 1), (2022, 12, 31)),
    'POST COVID': ((2023, 1, 1), (2024, 12, 31))
}

# Function to categorize periods based on date
def categorize_period(row):
    date = row['Date']
    for period, (start_date, end_date) in periods.items():
        if start_date <= (date.year, date.month, date.day) <= end_date:
            return period
    return None

# Apply categorization to create 'Period' column
finance_data['Period'] = finance_data.apply(categorize_period, axis=1)

# Encode categorical 'Period' variable to numeric
finance_data['Period_Code'] = finance_data['Period'].astype('category').cat.codes

# Select columns for correlation
correlation_columns = ['Sentiment_Score_GooglePay', 'Sentiment_Score_PhonePe', 'Sentiment_Score_Paytm', 
                       'Volume (Mn) By Customers', 'Value (Cr) by Customers', 'Period_Code']
correlation_data = finance_data[correlation_columns]

# Compute the correlation matrix
correlation_matrix = correlation_data.corr()

# Plot the correlation matrix
plt.figure(figsize=(10, 8))
plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='none')
plt.colorbar()
plt.title('Correlation Matrix of Sentiment Scores and Financial Metrics')
plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=45)
plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
plt.tight_layout()
plt.show()

# Print the correlation matrix
print("Correlation Matrix:")
print(correlation_matrix)


# In[29]:


import pandas as pd
import numpy as np
import tensorflow as tf
import random
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

# Set random seed for reproducibility
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)
random.seed(seed)

# Load the dataset
file_path = 'Finance_DataSet.csv'
finance_data = pd.read_csv(file_path)

# Convert Date column to datetime format
finance_data['Date'] = pd.to_datetime(finance_data['Date'], format='%d-%m-%Y')

# Sort data by Date
finance_data.sort_values(by='Date', inplace=True)

# Function to preprocess data for a specific UPI bank
def preprocess_data_for_bank(df, bank_name, column, look_back=1):
    bank_df = df[df['UPI Banks'] == bank_name].copy()
    
    # NaN values
    bank_df.fillna(method='ffill', inplace=True)
    bank_df.fillna(method='bfill', inplace=True)
    
    # Normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(bank_df[column].values.reshape(-1, 1))
    
    # Create the dataset with look_back
    X, Y = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    
    X = np.array(X)
    Y = np.array(Y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    return X, Y, scaler, bank_df

# Preprocess data for each UPI bank
look_back = 3
googlepay_X, googlepay_Y, googlepay_scaler, googlepay_df = preprocess_data_for_bank(finance_data, 'Google Pay', 'Volume (Mn) By Customers', look_back)
paytm_X, paytm_Y, paytm_scaler, paytm_df = preprocess_data_for_bank(finance_data, 'Paytm Payments Bank App', 'Volume (Mn) By Customers', look_back)
phonepe_X, phonepe_Y, phonepe_scaler, phonepe_df = preprocess_data_for_bank(finance_data, 'PhonePe', 'Volume (Mn) By Customers', look_back)

print("Preprocessing completed for all datasets!")

# Function to create and train LSTM model
def create_and_train_model(X, Y, epochs=50, batch_size=32):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(look_back, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X, Y, epochs=epochs, batch_size=batch_size, verbose=2)
    return model

# Train models for each app
googlepay_model = create_and_train_model(googlepay_X, googlepay_Y)
paytm_model = create_and_train_model(paytm_X, paytm_Y)
phonepe_model = create_and_train_model(phonepe_X, phonepe_Y)

# Function to make predictions and inverse transform
def make_predictions(model, X, scaler):
    predictions = model.predict(X)
    predictions = scaler.inverse_transform(predictions)
    return predictions

# Function to forecast future periods for coming year's
def forecast_future_periods(model, last_data, scaler, start_year=2025, end_year=2028):
    future_predictions = []
    data = last_data
    
    # Calculate number of periods from start_year to end_year
    periods = (end_year - start_year + 1) * 12
    
    for _ in range(periods):
        prediction = model.predict(data.reshape(1, look_back, 1))
        future_predictions.append(prediction[0, 0])
        data = np.roll(data, -1)
        data[-1] = prediction
    
    future_predictions = np.array(future_predictions).reshape(-1, 1)
    future_predictions = scaler.inverse_transform(future_predictions)
    return future_predictions

# Forecast future periods 
googlepay_last_data = googlepay_X[-1, :, 0]
paytm_last_data = paytm_X[-1, :, 0]
phonepe_last_data = phonepe_X[-1, :, 0]

googlepay_future = forecast_future_periods(googlepay_model, googlepay_last_data, googlepay_scaler, 2025, 2028)
paytm_future = forecast_future_periods(paytm_model, paytm_last_data, paytm_scaler, 2025, 2028)
phonepe_future = forecast_future_periods(phonepe_model, phonepe_last_data, phonepe_scaler, 2025, 2028)

# Calculate total forecasted usage
googlepay_total_future = np.sum(googlepay_future)
paytm_total_future = np.sum(paytm_future)
phonepe_total_future = np.sum(phonepe_future)

# Print the total forecasted usage
print(f"Total forecasted usage for Google Pay : {googlepay_total_future:.2f} Mn")
print(f"Total forecasted usage for Paytm : {paytm_total_future:.2f} Mn")
print(f"Total forecasted usage for PhonePe : {phonepe_total_future:.2f} Mn")

# Determine the most used app
usage_totals = {
    'Google Pay': googlepay_total_future,
    'Paytm': paytm_total_future,
    'PhonePe': phonepe_total_future
}




# function to create future dates
def create_future_dates(start_year, end_year):
    start_date = pd.Timestamp(f'{start_year}-01-01')
    end_date = pd.Timestamp(f'{end_year}-12-31')
    return pd.date_range(start=start_date, end=end_date, freq='M')

# Create future dates for 2025 and further
future_dates = create_future_dates(2025, 2028)

# Function to predict past values
def predict_past(model, X, scaler):
    past_predictions = model.predict(X)
    past_predictions = scaler.inverse_transform(past_predictions)
    return past_predictions

# Predict past values
googlepay_past = predict_past(googlepay_model, googlepay_X, googlepay_scaler)
paytm_past = predict_past(paytm_model, paytm_X, paytm_scaler)
phonepe_past = predict_past(phonepe_model, phonepe_X, phonepe_scaler)



# In[30]:


most_used_app = max(usage_totals, key=usage_totals.get)
print(f"The most used app in future will be : {most_used_app}")


# In[31]:


# Plotting the actual, past predicted, and future predicted data
plt.figure(figsize=(15, 15))

# Google Pay
plt.subplot(3, 1, 1)
plt.plot(googlepay_df['Date'], googlepay_df['Volume (Mn) By Customers'], label='Actual Google Pay', color='blue')
plt.plot(googlepay_df['Date'][look_back:], googlepay_past.squeeze(), label='Past Predicted Google Pay', color='orange')
plt.plot(future_dates, googlepay_future.squeeze(), label='Future Predicted Google Pay', color='green')
plt.fill_between(future_dates, googlepay_future.squeeze(), color='green', alpha=0.1)
plt.title('Google Pay UPI Volume Prediction')
plt.xlabel('Date')
plt.ylabel('Volume (Mn) By Customers')
plt.legend()

# Paytm
plt.subplot(3, 1, 2)
plt.plot(paytm_df['Date'], paytm_df['Volume (Mn) By Customers'], label='Actual Paytm', color='blue')
plt.plot(paytm_df['Date'][look_back:], paytm_past.squeeze(), label='Past Predicted Paytm', color='orange')
plt.plot(future_dates, paytm_future.squeeze(), label='Future Predicted Paytm', color='green')
plt.fill_between(future_dates, paytm_future.squeeze(), color='green', alpha=0.1)
plt.title('Paytm UPI Volume Prediction')
plt.xlabel('Date')
plt.ylabel('Volume (Mn) By Customers')
plt.legend()

# PhonePe
plt.subplot(3, 1, 3)
plt.plot(phonepe_df['Date'], phonepe_df['Volume (Mn) By Customers'], label='Actual PhonePe', color='blue')
plt.plot(phonepe_df['Date'][look_back:], phonepe_past.squeeze(), label='Past Predicted PhonePe', color='orange')
plt.plot(future_dates, phonepe_future.squeeze(), label='Future Predicted PhonePe', color='green')
plt.fill_between(future_dates, phonepe_future.squeeze(), color='green', alpha=0.1)
plt.title('PhonePe UPI Volume Prediction')
plt.xlabel('Date')
plt.ylabel('Volume (Mn) By Customers')
plt.legend()

plt.tight_layout()
plt.show()


# In[32]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Load the dataset
file_path = 'Finance_DataSet.csv'
finance_data = pd.read_csv(file_path)

# Convert Date column to datetime format
finance_data['Date'] = pd.to_datetime(finance_data['Date'], format='%d-%m-%Y')

# Sort data by Date
finance_data.sort_values(by='Date', inplace=True)

# Print first few rows of data to verify loading
print("Data Sample:")
print(finance_data.head())

# Function to fit ARIMA model, forecast future values, and predict past values
def fit_arima_forecast(df, bank_name, column, forecast_years=4):
    # Filter data for the specified bank
    bank_df = df[df['UPI Banks'] == bank_name].copy()
    bank_df.set_index('Date', inplace=True)
    bank_df.sort_index(inplace=True)
    
    # Fill missing values
    bank_df.fillna(method='ffill', inplace=True)
    bank_df.fillna(method='bfill', inplace=True)
    
    # Define the ARIMA model
    model = ARIMA(bank_df[column], order=(1, 1, 1))  # ARIMA(p, d, q) parameters
    results = model.fit()
    
    # Forecast future periods
    forecast_periods = forecast_years * 12  # Number of months to forecast
    forecast = results.get_forecast(steps=forecast_periods)
    forecast_index = pd.date_range(start=bank_df.index[-1] + pd.DateOffset(months=1), periods=forecast_periods, freq='M')
    
    forecast_df = forecast.summary_frame()
    forecast_df.index = forecast_index
    
    # Predict past values (historical data)
    past_forecast = results.get_prediction(start=bank_df.index[0], end=bank_df.index[-1])
    past_forecast_df = past_forecast.summary_frame()
    
    # Ensure the index is aligned with the historical data
    past_forecast_df.index = bank_df.index
    
    return past_forecast_df, forecast_df, results

# Forecast future periods and predict past values for each bank
googlepay_past_forecast, googlepay_forecast, googlepay_model = fit_arima_forecast(finance_data, 'Google Pay', 'Volume (Mn) By Customers')
paytm_past_forecast, paytm_forecast, paytm_model = fit_arima_forecast(finance_data, 'Paytm Payments Bank App', 'Volume (Mn) By Customers')
phonepe_past_forecast, phonepe_forecast, phonepe_model = fit_arima_forecast(finance_data, 'PhonePe', 'Volume (Mn) By Customers')

# Calculate total forecasted usage
googlepay_total_future = googlepay_forecast['mean'].sum()
paytm_total_future = paytm_forecast['mean'].sum()
phonepe_total_future = phonepe_forecast['mean'].sum()

# Print the total forecasted usage
print(f"Total forecasted usage for Google Pay : {googlepay_total_future:.2f} Mn")
print(f"Total forecasted usage for Paytm : {paytm_total_future:.2f} Mn")
print(f"Total forecasted usage for PhonePe : {phonepe_total_future:.2f} Mn")

# Determine the most used app in future
usage_totals = {
    'Google Pay': googlepay_total_future,
    'Paytm': paytm_total_future,
    'PhonePe': phonepe_total_future
}

most_used_app = max(usage_totals, key=usage_totals.get)
print(f"The most used app in future : {most_used_app}")



# In[33]:


# Plotting the actual, past predicted, and future forecasted data
plt.figure(figsize=(15, 15))

# Google Pay
plt.subplot(3, 1, 1)
plt.plot(finance_data[finance_data['UPI Banks'] == 'Google Pay']['Date'], 
         finance_data[finance_data['UPI Banks'] == 'Google Pay']['Volume (Mn) By Customers'], 
         label='Actual Google Pay', color='blue')
plt.plot(googlepay_past_forecast.index, googlepay_past_forecast['mean'], label='Past Predicted Google Pay', color='orange')
plt.plot(googlepay_forecast.index, googlepay_forecast['mean'], label='Future Forecasted Google Pay', color='green')
plt.fill_between(googlepay_forecast.index, googlepay_forecast['mean'], color='green', alpha=0.1)
plt.title('Google Pay UPI Volume Prediction')
plt.xlabel('Date')
plt.ylabel('Volume (Mn) By Customers')
plt.legend()

# Paytm
plt.subplot(3, 1, 2)
plt.plot(finance_data[finance_data['UPI Banks'] == 'Paytm Payments Bank App']['Date'], 
         finance_data[finance_data['UPI Banks'] == 'Paytm Payments Bank App']['Volume (Mn) By Customers'], 
         label='Actual Paytm', color='blue')
plt.plot(paytm_past_forecast.index, paytm_past_forecast['mean'], label='Past Predicted Paytm', color='orange')
plt.plot(paytm_forecast.index, paytm_forecast['mean'], label='Future Forecasted Paytm', color='green')
plt.fill_between(paytm_forecast.index, paytm_forecast['mean'], color='green', alpha=0.1)
plt.title('Paytm UPI Volume Prediction')
plt.xlabel('Date')
plt.ylabel('Volume (Mn) By Customers')
plt.legend()

# PhonePe
plt.subplot(3, 1, 3)
plt.plot(finance_data[finance_data['UPI Banks'] == 'PhonePe']['Date'], 
         finance_data[finance_data['UPI Banks'] == 'PhonePe']['Volume (Mn) By Customers'], 
         label='Actual PhonePe', color='blue')
plt.plot(phonepe_past_forecast.index, phonepe_past_forecast['mean'], label='Past Predicted PhonePe', color='orange')
plt.plot(phonepe_forecast.index, phonepe_forecast['mean'], label='Future Forecasted PhonePe', color='green')
plt.fill_between(phonepe_forecast.index, phonepe_forecast['mean'], color='green', alpha=0.1)
plt.title('PhonePe UPI Volume Prediction')
plt.xlabel('Date')
plt.ylabel('Volume (Mn) By Customers')
plt.legend()

plt.tight_layout()
plt.show()



# In[34]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Load the dataset
file_path = 'Finance_DataSet.csv'
finance_data = pd.read_csv(file_path)

# Convert Date column to datetime format
finance_data['Date'] = pd.to_datetime(finance_data['Date'], format='%d-%m-%Y')

# Sort data by Date
finance_data.sort_values(by='Date', inplace=True)

# Print first few rows of data to verify loading
print("Data Sample:")
print(finance_data.head())

# Function to fit SARIMA model and forecast future values
def fit_sarima_forecast(df, bank_name, column, forecast_years=4):
    # Filter data for the specified bank
    bank_df = df[df['UPI Banks'] == bank_name].copy()
    bank_df.set_index('Date', inplace=True)
    bank_df.sort_index(inplace=True)
    
    # Fill missing values
    bank_df.fillna(method='ffill', inplace=True)
    bank_df.fillna(method='bfill', inplace=True)
    
    # Define the SARIMA model
    model = SARIMAX(bank_df[column], 
                    order=(1, 1, 1),   # ARIMA(p, d, q) parameters
                    seasonal_order=(1, 1, 1, 12))  # SARIMA(p, d, q, s) parameters
    results = model.fit(disp=False)
    
    # Forecast future periods
    forecast_periods = forecast_years * 12  # Number of months to forecast
    forecast = results.get_forecast(steps=forecast_periods)
    forecast_index = pd.date_range(start=bank_df.index[-1] + pd.DateOffset(months=1), periods=forecast_periods, freq='M')
    
    forecast_df = forecast.summary_frame()
    forecast_df.index = forecast_index
    
    # Generate (past) predictions
    past_forecast = results.get_prediction(start=bank_df.index[0], end=bank_df.index[-1])
    past_forecast_df = past_forecast.summary_frame()
    
    return past_forecast_df, forecast_df, results

# Forecast future periods and get past predictions for each bank
googlepay_past_forecast, googlepay_forecast, googlepay_model = fit_sarima_forecast(finance_data, 'Google Pay', 'Volume (Mn) By Customers')
paytm_past_forecast, paytm_forecast, paytm_model = fit_sarima_forecast(finance_data, 'Paytm Payments Bank App', 'Volume (Mn) By Customers')
phonepe_past_forecast, phonepe_forecast, phonepe_model = fit_sarima_forecast(finance_data, 'PhonePe', 'Volume (Mn) By Customers')

# Calculate total forecasted usage
googlepay_total_future = googlepay_forecast['mean'].sum()
paytm_total_future = paytm_forecast['mean'].sum()
phonepe_total_future = phonepe_forecast['mean'].sum()

# Print the total forecasted usage
print(f"Total forecasted usage for Google Pay: {googlepay_total_future:.2f} Mn")
print(f"Total forecasted usage for Paytm: {paytm_total_future:.2f} Mn")
print(f"Total forecasted usage for PhonePe: {phonepe_total_future:.2f} Mn")

# Determine the most used app
usage_totals = {
    'Google Pay': googlepay_total_future,
    'Paytm': paytm_total_future,
    'PhonePe': phonepe_total_future
}

most_used_app = max(usage_totals, key=usage_totals.get)
print(f"The most used app will be: {most_used_app}")



# In[35]:


# Plotting the actual, past predictions, and forecasted data
plt.figure(figsize=(15, 15))

# Google Pay
plt.subplot(3, 1, 1)
plt.plot(finance_data[finance_data['UPI Banks'] == 'Google Pay']['Date'], 
         finance_data[finance_data['UPI Banks'] == 'Google Pay']['Volume (Mn) By Customers'], 
         label='Actual Google Pay', color='blue')
plt.plot(googlepay_past_forecast.index, googlepay_past_forecast['mean'], label='Past Predictions Google Pay', color='orange', linestyle='--')
plt.plot(googlepay_forecast.index, googlepay_forecast['mean'], label='Forecasted Google Pay', color='green')
plt.fill_between(googlepay_forecast.index, googlepay_forecast['mean'], color='green', alpha=0.1)
plt.title('Google Pay UPI Volume Future Prediction')
plt.xlabel('Date')
plt.ylabel('Volume (Mn) By Customers')
plt.legend()



# In[36]:


# Paytm
plt.figure(figsize=(15, 15))
plt.subplot(3, 1, 2)
plt.plot(finance_data[finance_data['UPI Banks'] == 'Paytm Payments Bank App']['Date'], 
         finance_data[finance_data['UPI Banks'] == 'Paytm Payments Bank App']['Volume (Mn) By Customers'], 
         label='Actual Paytm', color='blue')
plt.plot(paytm_past_forecast.index, paytm_past_forecast['mean'], label='Past Predictions Paytm', color='orange', linestyle='--')
plt.plot(paytm_forecast.index, paytm_forecast['mean'], label='Forecasted Paytm', color='green')
plt.fill_between(paytm_forecast.index, paytm_forecast['mean'], color='green', alpha=0.1)
plt.title('Paytm UPI Volume Future Prediction')
plt.xlabel('Date')
plt.ylabel('Volume (Mn) By Customers')
plt.legend()



# In[37]:


# PhonePe
plt.figure(figsize=(15, 15))
plt.subplot(3, 1, 3)
plt.plot(finance_data[finance_data['UPI Banks'] == 'PhonePe']['Date'], 
         finance_data[finance_data['UPI Banks'] == 'PhonePe']['Volume (Mn) By Customers'], 
         label='Actual PhonePe', color='blue')
plt.plot(phonepe_past_forecast.index, phonepe_past_forecast['mean'], label='Past Predictions PhonePe', color='orange', linestyle='--')
plt.plot(phonepe_forecast.index, phonepe_forecast['mean'], label='Forecasted PhonePe', color='green')
plt.fill_between(phonepe_forecast.index, phonepe_forecast['mean'], color='green', alpha=0.1)
plt.title('PhonePe UPI Volume Future Prediction')
plt.xlabel('Date')
plt.ylabel('Volume (Mn) By Customers')
plt.legend()

plt.tight_layout()
plt.show()


# In[ ]:




