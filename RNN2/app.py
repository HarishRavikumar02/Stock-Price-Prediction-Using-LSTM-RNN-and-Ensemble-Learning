from flask import Flask, render_template, request
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, SimpleRNN, Dropout
import tensorflow as tf
import plotly.graph_objs as go
from plotly.subplots import make_subplots

app = Flask(__name__)
tf.config.threading.set_intra_op_parallelism_threads(1)

@app.route('/')
@app.route('/home')
def login():
    return render_template('login1.html')

@app.route('/predict')
def index():
    return render_template('index.html')

# @app.route('/', methods=['GET', 'POST'])
@app.route('/prediction', methods=['GET', 'POST'])
def predict():
    symbol = request.form['symbol']
    start_date = request.form['start_date']
    end_date = request.form['end_date']
    days = 100

    # Download the stock data from Yahoo Finance
    df = yf.download(symbol, start_date, end_date)

    # Preprocess the data
    data = df.filter(['Close'])
    dataset = data.values
    training_data_len = len(dataset) - days
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(dataset)

    x_train = []
    y_train = []

    for i in range(days, len(scaled_data)):
        x_train.append(scaled_data[i-days:i, 0])
        y_train.append(scaled_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Train the LSTM model
    lstm_model = Sequential()
    lstm_model.add(LSTM(64, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    lstm_model.add(LSTM(32, return_sequences=False))
    lstm_model.add(Dense(32, activation='relu'))
    lstm_model.add(Dropout(0.2))
    lstm_model.add(Dense(1))

    lstm_model.compile(optimizer='adam', loss='mean_squared_error')

    lstm_model.fit(x_train, y_train, batch_size=55, epochs=1)

    # Train the RNN model
    rnn_model = Sequential()
    rnn_model.add(SimpleRNN(64, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    rnn_model.add(SimpleRNN(32, return_sequences=False))
    rnn_model.add(Dense(32,activation='relu'))
    lstm_model.add(Dropout(0.2))
    rnn_model.add(Dense(1))

    rnn_model.compile(optimizer='adam', loss='mean_squared_error')

    rnn_model.fit(x_train, y_train, batch_size=55, epochs=1)
    
    
    # Combining LSTM AND RNN
    #Training the Hybrid
    combined_model = Sequential()
    combined_model.add(LSTM(64, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    combined_model.add(LSTM(32, return_sequences=True))
    combined_model.add(SimpleRNN(32, return_sequences=False))
    combined_model.add(Dense(32, activation='relu'))
    combined_model.add(Dropout(0.2))
    combined_model.add(Dense(1))
    
    combined_model.compile(optimizer='adam', loss='mean_squared_error')

    combined_model.fit(x_train, y_train, batch_size=55, epochs=1)
    

    # Make predictions
    test_data = scaled_data[training_data_len - days: , :]
    x_test = []
    y_test = dataset[training_data_len:, :]

    for i in range(days, len(test_data)):
        x_test.append(test_data[i-days:i, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    lstm_predictions = lstm_model.predict(x_test)
    lstm_predictions = scaler.inverse_transform(lstm_predictions)

    rnn_predictions = rnn_model.predict(x_test)
    rnn_predictions = scaler.inverse_transform(rnn_predictions)
    
    combined_predictions = combined_model.predict(x_test)
    combined_predictions = scaler.inverse_transform(combined_predictions)

    # Format the predictions and actual values as data frames
    date_range = pd.date_range(end=end_date, periods=days)
    lstm_predicted_df = pd.DataFrame(lstm_predictions, columns=['LSTM Predicted Close'], index=date_range)
    rnn_predicted_df = pd.DataFrame(rnn_predictions, columns=['RNN Predicted Close'], index=date_range)
    combined_predicted_df = pd.DataFrame(combined_predictions, columns=['Hybrid Predicted Close'], index=date_range)
    actual_df = pd.DataFrame(y_test, columns=['Actual Close'], index=date_range)
    
    # Create a subplots figure to visualize the data
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True)

    # Add traces to the figure
    fig.add_trace(go.Scatter(x=actual_df.index, y=actual_df['Actual Close'], name='Actual Close'), row=1, col=1)
    fig.add_trace(go.Scatter(x=lstm_predicted_df.index, y=lstm_predicted_df['LSTM Predicted Close'], name='LSTM Predicted Close'), row=2, col=1)
    fig.add_trace(go.Scatter(x=rnn_predicted_df.index, y=rnn_predicted_df['RNN Predicted Close'], name='RNN Predicted Close'), row=2, col=1)
    fig.add_trace(go.Scatter(x=combined_predicted_df.index, y=combined_predicted_df['Hybrid Predicted Close'], name='Hybrid Predicted Close'), row=2, col=1)

    # Update the figure layout
    fig.update_layout(title=f'Stock Price Predictions for {symbol.upper()}',
                        xaxis_title='Date',
                        yaxis_title='Price (USD)')

    # Convert the figure to JSON and pass it to the template
    plot_json = fig.to_json()
    
    
    last_seven_days = date_range[-7:]  # Get the last seven days from the date range
    
    lstm_predict_df = pd.DataFrame(lstm_predictions[-7:], columns=['LSTM Predicted Close'], index=last_seven_days)
    rnn_predict_df = pd.DataFrame(rnn_predictions[-7:], columns=['RNN Predicted Close'], index=last_seven_days)
    combined_predict_df = pd.DataFrame(combined_predictions[-7:], columns=['Hybrid Predicted Close'], index=last_seven_days)
    actual_df = pd.DataFrame(y_test[-7:], columns=['Actual Close'], index=last_seven_days)
    
    results_df = lstm_predict_df.join([rnn_predict_df, combined_predict_df, actual_df])



    return render_template('predict2.html', plot_json=plot_json,results=results_df.to_html())

if __name__ == '__main__':
    app.run(debug=True)