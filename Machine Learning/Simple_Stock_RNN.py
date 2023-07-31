import yfinance as yf
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from sklearn.model_selection import train_test_split

def fetch_and_preprocess(symbol, start_date, end_date):
    # Fetch stock data and preprocess
    data = yf.download(symbol, start=start_date, end=end_date)
    data['Close'] = data['Close'].values
    return data

def create_dataset(df, look_back=1):
    # Creating the dataset with the look-back time period.
    X, Y = [], []
    for i in range(len(df)-look_back-1):
        a = df[i:(i+look_back)]
        X.append(a)
        Y.append(df[i + look_back])
    return np.array(X).reshape(-1, look_back, 1), np.array(Y)

def create_rnn_model(input_shape):
    # Setting up the model and adding a few layers. 
    model = Sequential()
    model.add(SimpleRNN(64, return_sequences=True, activation='tanh', input_shape=input_shape))
    model.add(SimpleRNN(64, return_sequences=False, activation='tanh'))
    model.add(Dense(32))
    model.add(Dense(1))
    return model

def predict_next_day_close_price(model, last_30_days):
    # Reshaping the data to be used for the model prediction and getting the actual model prediction. 
    X_pred = np.reshape(last_30_days, (1, 30, 1))
    predicted_price = model.predict(X_pred)
    return predicted_price[0][0]

def main(ticker):
    # Fetch and preprocess data
    data = fetch_and_preprocess(ticker, '2020-01-01', '2023-07-30')

    # Prepare and split data
    X, Y = create_dataset(data['Close'].values, look_back=30)

    # Reshape input to be [samples, time steps, features]
    X_train, _, Y_train, _ = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Create the RNN model and compile it
    model = create_rnn_model((30, 1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Fit the model
    model.fit(X_train, Y_train, epochs=100, batch_size=64)

    # Predict the next day close price
    last_30_days = data['Close'][-30:].values
    predicted_price = predict_next_day_close_price(model, last_30_days)
    predicted_price = round(predicted_price, 2)

    # Printing the output. 
    print(f"The predicted next day closing price for {ticker} is: {predicted_price:.2f}")

if __name__ == "__main__":
    main('AAPL')
