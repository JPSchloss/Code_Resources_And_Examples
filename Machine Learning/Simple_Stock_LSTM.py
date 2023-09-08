from datetime import datetime
import yfinance as yf
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, LeakyReLU
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping

class StockPredictor:
    # Initialize the StockPredictor object with a stock ticker and lookback period
    def __init__(self, ticker, lookback_period):
        self.ticker = ticker
        self.lookback_period = lookback_period
        self.data = None
        self.X_train = None
        self.X_test = None
        self.Y_train = None
        self.Y_test = None
        self.model = None

    # Fetch stock closing prices and preprocess the data
    def fetch_and_preprocess(self, start_date, end_date):
        self.data = yf.download(self.ticker, start=start_date, end=end_date)['Close'].values

    # Create datasets for training and testing
    def create_dataset(self):
        X, Y = [], []
        for i in range(len(self.data) - self.lookback_period - 1):
            a = self.data[i:(i + self.lookback_period)]
            X.append(a)
            Y.append(self.data[i + self.lookback_period])
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
            np.array(X).reshape(-1, self.lookback_period, 1), np.array(Y), test_size=0.2, random_state=42
        )

    # Create an LSTM model using Tanh as the activation function
    def create_lstm_model(self):
        self.model = Sequential()
        self.model.add(LSTM(128, return_sequences=True, activation='tanh', input_shape=(self.lookback_period, 1)))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(64, return_sequences=True, activation='tanh'))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(32, return_sequences=False, activation='tanh'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(16, activation='relu'))
        self.model.add(Dense(1))
        self.model.summary()

    # Create an LSTM model using ReLU as the activation function
    def create_lstm_model_relu(self):
        self.model = Sequential()
        self.model.add(LSTM(128, return_sequences=True, activation='relu', input_shape=(self.lookback_period, 1)))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(64, return_sequences=True, activation='relu'))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(32, return_sequences=False, activation='relu'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(16, activation='relu'))
        self.model.add(Dense(1))
        self.model.summary()

    # Create an LSTM model using Leaky ReLU as the activation function
    def create_lstm_model_leaky_relu(self):
        self.model = Sequential()
        self.model.add(LSTM(128, return_sequences=True, input_shape=(self.lookback_period, 1)))
        self.model.add(LeakyReLU())
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(64, return_sequences=True))
        self.model.add(LeakyReLU())
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(32, return_sequences=False))
        self.model.add(LeakyReLU())
        self.model.add(Dropout(0.2))
        self.model.add(Dense(16))
        self.model.add(LeakyReLU())
        self.model.add(Dense(1))
        self.model.summary()

    # Train the LSTM model
    def train_model(self):
        early_stopping = EarlyStopping(monitor='val_loss', patience=30)
        self.model.compile(optimizer='adam', loss='mean_squared_error')
        self.model.fit(self.X_train, self.Y_train, epochs=400, batch_size=64,
                       validation_data=(self.X_test, self.Y_test), callbacks=[early_stopping])

    # Predict the next day's stock closing price
    def predict_next_day_price(self):
        last_days = self.data[-self.lookback_period:]
        predicted_price = self.model.predict(last_days.reshape(1, self.lookback_period, 1))[0][0]
        predicted_price = round(predicted_price, 2)
        print(f"The predicted next day closing price for {self.ticker} is: {predicted_price}")

if __name__ == "__main__":
    lookback_period = 60  # Define the lookback period
    predictor = StockPredictor('GOOG', lookback_period)
    
    today_date = datetime.now().strftime('%Y-%m-%d')  # Get today's date in 'YYYY-MM-DD' format
    predictor.fetch_and_preprocess('2020-01-01', '2023-08-20')  # Fetch and preprocess data
    predictor.create_dataset()  # Create training and testing datasets
    predictor.create_lstm_model()  # Create LSTM model with Tanh activation
    #predictor.create_lstm_model_relu()  # Uncomment to use ReLU activation
    #predictor.create_lstm_model_leaky_relu()  # Uncomment to use Leaky ReLU activation
    predictor.train_model()  # Train the model
    predictor.predict_next_day_price()  # Make next day price prediction
