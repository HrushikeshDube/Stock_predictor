from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.callbacks import EarlyStopping
from typing import NoReturn, Tuple
from datetime import date,timedelta

# This python file will process the data and train it.
# No prediction will be done in here

WINDOW_SIZE = 10  # Sets a constant size for the rolling window used
scaler = MinMaxScaler(feature_range=(0, 1))  # set all data values between 0 and 1


def data_loader(filename: str) -> Tuple[pd.DataFrame, list]:
    """
    Function which reads in data from a csv file and uses pandas to turn it into a dataframe.

    Args:
        filename: String that represents the filename in csv format

    Returns:
        df: All stock information that was in the csv file that pandas read in
        dataset: List of tuples for every date and close value. Tuples are in the format (Date, Adj Close)

    """
    df = pd.read_csv(filename)  # converts pandas dataframe to csv
    df['Date'] = pd.to_datetime(df['Date'])  # Puts all dates into datetime format
    dataset = list(zip(df['Date'], df['Adj Close'].values))

    return df, dataset


def train_test_split(df: pd.DataFrame, dataset: list) -> np.ndarray:
    """
     Splits data into train and test so it can be used in predictions

     Args:
         df: Any pandas dataframe that has been returned from the yahoo financial database
         dataset: dataset value that was returned from the data_loader function

     Returns:
         train: Training data that the model will use to fit itself (used to help the program learn).
         test: Data that will be hidden from model during training phase,
         and will be used to predict how accurate the model is

     """
    trainingRange = int(len(dataset) - 20)
    print("The training range is", trainingRange)
    train = df['Adj Close'].iloc[:trainingRange].values
    test = df['Adj Close'].iloc[trainingRange:].values
    train = np.reshape(train, (-1, 1))
    test = np.reshape(test, (-1, 1))

    return train, test


def data_scaler(action: str, train: np.ndarray, test: np.ndarray) -> np.ndarray:
    """
    Function which scales the data between 0 and 1 or transforms it

     Args:
         action: Action of whether to inverse transform or scale data
         train: training data
         test: testing data


     Returns:
         train_scaled: If action == 'fit' {returns train_scaled array with all values scaled between 0 and one}
                      If action == 'inverse' {returns train_scaled array}

         test_scaled: If action == 'fit' {returns test_scaled array with all values scaled between 0 and one}
                      If action == 'inverse' {returns test_scaled array}

     """
    if action.lower() == 'fit':
        train_scaled = scaler.fit_transform(train)
        test_scaled = scaler.transform(test)

    elif action.lower() == 'inverse':
        train_scaled = scaler.inverse_transform(train)
        test_scaled = scaler.inverse_transform(test)

    return train_scaled, test_scaled


def to_sequences(data: np.ndarray, window_size: int) -> np.ndarray:
    """
    Converts data from 2D input data to timeseries 3D array.

    For more info: https://datascience.stackexchange.com/questions/30762/how-to-predict-the-future-values-of-time-horizon-with-keras

    Args:
        window_size: Size of window of data
        data: data being converted to sequence

    Returns:
        np.array(x): numpy array of x training data
        np.array(y): numpy array of y training daata

    """
    x = []
    y = []

    for i in range(len(data) - window_size - 1):
        window = data[i:(i + window_size)]
        x.append(window)
        y.append(data[i + window_size])
    return np.array(x), np.array(y)


def generate_sets(train_scaled: np.ndarray, test_scaled: np.ndarray) -> np.ndarray:
    """
    Function which takes in train_scaled and test_scaled and puts it into four sets of data.

    For more info: https://stackoverflow.com/questions/46495215/what-is-the-difference-between-x-train-and-x-test-in-keras

    Args:
        train_scaled: training data scaled
        test_scaled: testing data scaled

    Returns:
        X_train: training data set
        Y_train: Expected outcomes/labels of training data
         X_test: Test data set
        Y_test: Expected outcomes/label of test data

    """
    X_train, Y_train = to_sequences(train_scaled, WINDOW_SIZE)
    X_test, Y_test = to_sequences(test_scaled, WINDOW_SIZE)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    print("=============================================================")
    print(f"X_trains shape is {X_train.shape} ; Y_Trains shape is {Y_train.shape}")
    print(f"X_tests shape is {X_test.shape} ; Y_Tests shape is {Y_test.shape}")
    print("================================================================================")
    print(X_train.shape, Y_train.shape)

    return X_train, Y_train, X_test, Y_test


def build_model(X_train: np.ndarray, Y_train: np.ndarray) -> tf.keras.models.Sequential:
    """
    Function which builds the model for the predictions

    Args:
        X_train: training data set
        Y_train: test data set


    Returns:
        model: Weights and biases learned from the training data

    """

    model = Sequential()
    model.add(LSTM(64, input_shape=(X_train.shape[1], 1), return_sequences=True, activation='relu'))
    model.add(Dropout(.2))
    model.add(LSTM(32))
    model.add(Dropout(.2))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])  # adagrad?
    monitor = EarlyStopping(monitor='loss', patience=10, verbose=1, restore_best_weights=True)
    model.fit(X_train, Y_train, callbacks=[monitor], epochs=200)

    return model


def graph_format(dataset: list, train_predict: np.ndarray, test_predict: np.ndarray) -> np.ndarray:
    """
    Function which formats train_predict and test_predict to be formattted for graphing in matplot

    Args:
        dataset: Dates and Adj Close data stored into tuples
        train_predict: training predictions
        test_predict: testing predictions

    Returns:
        train_predict_plot: train_predict formatted for plotting
        test_predict_plot: test_predict formatted for plotting

    """
    train_predict, test_predict = data_scaler('inverse', train_predict, test_predict)
    train_predict_plot = np.empty_like(dataset)
    train_predict_plot[:, :] = np.nan
    train_predict_plot[WINDOW_SIZE:len(train_predict) + WINDOW_SIZE, :] = train_predict

    test_predict_plot = np.empty_like(dataset)
    test_predict_plot[:, :] = np.nan
    test_predict_plot[len(train_predict) + (WINDOW_SIZE * 2) + 1: len(dataset) - 1, :] = test_predict

    return train_predict_plot, test_predict_plot


def graph_data(df: pd.DataFrame, train_predict_plot: np.ndarray,
               test_predict_plot: np.ndarray, tkr: str) -> NoReturn:
    """
    Function which reads in data from a csv file and uses pandas to turn it into a dataframe.

    Args:
        df: Pandas dataframe which contains original stock information
        train_predict_plot: train predictions --in array formatted for plotting
        test_predict_plot: testing predictions  --in array formatted for plotting
        tkr: Stock ticker

    """

    plt.clf()  # Clears currently plotted figure
    plt.title(f"{tkr.upper()} Stock Information")
    print(f"Test predict plots shape is {test_predict_plot.shape}")
    plt.plot(df['Date'], test_predict_plot, "-y", label="Predcition")
    plt.plot(df['Date'], df['Adj Close'], "-g", label="Original")
    plt.xlim((date.today() - timedelta(days=60), date.today()))

    plt.xlabel('Date')
    plt.ylabel('Adj Close Price')
    plt.tight_layout()
    plt.gcf().autofmt_xdate()
    plt.legend(loc='upper left')
    plt.savefig('imgFile.png')
