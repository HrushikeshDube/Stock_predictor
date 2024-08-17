# IMPORTANT!! This file is for testing purposes only
from data_processor import data_loader
from data_processor import train_test_split
from data_processor import data_scaler
from data_processor import generate_sets
from data_processor import build_model


# TODO: Move this to different file
from data_processor import graph_format
from data_processor import graph_data

df, dataset = data_loader('datasets/TSLA.csv')

train, test = train_test_split(df, dataset)

train_scaled, test_scaled = data_scaler('fit',train, test)

X_train, Y_train, X_test, Y_test = generate_sets(train_scaled, test_scaled)

model = build_model(X_train, Y_train)

train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

train_predict_plot, test_predict_plot = graph_format(dataset, train_predict, test_predict)
print(test_predict_plot[-2][0])
graph_data(df, train_predict_plot, test_predict_plot, 'TSLA')