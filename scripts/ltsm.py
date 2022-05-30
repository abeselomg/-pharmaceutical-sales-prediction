import mlflow.tensorflow
import matplotlib as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from sklearn.preprocessing import Normalizer, MinMaxScaler, StandardScaler, RobustScaler


def scaler(df, columns, mode="minmax"):
    if mode == "minmax":
        minmax_scaler = MinMaxScaler()
        return (
            pd.DataFrame(minmax_scaler.fit_transform(df), columns=columns),
            minmax_scaler,
        )
    elif mode == "standard":
        scaler = StandardScaler()
        return pd.DataFrame(scaler.fit_transform(df), columns=columns), scaler
    elif mode == "robust":
        scaler = RobustScaler()
        return pd.DataFrame(scaler.fit_transform(df), columns=columns), scaler


def add_scaled_sales(df):
    scaled_sales, scaler_obj = scaler(
        df[["Sales"]], mode="minmax", columns=["scaled_sales"]
    )
    df["scaled_sales"] = scaled_sales["scaled_sales"].to_list()
    return df, scaler_obj

class TimeSeriesSalsesPred:
    def __init__(self, WINDOW_SIZE, BATCH_SIZE, sales_data):
        self.WINDOW_SIZE = WINDOW_SIZE
        self.BATCH_SIZE = BATCH_SIZE

        data_agg = sales_data.groupby("Date").agg({"Sales": "mean"})
        self.SIZE = len(data_agg["Sales"])

        self.scaled_df, self.scaler_obj = add_scaled_sales(data_agg)

        self.DateTrain = np.reshape(self.scaled_df.index.values[0:BATCH_SIZE], (-1, 1))
        self.DateValid = np.reshape(self.scaled_df.index.values[BATCH_SIZE:], (-1, 1))

        (
            self.train_sales,
            self.valid_sales,
            self.TrainDataset,
            self.ValidDataset,
        ) = self.prepare_data(WINDOW_SIZE, BATCH_SIZE, self.scaled_df)

    def prepare_data(self, WINDOW_SIZE, BATCH_SIZE, scaled_df):
        train_sales = scaled_df["scaled_sales"].values[0:BATCH_SIZE].astype("float32")
        valid_sales = scaled_df["scaled_sales"].values[BATCH_SIZE:].astype("float32")
        TrainDataset = self.windowed_dataset(train_sales, WINDOW_SIZE, BATCH_SIZE)
        ValidDataset = self.windowed_dataset(valid_sales, WINDOW_SIZE, BATCH_SIZE)

        return train_sales, valid_sales, TrainDataset, ValidDataset

    def train(
        self,
        EPOCHS,
        verbose=1,
    ):

        mlflow.set_experiment("Rossman-" + "Lstm_model")

        mlflow.tensorflow.autolog(every_n_iter=2, log_models=True)

        mlflow.end_run()
        with mlflow.start_run(run_name="Lstm_model-Base-line"):

            model = Sequential()
            model.add(LSTM(20, input_shape=[None, 1], return_sequences=True))
            model.add(LSTM(10, input_shape=[None, 1]))
            model.add(Dense(1))
            model.compile(loss="huber_loss", optimizer="adam")
            model.summary()

            history = model.fit(
                self.TrainDataset,
                epochs=EPOCHS,
                validation_data=self.ValidDataset,
                verbose=verbose,
            )

        self.plot_history(history)

        return model, history

    def plot_history(self, history):
        fig = plt.figure(figsize=(12, 9))
        plt.plot(history.history["loss"], label="loss")
        plt.plot(history.history["val_loss"], label="val_loss")
        plt.legend()
        plt.show()

        return fig

    def model_forecast_test(self, model):

        series = self.scaled_df["scaled_sales"].values[:, np.newaxis]

        ds = tf.data.Dataset.from_tensor_slices(series)
        ds = ds.window(self.WINDOW_SIZE, shift=1, drop_remainder=True)
        ds = ds.flat_map(lambda w: w.batch(self.WINDOW_SIZE))
        ds = ds.batch(self.SIZE).prefetch(1)
        forecast = model.predict(ds)

        Results = forecast[self.BATCH_SIZE - self.WINDOW_SIZE : -1]
        Results1 = self.scaler_obj.inverse_transform(Results.reshape(-1, 1))
        XValid1 = self.scaler_obj.inverse_transform(self.valid_sales.reshape(-1, 1))

        fig, MAE, RMSE = self.plot_forcast(
            Results, Results1, XValid1, self.DateValid, self.WINDOW_SIZE
        )

        return forecast, fig, MAE, RMSE

    def plot_forcast(self, Results, Results1, XValid1, DateValid, WINDOW_SIZE):
        fig = plt.figure(figsize=(30, 8))
        plt.title("LSTM Model Forecast Compared to Validation Data")

        plt.plot(DateValid.astype("datetime64"), Results1, label="Forecast series")
        plt.plot(
            DateValid.astype("datetime64"),
            np.reshape(XValid1, (2 * WINDOW_SIZE, 1)),
            label="Validation series",
        )

        plt.xlabel("Date")
        plt.ylabel("Thousands of Units")
        plt.xticks(DateValid.astype("datetime64")[:, -1], rotation=90)
        plt.legend(loc="upper right")

        MAE = tf.keras.metrics.mean_absolute_error(
            XValid1[:, -1], Results[:, -1]
        ).numpy()
        RMSE = np.sqrt(
            tf.keras.metrics.mean_squared_error(XValid1[:, -1], Results[:, -1]).numpy()
        )

        textstr = "MAE = " + "{:.3f}".format(MAE) + "  RMSE = " + "{:.3f}".format(RMSE)

        # place a text box in upper left in axes coords
        plt.annotate(textstr, xy=(0.87, 0.05), xycoords="axes fraction")
        plt.grid(True)

        plt.show()

        return fig, MAE, RMSE

    def windowed_dataset(self, series, window_size, batch_size):
        series = tf.expand_dims(series, axis=-1)
        dataset = tf.data.Dataset.from_tensor_slices(series)
        dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
        dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
        dataset = dataset.map(lambda window: (window[:-1], window[-1:]))
        dataset = dataset.batch(batch_size).prefetch(1)
        return dataset
