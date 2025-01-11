import mlflow
from tensorflow.keras.callbacks import Callback



class MLflowLogger(Callback):
    def __init__(self):
        super().__init__()

    def on_epoch_end(self, epoch, logs=None):
        mlflow.log_metric("loss", logs["loss"], step=epoch)
        mlflow.log_metric("accuracy", logs["accuracy"], step=epoch)

        mlflow.log_metric("val_loss", logs["val_loss"], step=epoch)
        mlflow.log_metric("val_accuracy", logs["val_accuracy"], step=epoch)
