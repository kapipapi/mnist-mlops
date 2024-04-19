import argparse
import dataset
import model_mnist
import mlflow

mlflow.tensorflow.autolog()

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
args = parser.parse_args()

ds_train, ds_test = dataset.prepare_dataset()

input_shape = (28, 28, 1)
num_classes = 10

model = model_mnist.prepare_model(input_shape, num_classes)

with mlflow.start_run():
    model.fit(x=ds_train, epochs=args.epochs)
