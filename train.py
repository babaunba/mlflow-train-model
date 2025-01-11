import mlflow
import argparse
import re

import metric_logger
import optuna

import json

import pandas as pd 
from markdown import markdown
from bs4 import BeautifulSoup

from sklearn.preprocessing import MultiLabelBinarizer
from transformers import BertTokenizer, BertModel
import torch


import tensorflow as tf

from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore")


THRESHOLD = 0.4
SEED = 123


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-run-id")
    return parser.parse_args()


def clean_markdown(text):
    html = markdown(text)
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup.find_all(["img", "a"]):
        tag.decompose()

    clean_text = soup.get_text()
    clean_text = re.sub(r'[*_~`]', '', clean_text)
    clean_text = ' '.join(clean_text.split())
    
    return clean_text


def get_bert_embeddings(texts, max_len=128):
    input_ids = []
    attention_masks = []

    for text in texts:
        encoded = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_len,
            truncation=True,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt'
        )
        input_ids.append(encoded['input_ids'])
        attention_masks.append(encoded['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    with torch.no_grad():
        outputs = bert_model(input_ids, attention_mask=attention_masks)
    return outputs[1].numpy()


def build_dataframe(data):
    ds = pd.DataFrame(data["issues"])
    ds["title"] = ds["title"].str.lower()
    ds["body"] = ds["body"].str.lower()
    ds["body"].fillna("", inplace=True)
    ds["clean_body"] = ds["body"].apply(clean_markdown)

    return ds


def get_embedding_vectors(ds):
    texts = list(ds["title"] + " " + ds["clean_body"])
    return get_bert_embeddings(texts)


def preprocess_labels(ds, project_labels):
    mlb = MultiLabelBinarizer(classes=project_labels)
    return mlb.fit_transform(ds["labels"]), list(mlb.classes_)


def init_model(params):
    # {'n_units': 80, 'activation': 'leaky_relu', 'dropout_rate': 0.30000000000000004, 'learning_rate': 0.0007190600962946125, 'alpha': 0.17}

    if params['activation'] == 'relu':
        dense_layer = tf.keras.layers.Dense(
            params['n_units'],
            activation=params['activation'],
        )
    else:
        dense_layer = tf.keras.layers.Dense(params['n_units'])

    if params['activation'] == 'leaky_relu':
        activation_layer = tf.keras.layers.LeakyReLU(alpha=params['alpha'])
    else:
        activation_layer = None

    layers = [
        tf.keras.layers.InputLayer(input_shape=(X_train.shape[1],)),
        dense_layer,
        activation_layer,
        tf.keras.layers.Dropout(params['dropout_rate']),
        tf.keras.layers.Dense(y_train.shape[1], activation='sigmoid')
    ]

    model = tf.keras.Sequential([ layer for layer in layers if layer is not None ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=params['learning_rate']),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


def create_model(trial):
    n_units = trial.suggest_int('n_units', 128, 256, step=64)
    activation = trial.suggest_categorical('activation', ['relu', 'leaky_relu'])
    dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5, step=0.1)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)
    alpha = trial.suggest_float('alpha', 0.01, 0.3, step=0.01)

    return init_model({
        'n_units': n_units,
        'activation': activation,
        'dropout_rate': dropout_rate,
        'learning_rate': learning_rate,
        'alpha': alpha,
    })


def objective(trial):
    model = create_model(trial)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=10,
        batch_size=16,
        verbose=0,
    )
    return max(history.history['val_accuracy'])


with mlflow.start_run():
    args = parse_args()
    tokenizer = BertTokenizer.from_pretrained('prajjwal1/bert-small')
    bert_model = BertModel.from_pretrained('prajjwal1/bert-small')

    print("[log] Download dataset ...")

    local_path = mlflow.artifacts.download_artifacts(
        run_id=args.dataset_run_id,
        artifact_path="dataset.json",
    )
    
    with open(local_path, "r") as file:
        data = json.load(file)

    print("[log] Prepair dataset for train ...")

    ds = build_dataframe(data)
    embedding_vectors = get_embedding_vectors(ds)
    labels, label_classes = preprocess_labels(ds, data["project_labels"])

    X_train, X_test, y_train, y_test = train_test_split(embedding_vectors, labels, test_size=0.2, random_state=SEED)

    print("[log] Find best hyper params ...")

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30)

    print("[log] Best params:")
    for key, value in study.best_params.items():
        print(f"- {key} = {value}")
        mlflow.log_param(key, value)

    print("[log] Initialize model ...")

    model = init_model(study.best_params)
    model.summary()

    print(model.summary())

    print("[log] Train model ...")

    history = model.fit(
        X_train,
        y_train,
        epochs=40,
        batch_size=8,
        validation_data=(X_test, y_test),
        callbacks=[metric_logger.MLflowLogger()],
    )

    print("[log] Check model work ...")

    probabilities = model.predict(X_test[:1])
    predicted_classes_flags = (probabilities > THRESHOLD).astype(int)[0]
    predicted_classes = [ label_classes[class_index] for class_index, is_predicted in enumerate(predicted_classes_flags) if is_predicted ]
    print(f"Predicted classes: {predicted_classes}")

    print("[log] Publish new model version ...")
    model_config = {
        "project_labels": label_classes,
        "threshold": THRESHOLD,
    }
    config_file = "config.json"

    mlflow.keras.log_model(
        model,
        "model",
        registered_model_name="label-suggestor",
    )

    with open(config_file, "w", encoding="utf-8") as file:
        json.dump(model_config, file)

    mlflow.log_artifact(config_file)
