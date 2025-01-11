import mlflow
import argparse
import re

import metric_logger

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

    print("[log] Initialize model ...")

    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(labels.shape[1], activation='sigmoid')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

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
