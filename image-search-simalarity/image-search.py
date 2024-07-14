import pandas as pd
import numpy as np
import torch
import cv2
import glob
import os
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn as nn
from transformers import ViTFeatureExtractor, ViTModel

import argparse
import chromadb


def main(opt):

    data = load_data()
    model, ft = load_model(opt)
    client = chromadb.PersistentClient(
        path="image-search-simalarity/test_storage")

    collection = client.get_or_create_collection(f"image-search-{opt.ft}")

    embedded_data = get_embedded_data(opt, data, model, ft)
    collection.add(embeddings=embedded_data, ids=[
                   f"id-{x}" for x in range(len(embedded_data))])


def load_data():
    data = {
        "brain": [],
        "butterfly": [],
    }

    for i in glob.glob("/Users/menghang/Desktop/ml-dev/ml-replicate/vector-db/image-search-simalarity/caltech-101/101_ObjectCategories/brain/*"):
        data["brain"].append(i)

    for i in glob.glob("/Users/menghang/Desktop/ml-dev/ml-replicate/vector-db/image-search-simalarity/caltech-101/101_ObjectCategories/butterfly/*"):
        data["butterfly"].append(i)

    return data


def load_model(opt):
    feature_extractor = None
    if opt.ft == "transformer":
        feature_extractor = ViTFeatureExtractor.from_pretrained(
            'google/vit-base-patch16-224-in21k')
        model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
    else:
        resnet50 = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        model = nn.Sequential(*list(model.children())[:-1])

    return model, feature_extractor


def query(opt):
    img = cv2.imread(
        "/Users/menghang/Desktop/ml-dev/ml-replicate/vector-db/image-search-simalarity/caltech-101/101_ObjectCategories/butterfly/image_0005.jpg")
    img = cv2.resize(img, (224, 224))

    model, ft = load_model(opt)
    if ft is None:
        img = img.tranpose((2, 0, 1))
        features = model(img)
        features = features.view(1, -1)[0].tolist()

    else:
        inputs = ft(img, return_tensors="pt")
        features = model(**inputs).last_hidden_state
        features = features[:, 0, :][0].tolist()

    client = chromadb.PersistentClient(
        path="image-search-simalarity/test_storage")

    collection = client.get_or_create_collection(f"image-search-{opt.ft}")
    print(collection.count())

    results = collection.query(
        query_embeddings=features, n_results=10, include=["distances"])
    print(results)


def get_embedded_data(opt, data, model, feature_extractor):
    embedded_data = []

    for brain_path in data["brain"]:
        img = cv2.imread(brain_path)
        img = cv2.resize(img, (224, 224))

        if opt.ft == "transformer":
            img = feature_extractor(images=img, return_tensors="pt")
            features = model(**img).last_hidden_state
            features = features[:, 0, :][0].tolist()
            embedded_data.append(features)
        else:
            transposed_img = img.transpose(2, 0, 1)
            features = model(torch.Tensor([transposed_img]))

            flattened_features = features.view(
                features.size(0), -1)[0].tolist()
            embedded_data.append(flattened_features)

    for bf_path in data["butterfly"]:
        img = cv2.imread(bf_path)
        img = cv2.resize(img, (224, 224))

        if opt.ft == "transformer":
            img = feature_extractor(images=img, return_tensors="pt")
            features = model(**img).last_hidden_state
            features = features[:, 0, :][0].tolist()
            embedded_data.append(features)
        else:
            transposed_img = img.transpose(2, 0, 1)
            features = model(torch.Tensor([transposed_img]))

            flattened_features = features.view(
                features.size(0), -1)[0].tolist()
            embedded_data.append(flattened_features)

    return embedded_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ft",  default="transformer")
    parser.add_argument("--mode",  default="query")

    opt = parser.parse_args()
    if opt.mode == "query":
        query(opt)
    else:
        main(opt=opt)
