from collections import Counter
import os
import json

import numpy as np
from sklearn.externals import joblib


ROOT = os.path.dirname(os.path.dirname(__file__))
BASE_DIR = os.path.join(ROOT, "oss_data")

EMBEDDING_MODEL = "models/128D_LFW.pkl"
TENSOR_NAME = "LFW"  # Label Faces in the Wild
CONFIG_PATH = os.path.join(BASE_DIR, "oss_demo_projector_config.json")
TENSOR_PATH = os.path.join(BASE_DIR, "tensor.bytes")
LABELS_PATH = os.path.join(BASE_DIR, "LFW_labels.tsv")
IMAGE_SPRITES_PATH = os.path.join(BASE_DIR, "sprites.png")
IMAGE_SIZE = 28


def main():

    labels, class_names, emb_arrays = joblib.load(EMBEDDING_MODEL)
    emb_arrays.astype(np.float32).tofile(TENSOR_PATH)

    with open(LABELS_PATH, 'w') as f:
        c = Counter(labels)
        for i in c:
            for _ in range(c[i]):
                f.write(class_names[i] + "\n")

    with open(CONFIG_PATH) as f:
        oss_json = json.load(f)

    json_to_append = {
        "tensorName": TENSOR_NAME,
        "tensorShape": [len(labels), emb_arrays.shape[1]],
        "tensorPath": TENSOR_PATH,
        "metadataPath": LABELS_PATH,
        "sprite": {
            "imagePath": IMAGE_SPRITES_PATH,
            "singleImageDim": [IMAGE_SIZE, IMAGE_SIZE]
        }
    }
    oss_json['embeddings'] = [json_to_append]
    with open(CONFIG_PATH, 'w+') as f:
        json.dump(oss_json, f, indent=4)


if __name__ == "__main__":
    main()
