import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

IMG_SIZE = 224
BATCH = 16
EPOCHS = 10

BASE_PATH = "Stressdetector/Stress&CogLoad_Dataset/train"
CSV_PATH = os.path.join(BASE_PATH, "_classes.csv")

#---------LOAD + FIX LABELS--------------
df = pd.read_csv(CSV_PATH)
df.columns = df.columns.str.strip()

# Non = 1 → not stressed → label 0
# Non = 0 → stressed → label 1
df["label"] = 1 - df["Non"]
df = df[["filename", "label"]]
df["label"] = df["label"].astype("float32")

#-------------TRAIN / TEST SPLIT-------------

train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

#------------TF.DATA PIPELINE-------------

def load_image(filename, label):
    img_path = tf.strings.join([BASE_PATH, "/", filename])
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    return img, label

train_ds = tf.data.Dataset.from_tensor_slices(
    (train_df["filename"].values, train_df["label"].values)
)
test_ds = tf.data.Dataset.from_tensor_slices(
    (test_df["filename"].values, test_df["label"].values)
)

train_ds = (
    train_ds
    .map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    .shuffle(500)
    .batch(BATCH)
    .prefetch(tf.data.AUTOTUNE)
)

test_ds = (
    test_ds
    .map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH)
    .prefetch(tf.data.AUTOTUNE)
)

#------------------MODEL----------------
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1)
])

model = tf.keras.Sequential([
    data_augmentation,
    tf.keras.layers.Conv2D(32, 3, activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.fit(train_ds, epochs=EPOCHS)

y_prob = model.predict(test_ds).ravel()
y_pred = (y_prob > 0.5).astype(int)
y_true = test_df["label"].astype(int).values

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print("\nEvaluation Metrics:")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-Score : {f1:.4f}")

os.makedirs("Stressdetector/models", exist_ok=True)
model.save("Stressdetector/models/stress_model.keras")
