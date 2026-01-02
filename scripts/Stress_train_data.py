import tensorflow as tf
import pandas as pd
import os

IMG_SIZE = 224
BATCH = 16

BASE_PATH = "Stressdetector/Stress_Detection/train"
CSV_PATH = os.path.join(BASE_PATH, "_classes.csv")

df = pd.read_csv(CSV_PATH)
df.columns = df.columns.str.strip()
df = df[["filename", "Stress"]]

def load_image(filename, label):
    img_path = tf.strings.join([BASE_PATH, "/", filename])
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    return img, label

dataset = tf.data.Dataset.from_tensor_slices(
    (df["filename"].values, df["Stress"].values)
)

dataset = dataset.map(load_image).shuffle(500).batch(BATCH)

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, 3, activation="relu", input_shape=(IMG_SIZE,IMG_SIZE,3)),
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

model.fit(dataset, epochs=10)
model.save("Stressdetector/models/stress_model.keras")
