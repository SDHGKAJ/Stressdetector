import tensorflow as tf
import pandas as pd
import os

IMG_SIZE = 224
BATCH = 16
EPOCHS = 5

BASE_PATH = "Stressdetector/Stress_Detection/train"
CSV_PATH = os.path.join(BASE_PATH, "_classes.csv")

df = pd.read_csv(CSV_PATH)
df.columns = df.columns.str.strip()
df["label"] = 1 - df["Non"]
df = df[["filename", "label"]]
df["label"] = df["label"].astype("float32")

# train / validation split
train_df = df.sample(frac=0.8, random_state=42)
val_df = df.drop(train_df.index)

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

val_ds = tf.data.Dataset.from_tensor_slices(
    (val_df["filename"].values, val_df["label"].values)
)

train_ds = (
    train_ds
    .map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    .shuffle(500)
    .batch(BATCH)
    .prefetch(tf.data.AUTOTUNE)
)

val_ds = (
    val_ds
    .map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH)
    .prefetch(tf.data.AUTOTUNE)
)

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


model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

os.makedirs("Stressdetector/models", exist_ok=True)
model.save("Stressdetector/models/stress_model.keras")
