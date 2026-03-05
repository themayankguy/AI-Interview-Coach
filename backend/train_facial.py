import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam


# ======================
# 1. LOAD DATA
# ======================

def load_data(img_size=48, batch_size=64):

    train_ds = tf.keras.utils.image_dataset_from_directory(
        "FER-2013/train",
        image_size=(img_size, img_size),
        batch_size=batch_size,
        color_mode="grayscale"
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        "FER-2013/test",
        image_size=(img_size, img_size),
        batch_size=batch_size,
        color_mode="grayscale"
    )

    class_names = train_ds.class_names
    num_classes = len(class_names)

    # Normalize
    normalization_layer = layers.Rescaling(1./255)
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))

    train_ds = train_ds.cache().shuffle(1000).prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.cache().prefetch(tf.data.AUTOTUNE)

    return train_ds, test_ds, num_classes, class_names


# ======================
# 2. BUILD MODEL
# ======================

def build_model(num_classes):

    model = models.Sequential([

        layers.Conv2D(32, (3,3), activation='relu', input_shape=(48,48,1)),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2,2),

        layers.Conv2D(64, (3,3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2,2),

        layers.Conv2D(128, (3,3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2,2),

        layers.GlobalAveragePooling2D(),

        layers.Dense(64, activation='relu'),
        layers.Dropout(0.6),

        layers.Dense(num_classes, activation='softmax')
    ])

    optimizer = Adam(learning_rate=0.0005)

    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


# ======================
# 3. TRAIN
# ======================

def train():

    train_ds, test_ds, num_classes, class_names = load_data()

    print("Classes:", class_names)

    model = build_model(num_classes)

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    lr_reduce = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6
    )

    history = model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=50,
        callbacks=[early_stop, lr_reduce]
    )

    os.makedirs("../models", exist_ok=True)
    model.save("../models/emotion_model.h5")

    print("Model saved successfully.")


# ======================
# 4. RUN
# ======================

if __name__ == "__main__":
    train()