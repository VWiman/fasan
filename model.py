import tensorflow as tf

from config import CNN_CONV_FILTERS, CNN_DENSE_UNITS, CNN_DROPOUT_RATE, IMAGE_SIZE, LEARNING_RATE


# ============================================================
# 1. CNN-MODELL
# ============================================================
#
# Modellen är en enkel convolutional neural network för binär bildklassificering.
# Convolution-lager lär sig visuella mönster i bilden, pooling-lager minskar
# storleken stegvis och GlobalAveragePooling2D sammanfattar feature maps utan att
# skapa ett mycket stort dense-lager. Modellens lager styrs från config.py så att
# arkitekturen kan justeras utan att bygga om funktionen.

# ------------------------------------------------------------
# 1.1 Bygg modell
# ------------------------------------------------------------
def build_cnn_model(
    image_size=IMAGE_SIZE,
    learning_rate=LEARNING_RATE,
    conv_filters=CNN_CONV_FILTERS,
    dense_units=CNN_DENSE_UNITS,
    dropout_rate=CNN_DROPOUT_RATE,
):
    input_shape = (image_size[1], image_size[0], 3)

    model = tf.keras.models.Sequential()
    
    model.add(tf.keras.Input(shape=input_shape))

    for filters in conv_filters:
        model.add(tf.keras.layers.Conv2D(filters, (3, 3), activation="relu", padding="same"))
        model.add(tf.keras.layers.MaxPooling2D((2, 2)))

    # model.add(tf.keras.layers.GlobalAveragePooling2D())
    model.add(tf.keras.layers.Flatten())
    for units in dense_units:
        model.add(tf.keras.layers.Dense(units, activation="relu"))
        if dropout_rate > 0:
            model.add(tf.keras.layers.Dropout(dropout_rate))
        
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model
