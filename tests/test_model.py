import tensorflow as tf

from model import build_cnn_model
from training import save_trained_model


# ============================================================
# 1. CNN-MODELL
# ============================================================
#
# Testerna kontrollerar att modellen kan byggas med en mindre bildstorlek. Det
# gör att arkitekturen kan valideras snabbt utan att hela datasetet behöver tränas.

# ------------------------------------------------------------
# 1.1 Bygg modell
# ------------------------------------------------------------
def test_build_cnn_model_creates_binary_classifier():
    model = build_cnn_model(image_size=(32, 32), learning_rate=0.001)

    assert model.input_shape == (None, 32, 32, 3)
    assert model.output_shape == (None, 1)
    assert model.loss == "binary_crossentropy"


# ------------------------------------------------------------
# 1.2 Reducera feature maps före output
# ------------------------------------------------------------
def test_build_cnn_model_reduces_feature_maps_before_output():
    model = build_cnn_model(image_size=(32, 32), learning_rate=0.001)

    layer_types = [type(layer) for layer in model.layers]

    assert (
        tf.keras.layers.GlobalAveragePooling2D in layer_types
        or tf.keras.layers.Flatten in layer_types
    )


# ------------------------------------------------------------
# 1.3 Bygg modell med anpassade lager
# ------------------------------------------------------------
def test_build_cnn_model_accepts_custom_layer_settings():
    model = build_cnn_model(
        image_size=(32, 32),
        learning_rate=0.001,
        conv_filters=[8, 16],
        dense_units=[32, 16],
        dropout_rate=0.25,
    )

    conv_layers = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Conv2D)]
    dense_layers = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Dense)]
    dropout_layers = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Dropout)]

    assert [layer.filters for layer in conv_layers] == [8, 16]
    assert [layer.units for layer in dense_layers] == [32, 16, 1]
    assert dropout_layers[0].rate == 0.25
    assert model.output_shape == (None, 1)


# ------------------------------------------------------------
# 1.4 Spara modell
# ------------------------------------------------------------
def test_save_trained_model_writes_keras_file(tmp_path):
    model = build_cnn_model(image_size=(32, 32), learning_rate=0.001)
    model_path = save_trained_model(model, checkpoint_dir=tmp_path, model_file_name="test_model.keras")

    assert model_path.exists()
    assert model_path.name == "test_model.keras"
