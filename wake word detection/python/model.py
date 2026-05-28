"""DS-CNN architecture for the Hey-Jarvis wake-word detector.

The model is a Depthwise Separable Convolutional Network — a well-known
TinyML-friendly architecture that hits ~95 % accuracy on the Google
Speech Commands benchmark while compiling to under 25 kB of flash when
quantized to int8.

The default width / depth are tuned for the Nano 33 BLE Sense's 256 kB
RAM and 1 MB flash.
"""
from __future__ import annotations

from typing import Tuple

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

from config import FEATURE_SHAPE, NUM_CLASSES, TRAIN


def _conv_block(x: tf.Tensor, filters: int, kernel: Tuple[int, int],
                strides: Tuple[int, int], name: str) -> tf.Tensor:
    x = layers.Conv2D(
        filters, kernel, strides=strides, padding="same", use_bias=False,
        kernel_regularizer=regularizers.l2(TRAIN.weight_decay),
        name=f"{name}_conv",
    )(x)
    x = layers.BatchNormalization(name=f"{name}_bn")(x)
    x = layers.ReLU(name=f"{name}_relu")(x)
    return x


def _ds_block(x: tf.Tensor, filters: int, name: str) -> tf.Tensor:
    x = layers.DepthwiseConv2D(
        (3, 3), padding="same", use_bias=False,
        depthwise_regularizer=regularizers.l2(TRAIN.weight_decay),
        name=f"{name}_dw",
    )(x)
    x = layers.BatchNormalization(name=f"{name}_dw_bn")(x)
    x = layers.ReLU(name=f"{name}_dw_relu")(x)
    x = layers.Conv2D(
        filters, (1, 1), padding="same", use_bias=False,
        kernel_regularizer=regularizers.l2(TRAIN.weight_decay),
        name=f"{name}_pw",
    )(x)
    x = layers.BatchNormalization(name=f"{name}_pw_bn")(x)
    x = layers.ReLU(name=f"{name}_pw_relu")(x)
    return x


def build_model(input_shape: Tuple[int, int, int] = FEATURE_SHAPE,
                num_classes: int = NUM_CLASSES,
                width: int = 64,
                num_ds_blocks: int = 4) -> tf.keras.Model:
    """Construct a Depthwise Separable CNN tailored for MFCC inputs."""
    inputs = layers.Input(shape=input_shape, name="mfcc_input")

    x = _conv_block(inputs, filters=width, kernel=(10, 4),
                    strides=(2, 2), name="stem")

    for i in range(num_ds_blocks):
        x = _ds_block(x, filters=width, name=f"block{i+1}")

    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(0.2, name="dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax",
                           name="probabilities")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="hey_jarvis_dscnn")
    return model


def compile_model(model: tf.keras.Model,
                  learning_rate: float = TRAIN.learning_rate) -> tf.keras.Model:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.SparseCategoricalCrossentropy(name="loss_metric"),
        ],
    )
    return model


if __name__ == "__main__":
    m = compile_model(build_model())
    m.summary()
