# patch TST
# normalization
# quantization.- split into buckets
# chronos - auto-regressive (technically encoder-decoder)
# transformers can view all the timesteps at once, no vanishing gradients

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Layer, Dense, LayerNormalization, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras import Input
from LSTM import create_dataset, sequential_dataset
from config import MODELS_DIR

@keras.utils.register_keras_serializable()
class PositionalEncoding(Layer):
    def __init__(self, time_step, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.time_step = time_step
        self.embed_dim = embed_dim
        positions = np.arange(time_step)[:, np.newaxis]
        dimensions = np.arange(embed_dim)[np.newaxis, :]
        angle_rates = 1 / np.power(10000, (2 * (dimensions // 2)) / np.float32(embed_dim))
        angle_rads = positions * angle_rates
        positional_encoding = np.zeros((time_step, embed_dim))
        positional_encoding[:, 0::2] = np.sin(angle_rads[:, 0::2])
        positional_encoding[:, 1::2] = np.cos(angle_rads[:, 1::2])
        self.positional_encoding = tf.cast(positional_encoding, dtype = tf.float32)

    def get_config(self):
        config = super().get_config()
        config.update({"time_step": self.time_step, "embed_dim": self.embed_dim})
        return config

    def call(self, inputs):
        return inputs + self.positional_encoding

@keras.utils.register_keras_serializable()
class MultiHeadSelfAttention(Layer):
    def __init__(self, embed_dim, num_heads = 8, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.projection_dim = embed_dim//num_heads
        self.query_dense = Dense(embed_dim)
        self.key_dense = Dense(embed_dim)
        self.value_dense = Dense(embed_dim)
        self.combine_heads = Dense(embed_dim)

    def get_config(self):
        config = super().get_config()
        config.update({"embed_dim": self.embed_dim, "num_heads": self.num_heads})
        return config

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b = True)
        scaled_score = score/tf.math.sqrt(tf.cast(self.projection_dim, tf.float32))
        weights = tf.nn.softmax(scaled_score, axis = -1)
        output = tf.matmul(weights, value)
        return output, weights

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm = [0,2,1,3])

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)
        attention_output, _ = self.attention(query, key, value)
        attention_output = tf.transpose(attention_output, perm = [0,2,1,3])
        concat_attention = tf.reshape(attention_output, (batch_size, -1, self.embed_dim))
        return self.combine_heads(concat_attention)

@keras.utils.register_keras_serializable()
class TransformerBlock(Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation = "relu"),
            Dense(embed_dim),
        ])
        self.layernorm1 = LayerNormalization(epsilon = 1e-6)
        self.layernorm2 = LayerNormalization(epsilon = 1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def get_config(self):
        config = super().get_config()
        config.update({"embed_dim": self.embed_dim, "num_heads": self.num_heads, "ff_dim": self.ff_dim, "rate": self.rate})
        return config

    def call(self, inputs, training = False):
        attn_output = self.att(inputs)
        attn_output = self.dropout1(attn_output, training = training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training = training)
        return self.layernorm2(out1 + ffn_output)

@keras.utils.register_keras_serializable()
class TransformerEncoder(Layer):
    def __init__(self, num_layers, embed_dim, num_heads, ff_dim, rate = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate
        self.enc_layers = [TransformerBlock(embed_dim, num_heads, ff_dim, rate) for _ in range(num_layers)]
        self.dropout = Dropout(rate)

    def get_config(self):
        config = super().get_config()
        config.update({"num_layers": self.num_layers, "embed_dim": self.embed_dim, "num_heads": self.num_heads, "ff_dim": self.ff_dim, "rate": self.rate})
        return config

    def call(self, inputs, training = False):
        x = inputs
        x = self.dropout(x, training = training)
        for layer in self.enc_layers:
            x = layer(x, training = training)
        return x

def construct_model(time_step, embed_dim = 32, num_heads = 8, ff_dim = 512, num_layers = 4, dropout_rate = 0.1):
    inputs = Input(shape =(time_step, 7))
    x = Dense(embed_dim)(inputs)
    x = PositionalEncoding(time_step, embed_dim)(x)
    encoder = TransformerEncoder(num_layers, embed_dim, num_heads, ff_dim, rate = dropout_rate)
    x = encoder(x)
    x = tf.keras.layers.Flatten()(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(48)(x)
    return Model(inputs, outputs)

def main():

    Xs_train, ys_train, Xs_val, ys_val = sequential_dataset()
    model = construct_model(48*7, embed_dim =64, num_heads = 2, ff_dim = 64, num_layers = 2, dropout_rate = 0.25)

    model.compile(optimizer = "adam", loss = "mae", metrics = ["mae"])
    early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
    MODELS_DIR.mkdir(parents = True, exist_ok = True)
    checkpoint = keras.callbacks.ModelCheckpoint(MODELS_DIR / "best_transformer.keras", monitor="val_loss", save_best_only=True)

    history = model.fit(Xs_train, ys_train,validation_data=(Xs_val, ys_val),epochs=20,batch_size=32,callbacks=[early_stopping, checkpoint],shuffle=True,verbose=1)

if __name__ == "__main__":
    main()
