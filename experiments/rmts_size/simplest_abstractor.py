import tensorflow as tf
import numpy as np
from abstracters import RelationalAbstracter, RelationalAbstracterLayer, SimpleAbstractor
from seq2seq_abstracter_models import Encoder, AddPositionalEmbedding
from tensorflow.keras import layers, Model, Sequential
from temporal_dense import TemporalDense


class SimplestAbstractorCNN(tf.keras.Model):
    def __init__(self, num_classes, sequence_len, embedding_dim, symbol_dim, name='simplest_abstractor_cnn'):
        super().__init__(name=name)
        self.sequence_len = sequence_len
        self.embedding_dim = embedding_dim
        self.symbol_dim = symbol_dim
        self.cnn_encoder = CnnEncoder(ff_dim2=embedding_dim, name='cnn_encoder')
        self.cnn_embedder = layers.TimeDistributed(self.cnn_encoder, name='cnn_embedder')
        self.cnn_embedder.trainable = False
        normal_initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)
        self.symbols = tf.Variable(
            normal_initializer(shape=(self.sequence_len, self.symbol_dim)),
            name='symbols', trainable=False)
        self.scale = 50
        self.flatten = layers.Flatten()
        self.query_projection = layers.Dense(embedding_dim, activation=None, name='query_projection')
        self.key_projection = layers.Dense(embedding_dim, activation=None, name='key_projection')
        self.hidden_layer = layers.Dense(32, activation='relu', name='hidden_layer')
        self.final_layer = layers.Dense(1, activation='sigmoid', name='final_layer')

    def call(self, inputs):
        source = inputs
        E = self.cnn_embedder(source)
        Q = self.query_projection(E)
        K = self.key_projection(E)
        self.Z = self.scale * tf.einsum('ijk,ilk->ijl', Q, K) / self.embedding_dim
        self.R = tf.nn.softmax(self.Z, axis=1)
        self.A = tf.einsum('ijk,kl->ijl', self.R, self.symbols)
        flattened_symbols = self.flatten(self.A)
        hidden_units = self.hidden_layer(flattened_symbols)
        output = self.final_layer(hidden_units)
        return output
        
class CnnEncoder(tf.keras.Model):
    def __init__(self, ff_dim1=64, ff_dim2=64, name='cnn_encoder'):
        super().__init__(name=name)

        self.conv_layer1 = layers.Conv2D(32, (2, 2), activation='relu', name='%s/conv_layer1' % name)
        self.pool_layer1 = layers.MaxPooling2D((2, 2))
        self.conv_layer2 = layers.Conv2D(32, (2, 2), activation='relu', name='%s/conv_layer2' % name)
        self.pool_layer2 = layers.MaxPooling2D((2, 2))
        self.flatten = layers.Flatten()
        #self.dense1 = layers.Dense(ff_dim1, activation='relu')
        self.dense2 = layers.Dense(ff_dim2, activation='relu', name='%s/dense_layer' % name)
        self.normalize = layers.LayerNormalization(center=False, scale=False, epsilon=1e-6)

    def call(self, inputs):
        source = inputs
        x = self.conv_layer1(source)
        x = self.pool_layer1(x)
        x = self.conv_layer2(x)
        x = self.pool_layer2(x)
        x = self.flatten(x)
        #x = self.dense1(x)
        x = self.dense2(x)
        outputs = self.normalize(x)
        return outputs
