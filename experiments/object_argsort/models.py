from tensorflow.keras import layers
import tensorflow as tf
from transformer_modules import AddPositionalEmbedding, Encoder
from abstracters import RelationalAbstracter

def create_transformer(seqs_length, object_dim, embedding_dim, num_layers, num_heads, dff, dropout_rate, output_dim):
    source_embedder = layers.TimeDistributed(layers.Dense(embedding_dim), name='source_embedder')
    pos_embedding_adder_input = AddPositionalEmbedding(name='add_pos_embedding_input')
    encoder = Encoder(num_layers=num_layers, num_heads=num_heads, dff=dff, dropout_rate=dropout_rate, name='encoder')
    final_layer = layers.Dense(output_dim, name='final_layer')

    inputs = layers.Input(shape=(seqs_length, object_dim))
    x = source_embedder(inputs)
    x = pos_embedding_adder_input(x)
    x = encoder(x)
    x = final_layer(x)

    model = tf.keras.Model(inputs=inputs, outputs=x, name='transformer')

    return model


def create_abstractor(seqs_length, object_dim, embedding_dim, num_layers, num_heads, dff, dropout_rate, output_dim):

    embedder = tf.keras.Sequential([layers.Dense(32, activation='relu'), layers.Dense(16)])
    source_embedder = layers.TimeDistributed(embedder, name='source_embedder')
    pos_embedding_adder_input = AddPositionalEmbedding(name='add_pos_embedding_input')
    encoder = RelationalAbstracter(num_layers=num_layers, num_heads=num_heads, dff=dff, dropout_rate=dropout_rate, name='abstractor')
    final_layer = layers.Dense(output_dim, name='final_layer')

    inputs = layers.Input(shape=(seqs_length, object_dim))
    x = source_embedder(inputs)
    x = pos_embedding_adder_input(x)
    x = encoder(x)
    x = final_layer(x)

    model = tf.keras.Model(inputs=inputs, outputs=x, name='abstractor')

    return model
