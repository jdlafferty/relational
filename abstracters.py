"""
Module Implementing different variants of the 'abstracter'.

The abstracter is a module for transformer-based models which aims to encourage 
learning abstract relations.

It is characterized by employing learned input-independent 'symbols' in its computation
and using adjusted cross-attention mechanisms (e.g.: relational attention).

Typically, an abstracter module follows an 'encoder'.
For Seq2Seq models, it may be followed by a decoder.
"""

import tensorflow as tf
from transformer_modules import AddPositionalEmbedding, FeedForward
from attention import GlobalSelfAttention, BaseAttention, RelationalAttention, SymbolicAttention



class RelationalAbstracter(tf.keras.layers.Layer):
    """
    The 'Relational Abstracter' module.


    The 'input' is a sequence of input-independent learnable symbolic vectors.
    Implements cross-attention between these symbols and the entities at the encoder.
    Uses the scheme Q=E, K=E, V=A.
    """
    def __init__(self, num_layers, num_heads, dff, use_pos_embedding=True,
               dropout_rate=0.1, name='relational_abstracter'):
        super(RelationalAbstracter, self).__init__(name=name)

        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dff = dff
        self.use_pos_embedding = use_pos_embedding
        self.dropout_rate = dropout_rate

    def build(self, input_shape):

        _, self.sequence_length, self.d_model = input_shape

        # define the input-independent symbolic input vector sequence at the decoder
        normal_initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)
        self.symbol_sequence = tf.Variable(
            normal_initializer(shape=(self.sequence_length, self.d_model)),
            trainable=True)

        # layer which adds positional embedding (to be used on symbol sequence)
        if self.use_pos_embedding:
            self.add_pos_embedding = AddPositionalEmbedding()

        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)

        self.abstracter_layers = [
            RelationalAbstracterLayer(d_model=self.d_model, num_heads=self.num_heads,
                         dff=self.dff, dropout_rate=self.dropout_rate)
            for _ in range(self.num_layers)]

        self.last_attn_scores = None

    def call(self, encoder_context):
        # symbol sequence is input independent, so use the same one for all computations in the given batch
        symbol_seq = tf.zeros_like(encoder_context) + self.symbol_sequence

        # add positional embedding
        if self.use_pos_embedding:
            symbol_seq = self.add_pos_embedding(symbol_seq)

        symbol_seq = self.dropout(symbol_seq)

        for i in range(self.num_layers):
            symbol_seq = self.abstracter_layers[i](symbol_seq, encoder_context)

#             self.last_attn_scores = self.dec_layers[-1].last_attn_scores

        return symbol_seq

class RelationalAbstracterLayer(tf.keras.layers.Layer):
  def __init__(self,
               *,
               d_model,
               num_heads,
               dff,
               dropout_rate=0.1):
    super(RelationalAbstracterLayer, self).__init__()

    self.self_attention = GlobalSelfAttention(
        num_heads=num_heads,
        key_dim=d_model,
        dropout=dropout_rate)

    self.episodic_attention = RelationalAttention(
        num_heads=num_heads,
        key_dim=d_model,
        dropout=dropout_rate)

    self.ffn = FeedForward(d_model, dff)

  def call(self, x, context):
    x = self.self_attention(x=x)
    x = self.episodic_attention(x=x, context=context)

    # Cache the last attention scores for plotting later
    self.last_attn_scores = self.episodic_attention.last_attn_scores

    x = self.ffn(x)  # Shape `(batch_size, seq_len, d_model)`.
    return x


class SymbolicAbstracter(tf.keras.layers.Layer):
    """
    The 'Symbolic Abstracter' module.

    The 'input' is a sequence of input-independent learnable symbolic vectors.
    Implements cross-attention between these symbols and the entities at the encoder.
    Uses the scheme Q=A, K=E, V=A.
    """

    def __init__(self, num_layers, num_heads, dff, use_pos_embedding=True,
               dropout_rate=0.1, name='symbolic_abstracter'):
        super(SymbolicAbstracter, self).__init__(name=name)

        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dff = dff
        self.use_pos_embedding = use_pos_embedding
        self.dropout_rate = dropout_rate

    def build(self, input_shape):

        _, self.sequence_length, self.d_model = input_shape

        # define the input-independent symbolic input vector sequence at the decoder
        normal_initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)
        self.symbol_sequence = tf.Variable(
            normal_initializer(shape=(self.sequence_length, self.d_model)),
            trainable=True)

        # layer which adds positional embedding (to be used on symbol sequence)
        if self.use_pos_embedding:
            self.add_pos_embedding = AddPositionalEmbedding()

        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)

        self.abstracter_layers = [
            SymbolicAbstracterLayer(d_model=self.d_model, num_heads=self.num_heads,
                         dff=self.dff, dropout_rate=self.dropout_rate)
            for _ in range(self.num_layers)]

        self.last_attn_scores = None

    def call(self, encoder_context):
        # symbol sequence is input independent, so use the same one for all computations in the given batch
        symbol_seq = tf.zeros_like(encoder_context) + self.symbol_sequence

        # add positional embedding
        if self.use_pos_embedding:
            symbol_seq = self.add_pos_embedding(symbol_seq)


        symbol_seq = self.dropout(symbol_seq)


        for i in range(self.num_layers):
            symbol_seq = self.abstracter_layers[i](symbol_seq, encoder_context)

#             self.last_attn_scores = self.dec_layers[-1].last_attn_scores

        return symbol_seq

class SymbolicAbstracterLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1, name=None):
        super(SymbolicAbstracterLayer, self).__init__(name=name)

        self.self_attention = GlobalSelfAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate)

        self.symbolic_attention = SymbolicAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate)

        self.ffn = FeedForward(d_model, dff)

    def call(self, x, context):
        x = self.self_attention(x=x)
        x = self.symbolic_attention(x=x, context=context)

        # Cache the last attention scores for plotting later
        self.last_attn_scores = self.symbolic_attention.last_attn_scores

        x = self.ffn(x)  # Shape `(batch_size, seq_len, d_model)`.

        return x
