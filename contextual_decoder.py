# implement a general cross-attention layer which supports all the different kinds of attention
import tensorflow as tf

from seq2seq_transformer import BaseAttention
from seq2seq_transformer import CausalSelfAttention, FeedForward

class ContextualCrossAttention(BaseAttention):
    """
    A more general cross-attention layer with queries, keys, and values specified in call.

    """
    def __init__(self, **kwargs):
        """
        create ContextualCrossAttention layer
        """

        super(ContextualCrossAttention, self).__init__(**kwargs)

    def call(self, input_seq, query_seq, key_seq, value_seq):

        attn_output, attn_scores = self.mha(
            query=query_seq,
            key=key_seq,
            value=value_seq,
            return_attention_scores=True)

        x = self.add([input_seq, attn_output])

        x = self.layernorm(x)

        # Cache the attention scores for plotting later.
        self.last_attn_scores = attn_scores


        return x

class ContextDecoderLayer(tf.keras.layers.Layer):
    """
    A generalized deocder layer with configurable cross-attention schemes.
    """

    def __init__(
        self,
        d_model,
        num_heads,
        dff,
        dropout_rate=0.1,
    ):

        super(ContextDecoderLayer, self).__init__()

        self.causal_self_attention = CausalSelfAttention(
            num_heads=num_heads, key_dim=d_model, dropout=dropout_rate
        )

        self.cross_attention = ContextualCrossAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate,
        )

        self.ffn = FeedForward(d_model, dff)

    def call(self, input_seq, query_seq, key_seq, value_seq):
        x = self.causal_self_attention(x=input_seq)
        x = self.cross_attention(input_seq=x, query_seq=query_seq, key_seq=key_seq, value_seq=value_seq)

        # Cache the last attention scores for plotting later
        self.last_attn_scores = self.cross_attention.last_attn_scores

        x = self.ffn(x)  # Shape `(batch_size, seq_len, d_model)`.

        return x

class ContextDecoder(tf.keras.layers.Layer):
    """A generalized decoder with configurable cross-attention schemes"""

    def __init__(
        self,
        num_layers,
        num_heads,
        dff,
        dropout_rate=0.1,
        name="decoder",
    ):

        super(ContextDecoder, self).__init__(name=name)

        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dff = dff
        self.dropout_rate = dropout_rate

    def build(self, input_shape):

        _, self.sequence_length, self.d_model = input_shape

        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)

        self.dec_layers = [
            ContextDecoderLayer(
                d_model=self.d_model,
                num_heads=self.num_heads,
                dff=self.dff,
                dropout_rate=self.dropout_rate
            )
            for _ in range(self.num_layers)
        ]

        self.last_attn_scores = None

    def call(self, input_seq, query_seq, key_seq, value_seq):

        x = self.dropout(input_seq)

        for i in range(self.num_layers):
            x = self.dec_layers[i](input_seq=x, query_seq=query_seq, key_seq=key_seq, value_seq=value_seq)

        #             self.last_attn_scores = self.dec_layers[-1].last_attn_scores

        return x