# implement a general cross-attention layer which supports all the different kinds of attention

from seq2seq_transformer import BaseAttention
from seq2seq_transformer import CausalSelfAttention, FeedForward

class ContextualCrossAttention(BaseAttention):
    """A general layer for implementing cross-attention, with configurable queries, keys, and values"""

    def __init__(self, cross_attention_type="standard", **kwargs):

        super(ContextualCrossAttention, self).__init__(**kwargs)

        if cross_attention_type in ("std_encoder_decoder", "symbolic", "relational"):
            self.cross_attention_type = cross_attention_type
        else:
            raise ValueError(f"`cross_attention_type` {cross_attention_type} is invalid")

    def call(self, input_seq, context_seq):

        if self.cross_attention_type == "std_encoder_decoder":
            # standard encoder-decoder cross-attention of transformers
            attn_output, attn_scores = self.mha(
                query=input_seq,
                key=context_seq,
                value=context_seq,
                return_attention_scores=True,
            )

            x = self.add([input_seq, attn_output])

            x = self.layernorm(x)

        elif self.cross_attention_type == "symbolic":
            # 'symbolic' cross-attention.
            #  input_seq is learned input-independent symbols
            attn_output, attn_scores = self.mha(
                query=input_seq,
                key=context_seq,
                value=input_seq,
                return_attention_scores=True,
            )

            x = self.add(
                [input_seq, attn_output]
            )  # TODO: think about this. should we keep this skip connection?

            x = self.layernorm(x)

        elif self.cross_attention_type == "relational":
            # 'relational' cross-attention.
            # queries and keys both come from the context sequence, thus their inner product computes relations
            attn_output, attn_scores = self.mha(
                query=context_seq,
                key=context_seq,
                value=input_seq,
                return_attention_scores=True,
            )

            x = self.add(
                [input_seq, attn_output]
            )  # TODO: think about this. should we keep this skip connection?

            x = self.layernorm(x)

        else:
            raise ValueError("unexpected `cross_attention_type`")

        # Cache the attention scores for plotting later.
        self.last_attn_scores = attn_scores

        return x


class ContextDecoderLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        d_model,
        num_heads,
        dff,
        cross_attention_type="std_encoder_decoder",
        dropout_rate=0.1,
    ):

        super(ContextDecoderLayer, self).__init__()

        self.cross_attention_type = cross_attention_type

        self.causal_self_attention = CausalSelfAttention(
            num_heads=num_heads, key_dim=d_model, dropout=dropout_rate
        )

        self.cross_attention = ContextualCrossAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate,
            cross_attention_type=self.cross_attention_type,
        )

        self.ffn = FeedForward(d_model, dff)

    def call(self, input_seq, context_seq):
        x = self.causal_self_attention(x=input_seq)
        x = self.cross_attention(input_seq=x, context_seq=context_seq)

        # Cache the last attention scores for plotting later
        self.last_attn_scores = self.cross_attention.last_attn_scores

        x = self.ffn(x)  # Shape `(batch_size, seq_len, d_model)`.

        return x


class ContextDecoder(tf.keras.layers.Layer):
    def __init__(
        self,
        num_layers,
        num_heads,
        dff,
        cross_attention_type="std_encoder_decoder",
        dropout_rate=0.1,
        name="decoder",
    ):

        super(ContextDecoder, self).__init__(name=name)

        self.cross_attention_type = cross_attention_type
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
                dropout_rate=self.dropout_rate,
                cross_attention_type=self.cross_attention_type,
            )
            for _ in range(self.num_layers)
        ]

        self.last_attn_scores = None

    def call(self, input_seq, context_seq):

        x = self.dropout(input_seq)

        for i in range(self.num_layers):
            x = self.dec_layers[i](input_seq=x, context_seq=context_seq)

        #             self.last_attn_scores = self.dec_layers[-1].last_attn_scores

        return x
