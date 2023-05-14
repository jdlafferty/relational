import tensorflow as tf
from tensorflow.keras import layers
from multi_head_relation import MultiHeadRelation
from transformer_modules import GlobalSelfAttention

class Abstractor(tf.keras.layers.Layer):
    def __init__(self,
        num_layers,
        rel_dim,
        symbol_dim=None,
        proj_dim=None,
        symmetric_rels=False,
        encoder_kwargs=None,
        rel_activation_type='softmax',
        use_self_attn=False,
        dropout_rate=0.,
        name=None):

        super().__init__(name=None)

        self.num_layers = num_layers
        self.rel_dim = rel_dim
        self.proj_dim = proj_dim
        self.symmetric_rels = symmetric_rels
        self.encoder_kwargs = encoder_kwargs
        self.symbol_dim = symbol_dim
        self.rel_activation_type = rel_activation_type
        self.use_self_attn = use_self_attn
        self.dropout_rate = dropout_rate
    
    def build(self, input_shape):

        _, self.sequence_length, self.object_dim = input_shape

        # symbol_dim is not given, use same dimension as objects
        if self.symbol_dim is None:
            self.symbol_dim = self.object_dim
        
        if self.proj_dim is None:
            self.proj_dim = self.object_dim

        # define the input-independent symbolic input vector sequence
        normal_initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)
        self.symbol_sequence = tf.Variable(
            normal_initializer(shape=(self.sequence_length, self.symbol_dim)),
            name='symbols', trainable=True)

        if self.use_self_attn:
            self.self_attention_layers = [GlobalSelfAttention(
                num_heads=self.rel_dim,
                key_dim=self.proj_dim,
                activation_type='softmax',
                dropout=self.dropout_rate) for _ in range(self.num_layers)]

        # MultiHeadRelation layer for each layer of Abstractor
        self.multi_head_relation_layers = [MultiHeadRelation(
            rel_dim=self.rel_dim, proj_dim=self.proj_dim,
            symmetric=self.symmetric_rels, dense_kwargs=self.encoder_kwargs)
            for _ in range(self.num_layers)]

        if self.rel_activation_type == 'softmax':
            self.rel_activation = tf.keras.layers.Softmax(axis=-2)
        else:
            self.rel_activation = tf.keras.layers.Activation(self.rel_activation_type)

        # a Reshape layer which collapses the final 'relation' dimension
        self.symbol_collapser = tf.keras.layers.Reshape(
            target_shape=(self.sequence_length, self.symbol_dim*self.rel_dim))

        # create dense layers to be applied after symbolic message-passing
        # (these transform the symbol sequence from dimension d_s * d_r to original dimension, d_s)
        # TODO: make these configurable
        self.symbol_dense_layers = [layers.Dense(self.symbol_dim) for _ in range(self.num_layers)]

        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)


    def call(self, inputs):
    
    
        for i in range(self.num_layers):

            # get relation tensor via MultiHeadRelation layer
            rel_tensor = self.multi_head_relation_layers[i](inputs) # shape: [b, m, m, d_r]

            # apply activation to relation tensor (e.g.: softmax)
            rel_tensor = self.rel_activation(rel_tensor)

            # perform symbolic message-passing based on relation tensor
            # A_bijr = sum_k R_bikr S_bkj (A = S.T @ R)
            if i == 0: # on first iteration, symbol equence is untransformed of shape [m, d_s]
                abstract_symbol_seq = tf.einsum('bikr,kj->bijr', rel_tensor, self.symbol_sequence) # shape: [b, m, d_s, d_r]
            else: # on next iterations, symbol sequence is transformed with shape [b, m, d_s]
                abstract_symbol_seq = tf.einsum('bikr,bkj->bijr', rel_tensor, abstract_symbol_seq) # shape: [b, m, d_s, d_r]

            # symbol_seq = tf.matmul(symbol_seq, rel_tensor, transpose_a=True) # shape: [b, m, d_s, d_r]

            # reshape to collapse final 'relation' dimension
            abstract_symbol_seq = self.symbol_collapser(abstract_symbol_seq) # shape: [b, m, d_s * d_r]

            # transform symbol sequence via dense layer to return to its original dimension
            abstract_symbol_seq = self.symbol_dense_layers[i](abstract_symbol_seq) # shape: [b, m, d_s]

            # apply self-attention to symbol sequence 
            if self.use_self_attn:
                # need to expand dims to add batch dim first
                abstract_symbol_seq = self.self_attention_layers[i](abstract_symbol_seq) # shape [b, m, d_s]

            # dropout
            abstract_symbol_seq = self.dropout(abstract_symbol_seq)

        return abstract_symbol_seq

    def get_config(self):
        config = super(Abstractor, self).get_config()

        config.update(
            {
                'num_layers': self.num_layers,
                'rel_dim': self.rel_dim,
                'proj_dim': self.proj_dim,
                'symmetric_rels': self.symmetric_rels,
                'encoder_kwargs': self.encoder_kwargs,
                'symbol_dim': self.symbol_dim,
                'rel_activation_type': self.rel_activation_type,
                'dropout_rate': self.dropout_rate
            })
        
        return config