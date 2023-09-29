import tensorflow as tf
from tensorflow.keras import layers, Model

from transformer_modules import Encoder, Decoder, AddPositionalEmbedding
from abstracters import SymbolicAbstracter, RelationalAbstracter, SimpleAbstractor, AblationAbstractor


class Transformer(tf.keras.Model):
    def __init__(self, num_layers, num_heads, dff,
            input_vocab, target_vocab, embedding_dim, output_dim,
            dropout_rate=0.1, name='transformer'):
        """A transformer model.

        Args:
            num_layers (int): # of layers in encoder and decoder
            num_heads (int): # of attention heads in attention operations
            dff (int): dimension of feedforward laeyrs
            input_vocab (int or str): if input is tokens, the size of vocabulary as an int. 
                if input is vectors, the string 'vector'. used to create embedder.
            target_vocab (int): if target is tokens, the size of the vocabulary as an int. 
                if input is vectors, the string 'vector'. used to create embedder.
            embedding_dim (int): embedding dimension to use. this is the model dimension.
            output_dim (int): dimension of final output. e.g.: # of classes.
            dropout_rate (float, optional): dropout rate. Defaults to 0.1.
            name (str, optional): name of model. Defaults to 'transformer'.
        """

        super().__init__(name=name)

        if isinstance(input_vocab, int):
            self.source_embedder = layers.Embedding(input_vocab, embedding_dim, name='source_embedder')
        elif input_vocab == 'vector':
            self.source_embedder = layers.TimeDistributed(layers.Dense(embedding_dim), name='source_embedder')
        else:
            raise ValueError(
                "`input_vocab` must be an integer if the input sequence is token-valued or "
                "'vector' if the input sequence is vector-valued.")

        if isinstance(target_vocab, int):
            self.target_embedder = layers.Embedding(target_vocab, embedding_dim, name='target_embedder')
        elif target_vocab == 'vector':
            self.target_embedder = layers.TimeDistributed(layers.Dense(embedding_dim), name='target_embedder')
        else:
            raise ValueError(
                "`input_vocab` must be an integer if the input sequence is token-valued or "
                "'vector' if the input sequence is vector-valued.")

        self.pos_embedding_adder_input = AddPositionalEmbedding(name='add_pos_embedding_input')
        self.pos_embedding_adder_target = AddPositionalEmbedding(name='add_pos_embedding_target')

        self.encoder = Encoder(num_layers=num_layers, num_heads=num_heads, dff=dff, dropout_rate=dropout_rate, name='encoder')
        self.decoder = Decoder(num_layers=num_layers, num_heads=num_heads, dff=dff,
          dropout_rate=dropout_rate, name='decoder')
        

        self.final_layer = layers.Dense(output_dim, name='final_layer')


    def call(self, inputs):
        source, target  = inputs

        x = self.source_embedder(source)
        x = self.pos_embedding_adder_input(x)

        encoder_context = self.encoder(x)

        target_embedding = self.target_embedder(target)
        target_embedding = self.pos_embedding_adder_target(target_embedding)

        x = self.decoder(x=target_embedding, context=encoder_context)

        logits = self.final_layer(x) 

        try:
          # Drop the keras mask, so it doesn't scale the losses/metrics.
          # b/250038731
          del logits._keras_mask
        except AttributeError:
          pass

        return logits


class Seq2SeqRelationalAbstracter(tf.keras.Model):
    def __init__(self, num_layers, num_heads, dff, rel_attention_activation,
            input_vocab, target_vocab, embedding_dim, output_dim,
            dropout_rate=0.1, name='seq2seq_relational_abstracter'):
        """
        Sequence-to-Sequence Relational Abstracter.

        Args:
            num_layers (int): # of layers in encoder and decoder
            num_heads (int): # of attention heads in attention operations
            dff (int): dimension of feedforward layers
            rel_attention_activation (str): the activation function to use in relational attention.
            input_vocab (int or str): if input is tokens, the size of vocabulary as an int. 
                if input is vectors, the string 'vector'. used to create embedder.
            target_vocab (int): if target is tokens, the size of the vocabulary as an int. 
                if input is vectors, the string 'vector'. used to create embedder.
            embedding_dim (int): embedding dimension to use. this is the model dimension.
            output_dim (int): dimension of final output. e.g.: # of classes.
            dropout_rate (float, optional): dropout rate. Defaults to 0.1.
            name (str, optional): name of model. Defaults to 'seq2seq_relational_abstracter'.
        """

        super().__init__(name=name)

        if isinstance(input_vocab, int):
            self.source_embedder = layers.Embedding(input_vocab, embedding_dim, name='source_embedder')
        elif input_vocab == 'vector':
            self.source_embedder = layers.TimeDistributed(layers.Dense(embedding_dim), name='source_embedder')
        else:
            raise ValueError(
                "`input_vocab` must be an integer if the input sequence is token-valued or "
                "'vector' if the input sequence is vector-valued.")

        if isinstance(target_vocab, int):
            self.target_embedder = layers.Embedding(target_vocab, embedding_dim, name='target_embedder')
        elif target_vocab == 'vector':
            self.target_embedder = layers.TimeDistributed(layers.Dense(embedding_dim), name='target_embedder')
        else:
            raise ValueError(
                "`input_vocab` must be an integer if the input sequence is token-valued or "
                "'vector' if the input sequence is vector-valued.")

        self.pos_embedding_adder_input = AddPositionalEmbedding(name='add_pos_embedding_input')
        self.pos_embedding_adder_target = AddPositionalEmbedding(name='add_pos_embedding_target')

        self.encoder = Encoder(num_layers=num_layers, num_heads=num_heads, dff=dff, dropout_rate=dropout_rate, name='encoder')
        self.abstracter = RelationalAbstracter(num_layers=num_layers, num_heads=num_heads, dff=dff,
            mha_activation_type=rel_attention_activation, dropout_rate=dropout_rate, name='abstracter')
        self.decoder = Decoder(num_layers=num_layers, num_heads=num_heads, dff=dff,
          dropout_rate=dropout_rate, name='decoder')
        self.final_layer = layers.Dense(output_dim, name='final_layer')


    def call(self, inputs):
        source, target  = inputs

        x = self.source_embedder(source)
        x = self.pos_embedding_adder_input(x)

        encoder_context = self.encoder(x)

        abstracted_context = self.abstracter(encoder_context)

        target_embedding = self.target_embedder(target)
        target_embedding = self.pos_embedding_adder_target(target_embedding)

        x = self.decoder(x=target_embedding, context=abstracted_context)

        logits = self.final_layer(x)

        try:
          # Drop the keras mask, so it doesn't scale the losses/metrics.
          # b/250038731
          del logits._keras_mask
        except AttributeError:
          pass

        return logits


class Seq2SeqSymbolicAbstracter(tf.keras.Model):
    def __init__(self, num_layers, num_heads, dff, rel_attention_activation,
            input_vocab, target_vocab, embedding_dim, output_dim,
            dropout_rate=0.1, name='seq2seq_symbolic_abstracter'):
        """
        Sequence-to-Sequence Symbolic Abstracter.

        Args:
            num_layers (int): # of layers in encoder and decoder
            num_heads (int): # of attention heads in attention operations
            dff (int): dimension of feedforward layers
            rel_attention_activation (str): the activation function to use in relational attention.
            input_vocab (int or str): if input is tokens, the size of vocabulary as an int. 
                if input is vectors, the string 'vector'. used to create embedder.
            target_vocab (int): if target is tokens, the size of the vocabulary as an int. 
                if input is vectors, the string 'vector'. used to create embedder.
            embedding_dim (int): embedding dimension to use. this is the model dimension.
            output_dim (int): dimension of final output. e.g.: # of classes.
            dropout_rate (float, optional): dropout rate. Defaults to 0.1.
            name (str, optional): name of model. Defaults to 'seq2seq_symbolic_abstracter'.
        """

        super().__init__(name=name)

        if isinstance(input_vocab, int):
            self.source_embedder = layers.Embedding(input_vocab, embedding_dim, name='source_embedder')
        elif input_vocab == 'vector':
            self.source_embedder = layers.TimeDistributed(layers.Dense(embedding_dim), name='source_embedder')
        else:
            raise ValueError(
                "`input_vocab` must be an integer if the input sequence is token-valued or "
                "'vector' if the input sequence is vector-valued.")

        if isinstance(target_vocab, int):
            self.target_embedder = layers.Embedding(target_vocab, embedding_dim, name='target_embedder')
        elif target_vocab == 'vector':
            self.target_embedder = layers.TimeDistributed(layers.Dense(embedding_dim), name='target_embedder')
        else:
            raise ValueError(
                "`input_vocab` must be an integer if the input sequence is token-valued or "
                "'vector' if the input sequence is vector-valued.")

        self.pos_embedding_adder_input = AddPositionalEmbedding(name='add_pos_embedding_input')
        self.pos_embedding_adder_target = AddPositionalEmbedding(name='add_pos_embedding_target')

        self.encoder = Encoder(num_layers=num_layers, num_heads=num_heads, dff=dff, dropout_rate=dropout_rate, name='encoder')
        self.abstracter = SymbolicAbstracter(num_layers=num_layers, num_heads=num_heads, dff=dff,
            mha_activation_type=rel_attention_activation, dropout_rate=dropout_rate, name='abstracter')
        self.decoder = Decoder(num_layers=num_layers, num_heads=num_heads, dff=dff,
          dropout_rate=dropout_rate, name='decoder')
        self.final_layer = layers.Dense(output_dim, name='final_layer')


    def call(self, inputs):
        source, target  = inputs

        x = self.source_embedder(source)
        x = self.pos_embedding_adder_input(x)

        encoder_context = self.encoder(x)

        abstracted_context = self.abstracter(encoder_context)

        target_embedding = self.target_embedder(target)
        target_embedding = self.pos_embedding_adder_target(target_embedding)

        x = self.decoder(x=target_embedding, context=abstracted_context)

        logits = self.final_layer(x)

        try:
          # Drop the keras mask, so it doesn't scale the losses/metrics.
          # b/250038731
          del logits._keras_mask
        except AttributeError:
          pass

        return logits

class AutoregressiveSimpleAbstractor(tf.keras.Model):
    def __init__(self, input_vocab, target_vocab, output_dim, 
        embedding_dim, abstractor_kwargs, decoder_kwargs, name=None):
        """create autoregressive SimpleAbstractor model.

        (x1, ..., xm) -> embedder -> SimpleAbstractor -> Decoder -> (y1, ..., ym)

        Parameters
        ----------
        input_vocab : int or str
            if input is tokens, the size of vocabulary as an int. 
            if input is vectors, the string 'vector'. used to create embedder.
        target_vocab : int or str
            if target is tokens, the size of the vocabulary as an int. 
            if input is vectors, the string 'vector'. used to create embedder.
        output_dim : int
            dimension of final output. e.g.: # of classes.
        embedding_dim : int
            embedding dimension to use. this is the model dimension.
        abstractor_kwargs : dict
            kwargs for SimpleAbstractor
        decoder_kwargs : dict
            kwargs for Decoder
        name : str, optional
            name of model, by default None
        """

        super().__init__(name=name)

        self.embedding_dim = embedding_dim
        self.input_vocab = input_vocab
        self.target_vocab = target_vocab
        self.output_dim = output_dim
        self.abstractor_kwargs = abstractor_kwargs
        self.decoder_kwargs = decoder_kwargs
  
    def build(self, input_shape):
    
        if isinstance(self.input_vocab, int):
            self.source_embedder = layers.Embedding(self.input_vocab, self.embedding_dim, name='source_embedder')
        elif self.input_vocab == 'vector':
            self.source_embedder = layers.TimeDistributed(layers.Dense(self.embedding_dim), name='source_embedder')
        else:
            raise ValueError(
                "`input_vocab` must be an integer if the input sequence is token-valued or "
                "'vector' if the input sequence is vector-valued.")

        if isinstance(self.target_vocab, int):
            self.target_embedder = layers.Embedding(self.target_vocab, self.embedding_dim, name='target_embedder')
        elif self.target_vocab == 'vector':
            self.target_embedder = layers.TimeDistributed(layers.Dense(self.embedding_dim), name='target_embedder')
        else:
            raise ValueError(
                "`input_vocab` must be an integer if the input sequence is token-valued or "
                "'vector' if the input sequence is vector-valued.")

        self.pos_embedding_adder_input = AddPositionalEmbedding(name='add_pos_embedding_input')
        self.pos_embedding_adder_target = AddPositionalEmbedding(name='add_pos_embedding_target')

        self.abstractor = SimpleAbstractor(**self.abstractor_kwargs)

        self.decoder = Decoder(**self.decoder_kwargs, name='decoder')
        self.final_layer = layers.Dense(self.output_dim, name='final_layer')

    def call(self, inputs):
        source, target  = inputs

        x = self.source_embedder(source)
        x = self.pos_embedding_adder_input(x)

        abstracted_context = self.abstractor(x)

        target_embedding = self.target_embedder(target)
        target_embedding = self.pos_embedding_adder_target(target_embedding)

        x = self.decoder(x=target_embedding, context=abstracted_context)

        logits = self.final_layer(x)

        try:
          # Drop the keras mask, so it doesn't scale the losses/metrics.
          # b/250038731
          del logits._keras_mask
        except AttributeError:
          pass

        return logits


class AutoregressiveAblationAbstractor(tf.keras.Model):
    def __init__(self, num_layers, num_heads, dff, mha_activation_type,
            input_vocab, target_vocab, embedding_dim, output_dim,
            use_encoder, use_self_attn,
            dropout_rate=0.1, name='seq2seq_ablation_abstractor'):
        """
        Sequence-to-Sequence Ablation Abstracter.

        A Seq2Seq Abstractor model where the abstractor's cross-attention
        scheme is standard cross-attention rather than relation cross-attention.
        Used to isolate for the effect of relational cross-attention in abstractor models.

        Args:
            num_layers (int): # of layers in encoder and decoder
            num_heads (int): # of attention heads in attention operations
            dff (int): dimension of feedforward layers
            mha_activation_type (str): the activation function to use in AblationAbstractor's cross-attention.
            input_vocab (int or str): if input is tokens, the size of vocabulary as an int. 
                if input is vectors, the string 'vector'. used to create embedder.
            target_vocab (int): if target is tokens, the size of the vocabulary as an int. 
                if input is vectors, the string 'vector'. used to create embedder.
            embedding_dim (int): embedding dimension to use. this is the model dimension.
            output_dim (int): dimension of final output. e.g.: # of classes.
            use_encoder (bool): whether to use a (Transformer) Encoder as first step of processing.
            use_self_attn (bool): whether to use self-attention in AblationAbstractor.
            dropout_rate (float, optional): dropout rate. Defaults to 0.1.
            name (str, optional): name of model. Defaults to 'seq2seq_relational_abstracter'.
        """

        super().__init__(name=name)

        if isinstance(input_vocab, int):
            self.source_embedder = layers.Embedding(input_vocab, embedding_dim, name='source_embedder')
        elif input_vocab == 'vector':
            self.source_embedder = layers.TimeDistributed(layers.Dense(embedding_dim), name='source_embedder')
        else:
            raise ValueError(
                "`input_vocab` must be an integer if the input sequence is token-valued or "
                "'vector' if the input sequence is vector-valued.")

        if isinstance(target_vocab, int):
            self.target_embedder = layers.Embedding(target_vocab, embedding_dim, name='target_embedder')
        elif target_vocab == 'vector':
            self.target_embedder = layers.TimeDistributed(layers.Dense(embedding_dim), name='target_embedder')
        else:
            raise ValueError(
                "`input_vocab` must be an integer if the input sequence is token-valued or "
                "'vector' if the input sequence is vector-valued.")

        self.pos_embedding_adder_input = AddPositionalEmbedding(name='add_pos_embedding_input')
        self.pos_embedding_adder_target = AddPositionalEmbedding(name='add_pos_embedding_target')

        self.use_encoder = use_encoder
        self.use_self_attn = use_self_attn
        if self.use_encoder:
            self.encoder = Encoder(num_layers=num_layers, num_heads=num_heads, dff=dff,
            dropout_rate=dropout_rate, name='encoder')
        self.abstractor = AblationAbstractor(num_layers=num_layers, num_heads=num_heads, dff=dff,
            mha_activation_type=mha_activation_type, use_self_attn=use_self_attn, dropout_rate=dropout_rate,
            name='ablation_abstractor')
        self.decoder = Decoder(num_layers=num_layers, num_heads=num_heads, dff=dff,
          dropout_rate=dropout_rate, name='decoder')
        self.final_layer = layers.Dense(output_dim, name='final_layer')


    def call(self, inputs):
        source, target  = inputs

        x = self.source_embedder(source)
        x = self.pos_embedding_adder_input(x)

        if self.use_encoder:
            encoder_context = self.encoder(x)
        else:
            encoder_context = x

        abstracted_context = self.abstractor(encoder_context)

        target_embedding = self.target_embedder(target)
        target_embedding = self.pos_embedding_adder_target(target_embedding)

        x = self.decoder(x=target_embedding, context=abstracted_context)

        logits = self.final_layer(x)

        try:
          # Drop the keras mask, so it doesn't scale the losses/metrics.
          # b/250038731
          del logits._keras_mask
        except AttributeError:
          pass

        return logits
