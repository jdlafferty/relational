import sys; sys.path += ['..', '../..']
from autoregressive_abstractor import AutoregressiveAbstractor
from seq2seq_abstracter_models import Transformer


#region common kwargs

d_model = 512
num_heads = 8
dff = 2048
num_layers = 1

def get_params_by_size(size):
    if size=='small':
        d_model, num_heads, dff, num_layers = (64, 2, 128, 1)
    elif size=='medium':
        d_model, num_heads, dff, num_layers = (128, 4, 256, 1)
    elif size=='large':
        d_model, num_heads, dff, num_layers = (256, 8, 1024, 2)
    elif size=='x-large':
        d_model, num_heads, dff, num_layers = (512, 8, 2048, 2)
    else:
        raise ValueError(f'size {size} invalid')

    return d_model, num_heads, dff, num_layers
#endregion

#region Transformer
def create_transformer(input_vocab_size, target_vocab_size, size='x-large'):
    d_model, num_heads, dff, num_layers = get_params_by_size(size)
    transformer = Transformer(
        num_layers=num_layers, num_heads=num_heads, dff=dff, embedding_dim=d_model,
        input_vocab=input_vocab_size, target_vocab=target_vocab_size,
        output_dim=target_vocab_size, dropout_rate=0.1,)

    return transformer
#endregion


#region Abstractor
def create_abstractor(input_vocab_size, target_vocab_size, size='x-large'):
    d_model, num_heads, dff, num_layers = get_params_by_size(size)
    encoder_kwargs = dict(num_layers=num_layers, num_heads=num_heads, dff=dff, dropout_rate=0.1,)
    decoder_kwargs = dict(num_layers=num_layers, num_heads=num_heads, dff=dff, dropout_rate=0.1,)
    abstractor_kwargs = dict(num_layers=num_layers, rel_dim=num_heads, symbol_dim=d_model,
        proj_dim=d_model//num_heads, symmetric_rels=False, encoder_kwargs=None,
        rel_activation_type='softmax', use_self_attn=False, use_layer_norm=False,
        dropout_rate=0.1)

    abstractor = AutoregressiveAbstractor(
        encoder_kwargs,
        abstractor_kwargs,
        decoder_kwargs,
        input_vocab=input_vocab_size,
        target_vocab=target_vocab_size,
        embedding_dim=d_model,
        output_dim=target_vocab_size,
        abstractor_type='abstractor', # 'abstractor', 'simple', 'relational', or 'symbolic'
        abstractor_on='encoder', # 'input' or 'encoder'
        decoder_on='encoder-abstractor', # 'abstractor' or 'encoder-abstractor'
        name='autoregressive_abstractor')
    return abstractor
#endregion

#region Abstractor
def create_abstractor2(input_vocab_size, target_vocab_size, size='x-large'):
    d_model, num_heads, dff, num_layers = get_params_by_size(size)

    encoder_kwargs = dict(num_layers=num_layers, num_heads=num_heads, dff=dff, dropout_rate=0.1,)
    decoder_kwargs = dict(num_layers=num_layers, num_heads=num_heads, dff=dff, dropout_rate=0.1,)
    abstractor_kwargs = dict(num_layers=num_layers, rel_dim=num_heads, symbol_dim=d_model,
        proj_dim=d_model//num_heads, symmetric_rels=False, encoder_kwargs=None,
        rel_activation_type='tanh', use_self_attn=True, use_layer_norm=False,
        dropout_rate=0.1)

    abstractor = AutoregressiveAbstractor(
        encoder_kwargs,
        abstractor_kwargs,
        decoder_kwargs,
        input_vocab=input_vocab_size,
        target_vocab=target_vocab_size,
        embedding_dim=d_model,
        output_dim=target_vocab_size,
        abstractor_type='abstractor', # 'abstractor', 'simple', 'relational', or 'symbolic'
        abstractor_on='encoder', # 'input' or 'encoder'
        decoder_on='encoder-abstractor', # 'abstractor' or 'encoder-abstractor'
        name='autoregressive_abstractor')
    return abstractor
#endregion

#region RelationalAbstractor
def create_relational_abstractor(input_vocab_size, target_vocab_size, size='x-large'):
    d_model, num_heads, dff, num_layers = get_params_by_size(size)

    encoder_kwargs = dict(num_layers=num_layers, num_heads=num_heads, dff=dff, dropout_rate=0.1,)
    decoder_kwargs = dict(num_layers=num_layers, num_heads=num_heads, dff=dff, dropout_rate=0.1,)
    abstractor_kwargs = dict(num_layers=num_layers, num_heads=num_heads, dff=dff,
        use_learned_symbols=False, mha_activation_type='softmax', use_self_attn=True)

    abstractor = AutoregressiveAbstractor(
        encoder_kwargs,
        abstractor_kwargs,
        decoder_kwargs,
        input_vocab=input_vocab_size,
        target_vocab=target_vocab_size,
        embedding_dim=d_model,
        output_dim=target_vocab_size,
        abstractor_type='relational', # 'abstractor', 'simple', 'relational', or 'symbolic'
        abstractor_on='encoder', # 'input' or 'encoder'
        decoder_on='encoder-abstractor', # 'abstractor' or 'encoder-abstractor'
        name='autoregressive_abstractor')
    return abstractor
#endregion

#region RelationalAbstractor
def create_relational_abstractor2(input_vocab_size, target_vocab_size, size='x-large'):
    d_model, num_heads, dff, num_layers = get_params_by_size(size)

    encoder_kwargs = dict(num_layers=num_layers, num_heads=num_heads, dff=dff, dropout_rate=0.1,)
    decoder_kwargs = dict(num_layers=num_layers, num_heads=num_heads, dff=dff, dropout_rate=0.1,)
    abstractor_kwargs = dict(num_layers=num_layers, num_heads=num_heads, dff=dff,
        use_learned_symbols=False, mha_activation_type='softmax', use_self_attn=True)

    abstractor = AutoregressiveAbstractor(
        encoder_kwargs,
        abstractor_kwargs,
        decoder_kwargs,
        input_vocab=input_vocab_size,
        target_vocab=target_vocab_size,
        embedding_dim=d_model,
        output_dim=target_vocab_size,
        abstractor_type='relational', # 'abstractor', 'simple', 'relational', or 'symbolic'
        abstractor_on='input', # 'input' or 'encoder'
        decoder_on='encoder-abstractor', # 'abstractor' or 'encoder-abstractor'
        name='autoregressive_abstractor')
    return abstractor
#endregion

model_creator_dict = dict(
    transformer=create_transformer,
    abstractor=create_abstractor,
    abstractor2=create_abstractor2,
    relational_abstractor=create_relational_abstractor,
    relational_abstractor2=create_relational_abstractor2
    )
