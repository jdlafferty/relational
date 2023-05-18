# Abstractors

This is the repository associated with
> "Abstractors: Transformer Modules for Symbolic Message Passing and Relational Reasoning" --- Anonymous Authors

The following is an outline of the repo:

- `abstractor.py` implements an $\mathrm{Abstractor}$ which models relations with a $\mathrm{MultiHeadRelation}$ module (implemented in `multi_head_relation.py`) and processes them via symbolic message-passing on a set of symbolic variables.
- `autoregressive_abstractor.py` implements autoregressive (sequence-to-sequence) Abstractor models. This can be configured to implement the architectures $\mathrm{Abstractor} \to \mathrm{Decoder}$, $\mathrm{Encoder} \to \mathrm{Abstractor} \to \mathrm{Decoder}$, as well as the 'sensory-connected' architecture in which the decoder attends to both the Encoder and the Abstractor
- `abstracters.py` implements some other variants of the $\mathrm{Abstractor}$ which use tensorflow's built-in implementation of multi-head attention to perform relational cross-attention. `seq2seq_abstracter_models.py` is an older implementation of sequence-to-sequnce models involving these variants of the Abstractor (`autoregressive_abstractor.py` is more general and can be configured to use these variants of the Abstractor as well.). `multi_head_attention.py` is a fork of tensorflow's implementation which we have adjusted to support different kinds of activation functions applied to the attention scores. `transformer_modules.py` includes implementations of different Transformer modules and utitlities (e.g.: Encoders, Decoders, etc.). Finally, `attention.py` implements different attention mechanisms for Transformers and Abstractors (including relational cross-attention).
- The `experiments` directory contains the code for all experiments in the paper. See the `readme`'s therein for details on the experiments and instructions for replicating them.
- The `paper` directory contains the source for the paper itself.