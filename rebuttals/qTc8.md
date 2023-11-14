> It is not clear to me how S works. Section 2.2 says "The symbols S can be either learned parameters of the model or nonparametric positional embeddings." How is S different from a positional embedding then? If S is unique per token, even if the token is a repeat of another token value-wise, then S is essentially a positional embedding. If S is unique per value of a token (ie all tokens of value v have the same s_i), then doesn't S implicity encode the value of the token?

In terms of implementation, the symbols $S$ are indeed positional embeddings (learned or nonparametric). The 

> If the outputs of an abstractor layer are "abstract states that represent purely relational information" (section 2.3), then how are features associated with objects passed along/learned in models that use abstractor layers? In all the examples presented in Figure 2, abstractors are used in conjunction with regular encoders. How can you ensure that the abstractor is learning meaningful information and that all the "information flow" does not happen through the encoder layers in parlalle setups (architectures c, d, and e)?

...

> Is there an ablation on model size? The baseline transformer is not very large, but it may be that a smaller transformer can learn with fewer training data points, or a larger transformer may converge faster.

...

> The experiments seem to be on simple problems with small models. Simple problems may not necessarily be an issue since they are relatively diverse problems, but it would be nice to see larger, more complicated problems. For example, the partial order task training set is small enough that one could consider using in context learning with a large LLM, which may offer comparable performance.

...

> The experiments seem to be on simple problems with small models. Simple problems may not necessarily be an issue since they are relatively diverse problems, but it would be nice to see larger, more complicated problems. For example, the partial order task training set is small enough that one could consider using in context learning with a large LLM, which may offer comparable performance.
