> It is not clear to me how S works. Section 2.2 says "The symbols S can be either learned parameters of the model or nonparametric positional embeddings." How is S different from a positional embedding then? If S is unique per token, even if the token is a repeat of another token value-wise, then S is essentially a positional embedding. If S is unique per value of a token (ie all tokens of value v have the same s_i), then doesn't S implicity encode the value of the token?

Please see the updated paper for an expanded discussion which is relevant to this question. In particular, please see section 2.3.

In terms of implementation, "positional symbols" are indeed positional embeddings, which can be learned parameters of the model or fixed positional embeddings. Hence, the symbol assigned to each object does not encode its features, but identifies it (in this case, via its position). In the case of "position-relative" symbols, the identifier of a symbol encodes its relative position with respect to the 'receiver'. In the case of "symbolic attention," a symbol is retrieved from a library of learned symbols (which are parameters of the model) via attention. This allows the symbol retrieved for each object to encode something about its role, hence identifying it not only by its position, but by its role.

The key inductive bias for all these symbol-assignment mechanisms is that the features of the input objects don't propagate through the values $V$ in relational cross-attention. I.e., instead of the self-attention configuration $(Q \gets X, K \gets X, V \gets X)$, in relational cross-attention we have $(Q \gets X, K \gets X, V \gets S)$, with $S$ identifying objects without encoding their features. We think that the message-passing interpretation of attention is useful to get some intuition about this.

Self attention can be thought of as a form of message-passing where the message from object $j$ to object $i$ encodes the sender's features as well as the relations between the two objects, $m_{j \to i} = (\phi_v(x_j), r(x_i, x_j))$, where $r(x_i, x_j) = \langle \phi_q(x_i), \phi_k(x_j)\rangle$ is an inner product relation. Then,
$$E_i = \mathrm{MessagePassing}(\{m_{j \to i}\}_j).$$

(To make this more concrete, one could define a commutative monoid with the aggregation operation $\oplus$ as $(x_1, r_1) \oplus (x_2, r_2) = \left(\frac{\exp(r_1) x_1 + \exp(r_2) x_2}{\exp(r_1) + \exp(r_2)}, \exp(r_1) + \exp(r_2)\right)$ to formalize self-attention in the standard Message Passing Neural Networks equations. But those details are not important to the intuition.)

Hence, the 'messages' in self-attention entangle relational information $r(x_i, x_j)$ with object-level information $\phi_v(x_j)$. The aim of the Abstractor is to produce disentangled relational representations, within the powerful framework of Transformers. The key idea is to replace $\phi_v(x_j)$ in $m_{j \to i}$ with vectors identifying the sender $j$ but not encoding any information about its object-level features---we call these vectors 'symbols'. The simplest way to do this is to simply sequentially assign each object in the input sequence a unique symbol, based on the position it appears (i.e., "positional symbols"). Although very simple, this works well in practice. We also explored symbolic attention as a symbol-assignment mechanism in our updated experiments (see section 4.3).

---------
> If the outputs of an abstractor layer are "abstract states that represent purely relational information" (section 2.3), then how are features associated with objects passed along/learned in models that use abstractor layers? In all the examples presented in Figure 2, abstractors are used in conjunction with regular encoders. How can you ensure that the abstractor is learning meaningful information and that all the "information flow" does not happen through the encoder layers in parlalle setups (architectures c, d, and e)?

Thanks for this question. A minor correction, architecture (a) of Figure 2 depicts a model without a standard Encoder. The "Ability to generalize to similar tasks" experiment of section 4.2 uses architecture (a). In the discriminative relational tasks of section 4.1, the Abstractor models also do not include an Encoder (this is an Abstractor-only model with no decoder either, since these are classification tasks).

But indeed, many architectures we consider do include an Encoder. The reason for this is that most real-world tasks rely on relational reasoning as well as more general-purpose sequence modeling which needs object-level features. In the architectures proposed in Section 3 and Figure 2 (e.g., architectures c, d, e), we think of the Abstractor as performing specialized relational processing in a branch of the model, while an Encoder performs more general-purpose processing in another branch. The Decoder then sequentially attends to the outputs of both.

> then how are features associated with objects passed along/learned in models that use abstractor layers?

This is done by the (multi-attention) Decoder when it integrates the information from the Abstractor and the Encoder.

> How can you ensure that the abstractor is learning meaningful information and that all the "information flow" does not happen through the encoder layers in parlalle setups

This is indicated by the performance difference compared to models which do not contain an Abstractor. For example, in the experiments of section 4.2, we compare to a standard Transformer of similar size as well as to an ablation model with matching architecture but with standard cross-attention rather than relational cross-attention. We observe a dramatic difference in performance. In the experiments of section 4.3, we compare to two Transformers of different size (one matching the hyperparameters of the Encoder/Decoder and a larger one to match parameter count). We observe a consistent difference in performance here as well.

Through comparison to these controlled baselines, we can attribute the performance difference to the addition of an Abstractor module. If the Encoder was doing all the work, the Abstractor-based models would not perform better.

---------
> Is there an ablation on model size? The baseline transformer is not very large, but it may be that a smaller transformer can learn with fewer training data points, or a larger transformer may converge faster.

Please see the answer to the previous question. We did include controls to test for the following question: given a certain budget of model size (as measured by parameter count), which would yield a greater performance, incorporating an Abstractor or simply enlarging a standard Transformer? We find that incorporating an Abstractor tends to yield greater performance benefits.

This was for a fixed model size, however. It would be interesting to see whether these performance benefits are persistent across model size and task complexity. We leave this to future work.

---------
> The experiments seem to be on simple problems with small models. Simple problems may not necessarily be an issue since they are relatively diverse problems, but it would be nice to see larger, more complicated problems. For example, the partial order task training set is small enough that one could consider using in context learning with a large LLM, which may offer comparable performance.

We agree that it would be interesting and important to evaluate Abstractors on larger, more complex problems. For now, this is outside the scope of this initial paper which aims to propose the idea and perform controlled evaluations.