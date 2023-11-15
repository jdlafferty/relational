Thank you for your thoughts and feedback! We are glad you found our work interesting.

> One minor exception is the omission of a baseline transformer for the Order Relation and SET tasks, though this was presumably an intentional omission based on the already superior performance of CoRelNet compared to Transformers?

Indeed, relational architectures (like CoRelNet) tend to perform better than Transformers on such discriminative relational tasks. Though, so far, relational architectures have not explicitly considered the generative or sequence-to-sequence setting. For this reason, for the discriminative tasks, we decided to focus our comparison to existing relational architectures, whereas for sequence-to-sequence tasks we focus our comparison on Transformers.

Of relevance here, one relational architecture we did not include initially as a baseline is the PrediNet architecture (Shanahan et al.). We have now added it to the list of baselines for the discriminative experiments. For completeness, we also added an MLP baseline (as expected, this performs worse than the relational architectures). You can find these results in the updated pdf.

> There is one consistently made claim which may be slightly overstated (this is a question to the authors) - namely that the output of Abstractors represents "purely relational information"; I believe this only holds if there are no residual connections in the abstractor module implementation being used (which is offered as an option); if there is a residual connection, then it seems the abstractor MLP could still learn to operate on information present in object-level representations.

The residual connection adds the *abstract states* from the previous layer $A^{(l-1)}$, not the object representations $X$. Hence, even with a residual connection, the information represented in $A^{(l)}$ should still be purely relational---albeit now incorporating relational information at multiple layers of hierarchy.

> Multi-attention decoder: Perhaps this form of decoder is standard, but why is CausalSelfAttention used before Cross-Attention (which is not causally masked?)

The paper ``Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer'' by Raffel et al. has some discussion on causal masking which might be useful (e.g., see section 3.2.1, Fig 3, and Fig 4. Autoregressive encoder-decoder models are depicted in the left-hand side of Fig 4). In an autoregressive Encoder-Decoder architecture, the input to the decoder is the right-shifted target sequence, and causal masking prevents attending to object $j > i$ when predicting the target at $i$. This allows for an encoder-decoder architecture to autoregressively produce an output sequence. But, the entire "context sequence" in the encoder can be attended to by the decoder, so there is no need for masking cross-attention. For example, in the case of the math seq2seq task, the when predicting the $i$-th token in the solution (output sequence) the model can attend to all information in the question (input sequence), as well as to the portion of the solution (output sequence) generated so far (i.e., up until token $i$).

Please let us know whether this answers your question.

> On the Math Problem experiments: It seems the dataset consists of 8 tasks, but only 5 of these are investigated in the experiments. Is there a particular reason for this omission?

The math dataset contains 8 "modules", but each module contains several tasks. We did not have the computational resources to run an evaluation on all tasks. The tasks were chosen semi-arbitrarily as a representative set of the full set of tasks.

> On the SET Comparison against a symbolic-input model: When pre-training the abstractor relations, is the input from the same pre-trained CNN as when training the multi-head abstractor, or also from the symbolic inputs?

When pre-pretraining the Abstractor relations for this experiment, the input is the embeddings generated by the pre-trained CNN. By contrast, the MLP receives as input a hard-coded binary representation of the four latent relations. The idea behind this comparison is to evaluate the quality of the geometry of the representations produced by an Abstractor. The hard-coded binary representation of relations is an ``ideal'' representation, in the sense that it is completely disentangled. The difference in learning curves between the MLP with the hard-coded representations and the Abstractor representations is taken as an indication of the quality of representational geometry produced by the Abstractor.

> The authors argue that relations are well-modelled as inner-products. I am curious as to which differences this might impose on the learned relations when compared to the relational-form used in the PrediNet, in which a difference of projections ("differential comparator") is used?

A very interesting question. Here are some thoughts on this based on our understanding of the PrediNet paper.

The PrediNet paper argues for a particular philosophy based on propositional logic about how knowledge and relations ought to be represented. But in terms of representations, the architectures consists of the following steps, 1) retrieve a pair of objects (one pair for each 'head'), 2) project the object representations into a one dimensional space (one for each relation), and 3) take the difference between those scalars. This produces a vector of `n_pairs x n_relations` entries (modulo some additional entries encoding position, since in their paper they consider the input to be an image). At its core, as you said, PrediNet represents a relation as a difference between one-dimensional projections whereas we represent relation as an inner product between two projections onto $d$-dimensional space.

In some sense, differences and inner products are ``equivalent'' in terms of information content. For example, if $x, y$ are unit-length, $\langle x, y \rangle$ is proportional to $\lVert x - y \rVert$^2. A couple of simple differences to begin with. In PrediNet, one relation is a comparison in 1-dimensional space, whereas in the Abstractor, one relation can be a comparison in a multi-dimensional space. Also, the PrediNet architecture computes relations by first retrieving a pair of objects via an attention mechanism, whereas we consider all possible pairs.

One advantage of inner products over differences as representations of relations is greater "invariance to representation". For example, applying an orthogonal transformation to a pair of vectors does not change the inner product, but would change the difference. Moreover, inner products have certain robustness properties that may be useful for learning representations (e.g, for a random matrix $\Phi$ with iid Gaussian entries, $\langle \Phi x, \Phi y\rangle \approx \langle x, y \rangle$; Zhou et al. (2009)). We explore some of this in section C.1 of the appendix, where we present some experiments on robustness to corruptive noise of different forms.

These are some of our initial thoughts on the topic. Indeed, with the addition of PrediNet as a baseline for the experiments in section 4.1, we observe that the Abstractor tends to perform better. But the PrediNet architecture includes some added confounders. It may be interesting to explore the differences between inner product relations and difference relations in a more controlled setting (e.g., by creating a simplest possible architecture with this inductive bias, similar to what CoRelNet aimed to do).