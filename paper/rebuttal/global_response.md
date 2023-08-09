

We sincerely thank the reviewers for their constructive comments, which will help us to significantly improve the paper. We found several misunderstandings in the reviews, which are no doubt due to our failure to communicate the key ideas in a clear manner. Below we make explicit what the contributions of the paper are, and how the paper can be modified to make these contributions clear to readers.

# Summary of contributions

The primary contribution of our work is a modification of transformers that is simple to implement, but has profound consequences for generalization and sample efficiency. The most important distinguishing feature of the Abstractor is that information processing is isolated from the sensory inputs by a ‘relational bottleneck’. This is accomplished by computing a relation tensor from the sensory inputs (treating them as the queries and keys in an attention operation), and then using this relation tensor to parameterize attention over a set of a standalone ‘symbols’ (treating these as the values in an attention operation). This has the consequence that symbols are only influenced by the relations between sensory inputs, not their specific content. This allows the Abstractor to learn relational tasks much faster than a standard transformer, and to generalize learned relations out-of-distribution. At the same time, the Abstractor maintains the strengths of the transformer framework, including the capacity to perform generative tasks, and the ease of modeling long-range dependencies between inputs.

# A clearer description of the architecture

Relational cross-attention is to an Abstractor what self-attention is to a Transformer—it is the core operation. In a transformer, self-attention has an interpretation as message-passing. Similarly, “symbolic relational message-passing” is the name we give the message-passing interpretation of the core operation in an Abstractor.

Below are the equations for self-attention and relational cross-attention.
$$
\begin{align*}&\text{Self-Attention}: \quad X \mathrm{softmax}(\phi_1(X)^\top \phi_2(X))\\ &\text{Relational Cross-Attention:} \quad S \sigma(\phi_1(X)^\top \phi_2(X)) \end{align*}
$$

where $\phi_1, \phi_2$ are the left and right encoders, $S$ are *learned input-independent parameters* (which we call ‘symbols’) and $\sigma$ is the activation function of the relation matrix. For both attention mechanisms, we can think of $\phi_1(X)^\top \phi_2(X)$ as a relation matrix (tensor, if multi-head) where the $(i,j)$ entry is the relation between $x_i$ and $x_j$.

If the activation function of the relation matrix/tensor is chosen to be $\sigma = \mathrm{softmax}$ (which is one specialization of the Abstractor), then this corresponds to

$$
\begin{align*}&\text{Self-Attention}: \quad \mathrm{Attention}(Q \gets X, K \gets X, V \gets X)\\ &\text{Relational Cross-Attention:} \quad \mathrm{Attention}(Q \gets X, K \gets X, V \gets S)\end{align*}
$$

Hence, the key difference is that the values of attention are input-independent learned symbols $S$. 

The abstract symbols $A$ computed using relational cross-attention represent information of the form “object $i$ has relation $R_{ij}$ with object $j$”, where $R = \sigma(\phi(X)^\top \phi(X))$ is the relation tensor. To see why this is the case, consider the expanded form of the attention (relational message-passing operation):

$$
a_i \gets \sum_{j} R[i,j] s_j.
$$

Here, the learned parameters $s_1, \ldots, s_n \in {\mathbb R}^{d_s}$ are called symbols because they act as abstract references to objects in the input sequence based on their position. To make this explicit, suppose that the symbols are simply the canonical basis vectors, $s_i = \mathbf{e}_i$ (i.e., 1 in position $i$ and 0 everywhere else). Then, the updated representation at the $i$-th position is exactly the vector of relations with each object in the input sequence:

$$
a_i = \begin{pmatrix}R[i,1] \\ \vdots \\ R[i,n]\end{pmatrix}
$$

The fact that the transformed symbols are purely relational, and do not contain any information about the encodings of individual objects is what we refer to as the relational bottleneck. This is the inductive bias that enables the Abstractor to have improved sample efficiency on relational tasks compared to a standard transformer.

# Comparison to other relational architectures

One of the main contributions of the Abstractor is that it proposes a class of *generative* models for relational learning. Existing architectures like CorelNet or PrediNet are purely *discriminative*, and therefore have no mechanism to perform many of the tasks that we investigate. This is done by developing a module which fits into transformer-based models. Our work is inspired by the findings of previous work on relational learning—in particular, that inner products model relations well, and that it is useful to separate the representations of value information from the representations of relational information.

In our main experiments, we don’t compare performance to existing relational architectures like CorelNet because they are unable to perform such generative sequence-to-sequence tasks without major modifications (e.g., the autoregressive sorting task). Instead, we focus our comparison to standard Transformer models. Our results show that, while a Transformer can learn relational tasks given enough data, incorporating an Abstractor results in superior sample efficiency.

An Abstractor can, like a Transformer, also perform discriminative tasks. In this case, comparison to existing relational architectures (like CorelNet) would be possible. But the main innovation of the work is to develop a relational architecture for relational generative/sequence-to-sequence tasks which fits within the powerful transformer framework. This is what we chose to focus our experiments on.  With more space by tightening up the presentation, we will be able to include more baseline experiments on discriminative tasks.

# Comment on clarity of presentation

We appreciate the feedback on the lack of clarity in the presentation.  The core idea, which we believe to have significant implications, is simple and should have been described more succinctly. We will revamp the sections describing the architecture, making them more concise and to-the-point. We will do this by immediately presenting the equations of relational cross-attention and describing the significance of the modification, contrasting it to standard self-attention, and explaining how it gives rise to the relational bottleneck. The message-passing interpretation will be moved to later or eliminated. (The message-passing formulation is useful for the statement of the theory which is presented mainly in the supplement, but it is clear that this distracted from the core idea in the main paper.)
