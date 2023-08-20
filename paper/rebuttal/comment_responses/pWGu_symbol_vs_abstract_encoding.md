Thank you for elaborating on your concern and for the question.

We believe the theory on function classes should help clarify some of the confusion. The explanation in the global response may also help. We will try to explain this here and hope the reviewers asks follow-up questions if this remains unclear.

The symbols $s_i$ should be thought of as *references* to objects. They do not contain any information about the contents/values of the objects, but rather refer to objects via their position. This is why we called them 'symbols'. They are a symbol in the same sense that $x$ is a symbol in an equation like $y = x^2$---they reference to an unspecified value.

Letting $X = (x_1, \ldots, x_m)$ be the sequence of objects, and $S = (s_1, \ldots, s_m)$ the symbols, relational cross-attention takes the form

$$\mathrm{RelationalCrossAttn}(X) = \mathrm{Attention}(Q \gets X, K \gets X, V \gets S)$$

By having the values be the input-independent symbols, the result is that the abstract encodings $A = (A_1, \ldots, A_m) \gets \mathrm{RelationalCrossAttn}(X)$ represent purely relational information and do not represent any information about the values of individual objects. This is the relational bottleneck.

The theory on function classes makes this formal. The abstract encodings $A = (A_1, \ldots, A_m)$ represent information about the relation tensor,
$$
R = \sigma(\phi_1(X)^\top \phi_2(X)),
$$

where $\phi_1$ is the left-encoder, $\phi_2$ is the right encoder, and $\sigma$ is the relation activation function. In particular, for a one layer Abstractor, $A_i$ is a representation of the components of the relation tensor involving object $i$. That is, $A_i$ represents object $i$'s relations with each other object.

This may be part of the confusion.

> Over training, the model will see various different sensory objects which would all correspond to the same abstract encoding that is learned since this is only position-dependent.

The different objects occuring in the same position will have the same symbol $s_i$, but they will not have the same abstract encoding $A_i$. The abstract encoding $A_i$ will be a function of the *relations* involving object $i$.

The Abstractor outputs a sequence of abstract encodings $A = (A_1, \ldots, A_m)$ which represent the relations between the input objects. The architecture is followed by a decoder. The decoder cross-attends to those abstract encodings, retrieving the relational information which is relevant to the decoding process. In a purely relational architecture, the decoder would cross-attend only to the abstract encodings $A$. In a "sensory-connected" architecture, the decoder attends to both the Abstractor and the Encoder. Cross attending to the encoder provides a connection back to the sensory information.

The choice of architecture should depend on the task. For example, the argsort task is purely relational, so we used a purely relational architecture in the experiments. In general, most "real-world" tasks would be partially-relational. Here, the hypothesis is that the Abstractor and the relational bottleneck it implements in one branch of the model enable enhanced relational reasoning on the relational component of the task. The sensory information remains available in another branch of the model.

Please let us know if this has clarified things or if you have any remaining questions or concerns.