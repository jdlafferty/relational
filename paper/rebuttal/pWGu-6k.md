> While the authors claim that this is closely motivated by human cognition, it is not clear why it makes sense to learn a different location specific object encoding independent of the sensory information. For example, given a sequence of objects, even if object $i$ and object $j$ are identical, they would have different $s_i$ and $s_j$, which does not make sense to me. Maybe the authors could clarify their stance on this?

While the symbols $s_i$ and $s_j$ are different, the *transformed* abstract symbols $A_i = \frac{1}{2}(s_i+s_j) = A_j$ would be the same in this case. Please see the uploaded Figure that makes this more explicit.

The symbols being independent of the sensory information is how the relational bottleneck is achieved and is a crucial element of the model. The goal of the relational bottleneck which the Abstractor implements is that the output of the Abstractor should depend purely on relations between objects in the input sequence but not on the sensory values of any individual objects.

This seems to be a point that we failed to communicate clearly. We will make every effort to clarify this in the final version of the paper.

> Further, by disassociating the sensory information from the retrieval pipeline, it is not clear how useful the model would be when sensory information plays a vital role for reasoning and prediction.

The question of how the Abstractor framework, and relational architectures more generally, perform on tasks which 
rely on sensory information as well as relational information is an important one. This is discussed in the paper 
when we refer to “partially-relational tasks” and the “sensory-connected Abstractor” architecture. Indeed, many 
interesting tasks rely crucially on both sensory information and relational information. In such cases, the 
Abstractor framework enables implementing a relational bottleneck in one branch of the model which processes 
relational information, reaping the benefits of this inductive bias, before merging back into the sensory branch of 
the model. Transformers are powerful sequence models, but they mix relational information and sensory information. 
The Abstractor enables transformer-based models (e.g., a sensory-connected abstractor) to reap the benefits of 
useful inductive biases for learning to represent and process relational information without losing the power of 
transformers in processing sensory information.

This is in fact one of the main advantages of the Abstractor over existing relational architectures. We will revise the paper to better explain this point.

The focus of the experiments is on purely relational tasks because improvements in processing relational information is the main contribution of the Abstractor over standard Transformer-based models. Our work validates that the framework achieves improved sample-efficiency on relational tasks. We agree that experiments evaluating "mixed" architectures like the "sensory-connected" architecture we propose will be an important direction for future work.

> In line with the above problem, it also hinders the ability of the model to generalize and reason beyond the input sequence length seen as the abstract symbols are closely tied with their positions. This is a big limitation and vastly reduces the applicability of the system when compared to standard transformer models.

The abstract symbols being tied to their positions (rather than the values of the objects they reference) is a crucial part of the architecture and is what enables the implementation of the relational bottleneck. However, as with standard transformers, a fixed (rather than learned) coding scheme could be used (e.g., sinusoidal position codes), which would in principle allow the same degree of generalization to longer sequences. In practice, this approach would not be likely to generalize reliably, just as with standard transformers (see for example “The Devil is in the Detail: Simple Tricks Improve Systematic Generalization of Transformers” by Csordas et. al., and the references therein). To address this, the same approaches used to improve generalization in transformers could be employed, such as using *relative* position encodings. In the context of symbolic message-passing, this would amount to using *position-relative* symbols. That is,

$$
a_i = \sum_j R[i,j] s_{j-i},
$$

where the learned parameters are now $S = (s_{-m+1}, \ldots, s_{-1}, s_0, s_1, \ldots, s_{m-1}) \in {\mathbb R}^{d_s \times (2m - 1)}$.  This results in $a_i$ representing information about object $i$’s relations with the other objects in the sequence in a ‘centered’  coordinate system. Thus, abstractors have the same strengths and limitations as standard transformers with respect to generalization to longer sequences, and the same approaches can be employed to remedy this issue.
[jdc]: <It seems like we might be able to say something even stronger here: that the relational 
bottleneck of the Abstractor may provide an inductive bias that allows it to learn such relative position codes more 
effectively than standard architectures>

> While the authors encourage the use of asymmetric relationships by using different parameters for queries and keys, it would be nice to get a comparison with CoRelNet and other approaches on tasks that actually rely on symmetric relationships. This is important to understand whether the proposed method is still able to decently model such a setting, or just fails to generalize OoD here.

The Abstractor is able to model symmetric relations by simply constraining the left-encoder and right-encoder to be 
equal. And CorelNet is able to model asymmetric relations by allowing for two different encoders. Whether relations 
are modeled as symmetric or asymmetric is not a fundamental limitation of either architecture. Section 4.1 is 
intended as a brief comment adding to the discussion on one of the findings of the CorelNet paper—that having 
symmetric relations is a good inductive bias. This was certainly the case in the tasks considered in that paper. 
Section 4.1 is making the point that the symmetry constraint is not always a good inductive bias and that many 
interesting relations are non-symmetric. A general relational architecture ought to be able to model asymmetric 
relations as well, which is what we test in this paper.
The Abstractor is of course able to model symmetric relations by simply constraining the left-encoder and right-encoder to be equal. And CorelNet is able to model asymmetric relations by allowing for two different encoders. Whether relations are modeled as symmetric or asymmetric is not a fundamental limitation of either architecture. Section 4.1 is intended as a brief comment adding to the discussion on one of the findings of the CorelNet paper—that having symmetric relations can be a good inductive bias. This was certainly the case in the tasks considered in that paper. Section 4.1 is making the point that the symmetry constraint is not always a good inductive bias and that many interesting relations are non-symmetric. A general relational architecture ought to be able to model asymmetric relations as well.