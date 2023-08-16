We thank reviewer pWGu for raising their concerns about length-generalization. As we explain below, the Abstractor has the same strengths and limitations as the Transformer with regards to length generalization. Effective generalization to longer sequences is a challenging problem for Transformers, but a number of methods have been proposed to address this. All of these methods can be readily applied to Abstractors.

**Length generalization in Transformers**

The standard way that Transformers can be applied to sequences of arbitrary length is by using non-parameteric absolute position embeddings (the original Vaswani et al. paper proposes sinusoidal embeddings). This is in contrast to the use of *learned* absolute position embeddings, which are used, for instance, in GPT3 (Brown et al. 2020) and OPT (Zhang et al. 2022).

While using non-parametric position embeddings enables the model to be applied to sequences of arbitrary length in principle, in practice this does not work well. This issue is investigated in the paper "The Impact of Positional Encodings on Length Generalization in Transformers" by Kazemnejad et al.:

> The ability to generalize from smaller training context sizes to larger ones, commonly known as length generalization, is a major challenge for Transformer-based language models (Vaswani et al., 2017; Deletang et al., 2023; Zhang et al., 2023). Even with larger Transformers, this issue persists (Brown et al., 2020; Furrer et al., 2020).

> the original Transformer architecture (Vaswani et al., 2017) used non-parametric periodic functions to represent absolute position embeddings (APE) in a systematic manner, but further studies have shown that these functions are inadequate for length generalization (Ontanon et al., 2022).

Thus, though non-parametric position embeddings can in principle be extended to longer sequences than those observed during training, this approach does not work very well in practice. Length generalization remains an unsolved problem and is an active area of research.

**The Abstractor has the same abilities/limitations with regard to length generalization as Transformers**

There is nothing fundamental about the Abstractor that incurs additional limitations with respect to length-generalization compared to Transformers. In particular, while we present the symbols as being learned parameters, they can also be non-parametric position embeddings. As we mentioned in one of our other responses, the theory shows that this would not reduce the representational power of the Abstractor--as long as the symbols are well-separated and full rank, the function class remains the same (see section B.2 of the supplement).

In the experiments we report in the paper, the symbols are learned. But we also ran experiments in which the symbols were initialized randomly and fixed. We indeed observe that this does not hurt performance. Another option is to fix the the symbols to be sinusoidal positional embeddings. We have also tried this and found it to work as well (such symbols are also well-separated). We would be happy to add these experiments to the appendix. We did not run experiment on whether this allows the model to directly generalize to longer sequences.

Thus, when using non-parametric absolute position embeddings, the Abstractor has the same abilities and limitations with regard to length-generalization as standard transformers. Moreover, many of the same approaches used in Transformers to tackle length-generalization can be used in Abstractors. For example, one approach which helps with generalization to longer sequences is using *relative* positional embeddings. In the context of symbolic message-passing, this would amount to using position-relative symbols. That is,
$$a_i \gets \sum_j R[i,j] s_{j-i}$$

where $S = (s_{-m+1}, \ldots, s_{m-1})$ are the position-relative symbols (which can be either learned or non-parametric). This incurs a computational cost, however.

----

While length generalization is an interesting problem in its own right, it remains an ongoing area of research for Transformers, and it is not the focus of our work. Moreover, the Abstractor does not have any added fundamental limitations towards length-generealization compared Transformers since all the same techniques which tackle length-generalization can be readily incorporated into the Abstractor architecture (e.g., non-parametric symbols, position-relative symbols, etc.). We will add some discussion on length-generalization in the Abstractor framework to the final version of the paper, including the comments we have made here.
