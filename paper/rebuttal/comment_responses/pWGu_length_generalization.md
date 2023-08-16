We thank reviewer pWGu for raising their concerns about length-generalization. We think that length-generalization is an interesting challenge in Transformer-based models, including the Abstractor framework. We believe there are several crucial points to keep in mind regarding this.

1. The context of the literature on length-generalization for transformers.
2. The Abstractor's ability/limitations to generalize to longer lengths compared to transformers.
3. The appropriateness of length generalization as a criterion for evaluating our work.

**Length generalization in Transformers**

The reviewer claims that length-generalization is the "biggest benifit" of Transformer models. It is important to keep in mind the context of the literature on the Transformer's ability to generalize to longer sequences. We commented on this in our previous response. We can elaborate on this here. Another recent paper which explains the context of research into length-generalization in transformers is "The Impact of Positional Encodings on Length Generalization in Transformers" by Kazemnejad et al. The standard way that Transformers can be applied to sequences of arbitrary length is by using non-parameteric absolute position embeddings (the original Vaswani et al. paper proposes sinusoidal embeddings). However, another option that is sometimes used in Transformers is *learned* absolute position embeddings. This is used in GPT3 (Brown et al. 2020) and OPT (Zhang et al. 2022), for example.

While using non-parametric position embeddings enables the model to be applied to sequences of arbitrary length in principle, in practice this does not work well. We quote the above mentioned paper to provide some context.

> The ability to generalize from smaller training context sizes to larger ones, commonly known as length generalization, is a major challenge for Transformer-based language models (Vaswani et al., 2017; Deletang et al., 2023; Zhang et al., 2023). Even with larger Transformers, this issue persists (Brown et al., 2020; Furrer et al., 2020).

> the original Transformer architecture (Vaswani et al., 2017) used non-parametric periodic functions to represent absolute position embeddings (APE) in a systematic manner, but further studies have shown that these functions are inadequate for length generalization (Ontanon et al., 2022).

Thus, it is important to keep in mind that non-parametric position embeddings do not solve the problem of length generalization. Length generalization remains an unsolved problem and is an active area of research.

**The Abstractor has the same abilities/limitations with regard to length generalizaztion as transformers**

There is nothing fundamental about the Abstractor that incurs additional limitations with respect to length-generalization compared to Transformers. In particular, while we present the symbols as being learned parameters, they can also be non-parametric position embeddings. As we mentioned in one of our other responses, the theory shows that this would not reduce the representational power of the Abstractor--as long as the symbols are well-separated and full rank, the function class remains the same (see section B.2 of the supplement).

In the experiments we report in the paper, the symbols are learned. But we also ran experiments in which the symbols were initialized randomly and fixed. We indeed observe that this does not hurt performance. Another option is to fix the the symbols to be sinusoidal positional embeddings. We have also tried this and found it to work as well (such symbols are also well-separated). We would be happy to add these experiments to the appendix. We did not experiment with whether this allows the model to directly generalize to longer sequences.

Thus, when using non-parametric absolute position embeddings, the Abstractor has the same abilities and limitations with regard to length-generalization as standard transformers. Moreover, many of the same approaches used in Transformers to tackle length-generalization can be used in Abstractors. For example, one approach which helps with generalization to longer sequences is using *relative* positional embeddings. In the context of symbolic message-passing, this would amount to using position-relative symbols. That is,
$$a_i \gets \sum_j R[i,j] s_{j-i}$$

where $S = (s_{-m+1}, \ldots, s_{m-1})$ are the position-relative symbols (which can be either learned or non-parametric).

We think that position-relative symbols are an interesting variant of the Abstractor for these and other reasons, although they incur a computatational cost. We will add a brief discussion to the paper about the challenge of generalization to longer sequences and the position-relative symbols variant of the Abstractor.

**Length generalization as a criterion for evaluating this work**

While length generalization is an interesting problem in its own right, it remains an ongoing area of research for Transformers, and it is simply not the focus of our work. We believe it would be inappropriate to dismiss our work based solely on this criterion. Our work proposes a relational architecture, and it should be evaluated on that basis. Moreover, the Abstractor does not have any added fundamental limitations towards length-generealization compared Transformers since all the same techniques which tackle length-generalization can be readily incorporated into the Abstractor architecture (e.g., non-parametric symbols, position-relative symbols, etc.). We do believe that length-generalization is an interesting problem, and we will add some discussion on it to the final version of the paper, including the comments we have made here. However, length-generalization is an unsolved problem in Transformers and solving it is simply not the focus of this work. A more systematic evaluation of how different position embedding schemes in the symbols affect the Abstractor's ability to generalize to longer sequences would be an interesting avenue for future work.