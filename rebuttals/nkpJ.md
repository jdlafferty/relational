Thank you for taking the time to read our work and provide this feedback.

> Could you compare both the performance and sample efficiency of Abstractor with other relational structures like PrediNet?

Thank you for the suggestion! We agree that comparing to PrediNet is an important baseline for the discriminative relational tasks in addition to CoRelNet. We have added this baseline to the paper. On the pairwise order relation task, we observe that while PrediNet performs better than an MLP and the standard symmetric variant of CoRelNet, it struggles compared to the Abstractor and even the asymmetric variant of CoRelNet. On the SET task, however, PrediNet performs better than CoRelNet, although it still lags behind the Abstractor in terms of sample efficiency. This might be explained by the fact that, unlike CoRelNet, PrediNet is able to represent multiple relations simultaneously. The superiority of the Abstractor may be due to inner products being a better inductive bias for modeling relations than the difference comparators used in PrediNet (e.g., inner products obey a stricter relational bottleneck).

> STSN (Mondal et al., 2023) has used Transformer for RAVEN and PGM problems, which involve relational reasoning, how much would the Abstractor improve over Transformer in those tasks?

[[TODO: write a response]]