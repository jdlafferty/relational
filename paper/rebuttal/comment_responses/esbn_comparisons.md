
**Results on ESBN tasks**

We have run multiple experiments on the relational tasks described in the paper "Emergent symbols through binding in external memory" (ESBN). These are fairly simple discriminative tasks for relational learning, based on a set of 100 32x32 black and white images:

*Same/different:* <br>
AA (same) or AB (different)

*Relational match-to-sample:* <br> 
AA BB CD (first and second pairs match)<br>
AB CC DE (first and third pairs match)

*Distribution of three:*<br>
A B C <br>
C B ?  
Possible answers: C A B D (A is correct)

*Identity rules:*<br>
A B A <br>
C A ?  
Possible answers: E C F A (C is correct)

Overall, we find that Abstractors 
handle these discriminative tasks very well, 
with out-of-domain generalization that 
matches the best models presented in that paper. For example, for the same/different task,
when trained on pairs that use only 5 or 2 of the 100 images, and tested on pairs that use the remaining 95 or 98 images, the Abstractor 
always achieves 100% test accuracy, matching the performance of ESBN reported in that paper, and out-performing the other models that include Transformers, Neural Turing Machines, LSTMs, PrediNet, Metalearned Neural Memories, 
and Relation Nets.

*Some implementation details:* The images were processed using a CNN with two convolutional layers, 
each with 32 filters of size 2x2 followed by max-pooling. The resulting feature maps were followed 
by two dense layers and 64 output neurons, with normalization so that the $\ell_2$ norm is one. (We note that the ESBN work uses "temporal context normalization.) The CNN embedder was randomly 
initialized, and not trained, which was sufficient to give separation between these simple images. The Abstractors used symbols of dimension 64.

In fact, we went further than the experiments in the ESBN paper, in several ways. First, 
we modified the data to include scaled images, and replaced the same/different task with the task of learning the asymmetric relation "larger than/smaller than." For the corresponding RMTS task, the Abstractor again achieved 100% test accuracy, when trained on only a small fraction of the images. The original CorelNet is unable to perform this task due to the asymmetric relation. 

Moreover, we considered learning 
curves for the various tasks. In the scaled relational match-to-sample task (RMTS), we learned an Abstractor on larger than/smaller than task, and then used this as a pretrained relation in an Abstractor to perform RMTS, which has 12 input images (see above). This experiment parallels what we reported with pretraining for the sorting task. Our results were as follows:


| Train Size | 5           | 10          | 50          | 100         | 150         | 200         | 250    | 
|-----------:|:------------|:------------|:------------|:------------|:------------|:------------|:-------|
| RMTS Abstractor | 0.51   | 0.82        | 0.91        | 0.97        | 1.0         | 1.0         | 1.0    |


Perhaps the most interesting variant we studied was 
"distribution $n$" as a generalization of the "distribution of 3" task considered in the ESBN task. 
In an instance of "distribution of $n$", the input 
consists of a sequence of $n$ objects, followed by 
the first $n-1$ objects in a permutation of the first $n$.  For example, an instance of distribution of 5 might look like 

A B C D E <br>
E C B A ? <br>
Possible answers: B A F D (correct answer: D)

We trained Abstractors all the way up to "distribution of 20", and trained *only on the first 23 images* (which is the minimum possible). The Abstractor achieved 100% test accuracy on previously unseen images. 

It is instructive (if non-intuitive) to consider how the Abstractor achieves this result. Effectively, the transformed abstract symbol $A_j$ for object $j$ is counting how 
many times the encoded value $E_j$ appears among the 
other encoded values. For answer F above, it appears zero times; for B it appears twice, and for D it appears exactly once. The perceptron that processes the transformed abstract symbols to train the discriminant function is able to easily discriminate 
these values.

