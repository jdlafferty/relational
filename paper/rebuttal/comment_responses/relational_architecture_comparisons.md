Hello reviewers and AC. Several reviewers have asked about how the Abstractor compares to existing relational architectures on discriminative tasks. We have put together a summary of some experimental results on this. In the paper, we decided to focus on generative tasks because we think this is the more interesting aspect of the Abstractor. When the Abstractor is compared to other relational architectures on discriminative tasks, it can usually do just as well (or better) since it incorporates the lessons learned from that line of work (modeling relations as inner products, separating sensory information from relational information, etc.)

**Pairwise $\prec$**

In section 4.1, we presented a "warm up" experiment on the discriminative task of predicting the order relation $\prec$ between two objects. In the paper, we compared against the standard variant of CorelNet which models relations as symmetric. This was an opportunity to add to the discussion of symmetry as an inductive bias, which was started in the CorelNet paper. Of course, we can also compare the Abstractor to an asymmetric variant of CorelNet.

*The task:* Described in section 4.1 of the paper.

*Experimental details:* The model architectures follow what is described in the supplement under section C.1, with the exception that we now use an asymmetric variant of CorelNet.

We report learning curves below (10 trials, mean ± std of test accuracy for each train size):

|Train Size|5|25|50|100|200|300|400|495|
|-:|:-|:-|:-|:-|:-|:-|:-|:-|
|Abstractor|0.522±0.037|0.520±0.054|0.519±0.066|0.602±0.056|0.745±0.081|0.775±0.071|0.858±0.032|0.886±0.021|
|CorelNet (asymmetric)|0.480±0.019|0.532±0.037|0.526±0.051|0.522±0.047|0.610±0.054|0.661±0.069|0.724±0.050|0.721±0.093|

This modification allows CorelNet to learn something about the $\prec$ relation (whereas the symmetric variant could not model the relation at all). However, the learning curve of the Abstractor remains significantly better. Recall that, as mentioned in the paper, since will have never seen the object pairs in the test set, they would need to generalize via the transitivitiy of the $\prec$ relation.

**SET Classification**

*The task:* given a triplet of images of "SET!" cards, predict whether they form a "set" or not. Please see section 4.5 or e.g., https://en.wikipedia.org/wiki/Set_(card_game) for a description of the SET game.

*Experimental details:* A CNN classifier is trained on the card images to predict the four attributes. An intermediate layer is used as an embedder for all relational models. We compare an Abstractor model to CorelNet. The shared architecture is CNN embedder -> Dense -> {Abstractor or CorelNet} -> Flatten -> Dense -> Prediction. The Abstractor has hyperparamaters: 2 layers, 4-dimensional symmetric relations, tanh relation activation. We tested against the standard version of CorelNet, but found that it did not learn anything. We iterated over the hyperparamters and architecture to improve its performance. We found that removing that softmax activation in CorelNet improved performance a bit. We report learning curves below (10 trials, mean ± std of test accuracy):

|Train Size|500|1000|2000|5000|
|-:|:-|:-|:-|:-|
|Abstractor|0.608±0.024|0.752±0.036|0.940±0.018|0.966±0.012|
|CorelNet|0.517±0.014|0.519±0.010|0.514±0.013|0.532±0.008|
|No-softmax CorelNet|0.543±0.014|0.583±0.020|0.635±0.012|0.683±0.013|

We hypothesize that the Abstractor out-performs CorelNet in the above two experiments due to its ability to model multi-dimensional relations.

**ESBN-type experiments**

During the project, we tested the Abstractor against the tasks described in the ESBN paper. These are fairly simple discriminative tasks for relational learning, based on a set of 100 32x32 greyscale images. Overall, we find that Abstractors achieve the same out-of-domain generalization reported in that paper. For example, for the same/different task, when trained on pairs that use only 5 or 2 of the 100 images, and tested on pairs that use the remaining 95 or 98 images, the Abstractor always achieves 100% test accuracy, matching the performance of ESBN reported in that paper, and out-performing the other models that include Transformers, Neural Turing Machines, LSTMs, PrediNet, Metalearned Neural Memories, and Relation Nets. Given the inductive biases of normalization (ESBN/CorelNet use TCN while we used $\ell_2$ normalization) and modeling relations as inner products, these tasks are very easy to learn for both ESBN/CorelNet and the Abstractor, so we did not carry out systematic experiments as we found other experiments to be more interesting.

-----

The first experiment can be easily added to section 4.1 (it would serve as an additional ablation driving the point home about the importance of asymmetry). We think we can fit the second experiment into the paper as well, or at least add it to the appendix.

Please let us know if you have any further questions or comments.