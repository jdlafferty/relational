# Experiments

This directory contains code for reproducing the experiments in the paper. It contians the following experiments:

- `pairwise_order`: learn an asymmetric $\prec$ relation on randomly-generated objects.
- `object_argsort_autoregressive`: sort sequences of objects according to an underlying $\prec$ relation which needs to be learned end-to-end.
- `sorting_w_relation_prelearning`: sort sequences of randomly-generated objects by modularly learning the relation on a pair-wise subtask.
- `robustness_object_sorting`: evaluates robustness to different kinds of corruption of the object representations.
- ... [[TODO: add the rest]]

Each directory contains a `readme` which describes the experiment in more detail and contains instructions for how to reproduce the results reported in the paper.

For all experiments, you can replicate our python environment by using the `conda_environment.yml` file, via:
```
conda env create -f conda_environment.yml
```