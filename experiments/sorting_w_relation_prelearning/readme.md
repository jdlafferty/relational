# Sorting with relation pre-learning

In this experiment we evaluate the modularity of Abstractors. The question is: can the Abstractor be pre-trained on a sub-task and generalize its learned representations to a larger task. We do this object-sorting (see `object_argsort_autoregressive`).

In particular, the sub-task is to learn the pairwise order relation $\prec$ (i.e.: given a pair of objects $(o_i, o_j)$, predict whether $o_i \prec o_j$), and the full task is to sort sequences of 10 objects (i.e.: return the argsort).

Steps to replicate results:
1) Run `generate_random_object_sorting_tasks.ipynb` to generate the dataset for the object-sorting task.
2) Run `relation_prelearning.ipynb` to train models on pre-training subtask. This saves weights so they can be used for initialization on the full task.
3) Run the following for each `training_mode` in `['end-to-end', '']`:
    ```
    python evaluate_learning_curves.py --training_mode {{training_mode}} --n_epochs 200 --early_stopping True --min_train_size 100 --max_train_size 3000 --train_size_step 100 --num_trials 5 --start_trial 5 --wandb_project_name object_sorting_with_relation_prelearning
    ```
4) Plotting and analyzing learning curves is done in `learning_curve_analysis.ipynb`.

The complete logs of our runs are available to view at: [[W&B link. Temporarily hidden for anonymization]].