# LEARNING CURVES AND ABSTRACTER GENERALIZATION: RANDOM OBJECT SORTING WITH ABSTRACTORS
# We generate random objects (as gaussian vectors) and associate an ordering to them.
# We train abstracter models to learn how to sort these objects
# To test the generalization of abstracters, we first train one on another object-sorting task, 
# then fix the abstracter module's weights and re-train the encoder
# The models do 'argsorting', meaning they predict the argsort of the sequnce.

# NOTE: this is 'one-shot' in the sense that we predict the entire argsort of the sequence 
# in one call of the model; not autoregressively as is typical with seq2seq models

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import argparse
import os

import tensorflow as tf

import sklearn.metrics
from sklearn.model_selection import train_test_split

import sys; sys.path.append('../'); sys.path.append('../..')
import models
import utils

# region SETUP

seed = 314159

# parse script arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, choices=('transformer', 'rel-abstracter'),
    help='the model to evaluate learning curves on')
parser.add_argument('--pretraining_mode', default='none', type=str,
    choices=('none', 'pretraining'),
    help='whether and how to pre-train on pre-training task')
parser.add_argument('--pretraining_task_data_path', default='object_sorting_datasets/task1_object_sort_dataset.npy', 
    type=str, help='path to npy file containing sorting task dataset')
parser.add_argument('--eval_task_data_path', default='object_sorting_datasets/task2_object_sort_dataset.npy', 
    type=str, help='path to npy file containing sorting task dataset')
parser.add_argument('--pretraining_train_size', default=10_000, type=int,
    help='training set size for pre-training (only used for pre-training tasks)')
parser.add_argument('--n_epochs', default=200, type=int, help='number of epochs to train each model for')
parser.add_argument('--early_stopping', default=True, type=bool, help='whether to use early stopping')
parser.add_argument('--min_train_size', default=500, type=int, help='minimum training set size')
parser.add_argument('--max_train_size', default=5000, type=int, help='maximum training set size')
parser.add_argument('--train_size_step', default=50, type=int, help='training set step size')
parser.add_argument('--num_trials', default=1, type=int, help='number of trials per training set size')
parser.add_argument('--start_trial', default=0, type=int, help='what to call first trial')
parser.add_argument('--wandb_project_name', default='abstracter_argsort_generalization', 
    type=str, help='W&B project name')
args = parser.parse_args()

utils.print_section("SET UP")

print(f'received the following arguments: {args}')

# check if GPU is being used
print(tf.config.list_physical_devices())
assert len(tf.config.list_physical_devices('GPU')) > 0

# set up W&B logging
os.environ["WANDB_SILENT"] = "true"
import wandb
wandb.login()

import logging
logger = logging.getLogger("wandb")
logger.setLevel(logging.ERROR)

wandb_project_name = args.wandb_project_name


def create_callbacks(monitor='loss'):
    callbacks = [
#         tf.keras.callbacks.ReduceLROnPlateau( monitor='val_loss', factor=0.1, patience=5, verbose=1, mode='auto'),
        wandb.keras.WandbMetricsLogger(log_freq='epoch'),
        ]

    if args.early_stopping:
        callbacks.append(tf.keras.callbacks.EarlyStopping(monitor='loss', patience=20, mode='auto', restore_best_weights=True))

    return callbacks

metrics = [tf.keras.metrics.sparse_categorical_accuracy]


loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, ignore_class=None, name='sparse_categorical_crossentropy')
create_opt = lambda : tf.keras.optimizers.Adam()

fit_kwargs = {'epochs': args.n_epochs, 'batch_size': 128}

#region Dataset

eval_task_data = np.load(args.eval_task_data_path, allow_pickle=True).item()

objects, seqs, sorted_seqs, object_seqs, argsort_labels = (eval_task_data['objects'], eval_task_data['seqs'], \
    eval_task_data['sorted_seqs'], eval_task_data['object_seqs'], eval_task_data['labels'])

test_size = 0.2
val_size = 0.1

seqs_train, seqs_test, sorted_seqs_train, sorted_seqs_test, object_seqs_train, object_seqs_test, labels_train, labels_test = train_test_split(
    seqs, sorted_seqs, object_seqs, argsort_labels, test_size=test_size, random_state=seed)
seqs_train, seqs_val, sorted_seqs_train, sorted_seqs_val, object_seqs_train, object_seqs_val, labels_train, labels_val = train_test_split(
    seqs_train, sorted_seqs_train, object_seqs_train, labels_train, test_size=val_size/(1-test_size), random_state=seed)

_, seqs_length, object_dim = object_seqs.shape

source_train, source_val, source_test = object_seqs_train, object_seqs_val, object_seqs_test
#endregion

# region kwargs for all the models
common_args = dict(
    num_layers=4, num_heads=8, dff=64, 
    output_dim=seqs_length, embedding_dim=64, 
    dropout_rate=0.1)
transformer_kwargs = rel_abstractor_kwargs = common_args

# endregion

# region evaluation code
def evaluate_argsort_model(model, source_test, labels_test, print_=False):
    
    preds = model(source_test)
    preds = np.argmax(preds, axis=-1)

    seq_acc = np.mean(np.all(preds == labels_test, axis=-1))
    elementwise_acc = np.mean(preds==labels_test)

    acc_per_position = [np.mean(preds[:, i] == labels_test[:, i]) for i in range(seqs_length)]
    
    if print_:
        print('element-wise accuracy: %.2f%%' % (100*elementwise_acc))
        print('full sequence accuracy: %.2f%%' % (100*seq_acc))


    return_dict = {
        'elementwise_accuracy': elementwise_acc,
        'full_sequence_accuracy': seq_acc,
        'acc_by_position': acc_per_position
        }

    return return_dict

def log_to_wandb(model, evaluation_dict):
    acc_by_position_table = wandb.Table(
        data=[(i, acc) for i, acc in enumerate(evaluation_dict['acc_by_position'])], 
        columns=["position", "element-wise accuracy at position"])

    evaluation_dict['acc_by_position'] = wandb.plot.line(
        acc_by_position_table, "position", "element-wise accuracy at position",
        title="Element-wise Accuracy By Position")

    wandb.log(evaluation_dict)

max_train_size = args.max_train_size
train_size_step = args.train_size_step
min_train_size = args.min_train_size
train_sizes = np.arange(min_train_size, max_train_size+1, step=train_size_step)

num_trials = args.num_trials # num of trials per train set size
start_trial = args.start_trial

print(f'will evaluate learning curve for `train_sizes` from {min_train_size} to {max_train_size} in increments of {train_size_step}.')
print(f'will run {num_trials} trials for each of the {len(train_sizes)} training set sizes for a total of {num_trials * len(train_sizes)} trials')

def evaluate_learning_curves(create_model, group_name, 
    source_train=source_train, labels_train=labels_train,
    source_val=source_val, labels_val=labels_val,
    source_test=source_test, labels_test=labels_test,
    train_sizes=train_sizes, num_trials=num_trials):

    for train_size in tqdm(train_sizes, desc='train size'):

        for trial in trange(start_trial, start_trial + num_trials, desc='trial', leave=False):
            run = wandb.init(project=wandb_project_name, group=group_name, name=f'train size = {train_size}; trial = {trial}',
                            config={'train size': train_size, 'trial': trial, 'group': group_name})
            model = create_model()

            X_train = source_train[:train_size]
            y_train = labels_train[:train_size]
            X_val = source_val
            y_val = labels_val

            history = model.fit(X_train, y_train, validation_data=(X_val, y_val), verbose=0, callbacks=create_callbacks(), **fit_kwargs)

            # if fitting pre-trained model, unfreeze all weights and re-train after initial training
            if 'pretraining' in args.pretraining_mode:
                stage1_epochs = max(history.epoch)
                fit_kwargs_ = {'epochs': fit_kwargs['epochs'] + stage1_epochs,
                'batch_size': fit_kwargs['batch_size'], 'initial_epoch': max(history.epoch)}
                for layer in model.layers: # unfreeze all layers and continue training
                    layer.trainable = True
                history = model.fit(X_train, y_train, validation_data=(X_val, y_val), verbose=0, callbacks=create_callbacks(), **fit_kwargs_)
                stage2_epochs = max(history.epochs) - stage1_epochs
                wandb.log({'stage1_epochs': stage1_epochs, 'stage2_epochs': stage2_epochs}) # log # of epochs trained in each stage

            eval_dict = evaluate_argsort_model(model, source_test, labels_test, print_=False)
            log_to_wandb(model, eval_dict)
            wandb.finish(quiet=True)

            del model

# endregion


# region define models and model set up code

# pre-training set up
if 'pretraining' in args.pretraining_mode:

    # load pre-training task data
    pretraining_task_data = np.load(args.eval_task_data_path, allow_pickle=True).item()

    object_seqs_pretraining, labels_pretraining = (pretraining_task_data['object_seqs'], pretraining_task_data['labels'])

    test_size = 0.2
    val_size = 0.1

    (object_seqs_train_pretraining, object_seqs_test_pretraining,  
        labels_train_pretraining, labels_test_pretraining) = train_test_split(
        object_seqs_pretraining, labels_pretraining, test_size=test_size, random_state=seed)
    (object_seqs_train_pretraining, object_seqs_val_pretraining, 
        labels_train_pretraining, labels_val_pretraining) = train_test_split(
        object_seqs_pretraining, labels_pretraining, test_size=val_size/(1-test_size), random_state=seed)

    (source_train_pretraining, source_val_pretraining, source_test_pretraining) = (object_seqs_train_pretraining,
        object_seqs_val_pretraining, object_seqs_test_pretraining)

    X_train_pretraining = source_train_pretraining[:args.pretraining_train_size]
    y_train_pretraining = labels_train_pretraining[:args.pretraining_train_size]
    X_val_pretraining = source_val_pretraining[:args.pretraining_train_size]
    y_val_pretraining = labels_val_pretraining[:args.pretraining_train_size]

    

# transformer
if args.model == 'transformer':
    if args.pretraining_mode == 'none':
        def create_model():
            argsort_model = models.create_transformer(seqs_length, object_dim,
                **transformer_kwargs)

            argsort_model.compile(loss=loss, optimizer=create_opt(), metrics=metrics)

            return argsort_model
        
        group_name = 'Transformer'

    # pre-training set up for transformer model
    elif args.pretraining_mode in ['pretraining']:
        
        # fit model on pre-training task
        pretrained_model = models.create_transformer(seqs_length, object_dim, 
                **transformer_kwargs)


        pretrained_model.compile(loss=loss, optimizer=create_opt(), metrics=metrics)

        utils.print_section('Fitting Model on Pre-training Task')
        run = wandb.init(project=wandb_project_name, name=f'pretraining_mode={args.pretraining_mode}',
            group='Pre-Training Task; Transformer', 
            config={
                'train size': len(source_train_pretraining), 
                'group': 'Pre-Training Task; Transformer',
                'pretraining_mode': args.pretraining_mode}
            )
        history = pretrained_model.fit(X_train_pretraining, y_train_pretraining, 
            validation_data=(X_val_pretraining, y_val_pretraining), verbose=0, callbacks=create_callbacks(), **fit_kwargs)
        eval_dict = evaluate_argsort_model(pretrained_model, source_test_pretraining, labels_test_pretraining,
            print_=False)
        log_to_wandb(pretrained_model, eval_dict)
        wandb.finish(quiet=True)

        if args.pretraining_mode == 'pretraining':

            def create_model():
                argsort_model = models.create_transformer(seqs_length, object_dim, 
                    **transformer_kwargs)

                argsort_model.compile(loss=loss, optimizer=create_opt(), metrics=metrics)

                argsort_model.get_layer('encoder').set_weights(pretrained_model.get_layer('encoder').weights)
                argsort_model.get_layer('encoder').trainable = False
                argsort_model.get_layer('source_embedder').set_weights(pretrained_model.get_layer('source_embedder').weights)
                argsort_model.get_layer('source_embedder').trainable = False

                return argsort_model

            group_name = 'Transformer (Pre-Trained)'
        
    else:
        raise ValueError(f'`pretraining_mode` {args.pretraining_mode} is invalid')


# Relational abstracter
elif args.model == 'rel-abstracter':
    # standard evaluation
    if args.pretraining_mode == 'none':
        def create_model():
            argsort_model = models.create_abstractor(seqs_length, object_dim, 
                **rel_abstractor_kwargs)

            argsort_model.compile(loss=loss, optimizer=create_opt(), metrics=metrics)

            return argsort_model
        
        group_name = 'Relational Abstracter'
    
    # if evaluating generalization via pre-training
    elif args.pretraining_mode in ['pretraining']:
        
        # fit model on pre-training task
        pretrained_model = models.create_abstractor(seqs_length, object_dim, 
                **rel_abstractor_kwargs)


        pretrained_model.compile(loss=loss, optimizer=create_opt(), metrics=metrics)

        utils.print_section('Fitting Model on Pre-training Task')
        run = wandb.init(project=wandb_project_name, name=f'pretraining_mode={args.pretraining_mode}',
            group='Pre-training Task, Relational Abstracter', 
            config={
                'train size': len(source_train_pretraining), 
                'group': 'Pre-Training Task; Rel-Abstracter',
                'pretraining_mode': args.pretraining_mode}
            )
        history = pretrained_model.fit(X_train_pretraining, y_train_pretraining, 
            validation_data=(X_val_pretraining, y_val_pretraining), verbose=0, callbacks=create_callbacks(), **fit_kwargs)
        eval_dict = evaluate_argsort_model(pretrained_model, source_test_pretraining, labels_test_pretraining,
            print_=False)
        log_to_wandb(pretrained_model, eval_dict)
        wandb.finish(quiet=True)

        if args.pretraining_mode == 'pretraining':

            def create_model():
                argsort_model = models.create_abstractor(seqs_length, object_dim, 
                    **rel_abstractor_kwargs)

                argsort_model.compile(loss=loss, optimizer=create_opt(), metrics=metrics)

                argsort_model.get_layer('abstractor').set_weights(pretrained_model.get_layer('abstractor').weights)
                argsort_model.get_layer('abstractor').trainable = False
                argsort_model.get_layer('source_embedder').set_weights(pretrained_model.get_layer('source_embedder').weights)
                argsort_model.get_layer('source_embedder').trainable = False

                return argsort_model

            group_name = 'Relational Abstracter (Pre-Trained)'
        
    else:
        raise ValueError(f'`pretraining_mode` {args.pretraining_mode} is invalid')

else:
    raise ValueError(f'`model` argument {args.model} is invalid')


# endregion


# region Evaluate Learning Curves

utils.print_section("EVALUATING LEARNING CURVES")
evaluate_learning_curves(create_model, group_name=group_name)

# endregion
