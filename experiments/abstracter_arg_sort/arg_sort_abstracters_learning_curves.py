# LEARNING CURVES: ARGSORT SEQUENCE-TO-SEQUENCE ABSTRACTERS
# This notebook generates a set of random objects (each described by a random gaussian vector) with some associated ordering
# It then trains a transformer and an abstracter model on sorting
# The models do 'argsorting', meaning they predict the argsort of the sequnce rather than outputting the sorted sequence itself.

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import argparse

import tensorflow as tf

import sklearn.metrics
from sklearn.model_selection import train_test_split

import sys; sys.path.append('../'); sys.path.append('../..')
import arg_sort_models
import utils

# region SETUP

# parse arguments to script
parser = argparse.ArgumentParser()
parser.add_argument('--vocab_size', default=32, type=int, help='size of set of objects')
parser.add_argument('--dim', default=8, type=int, help='dimension of random object vectors')
parser.add_argument('--seqs_length', default=10, type=int, help='length of sequence to sort')
parser.add_argument('--min_train_size', default=50, type=int, help='minimum training set size')
parser.add_argument('--max_train_size', default=2000, type=int, help='maximum training set size')
parser.add_argument('--train_size_step', default=50, type=int, help='training set step size')
parser.add_argument('--num_trials', default=1, type=int, help='number of trials per training set size')
parser.add_argument('--wandb_project_name', default='card-sorting-abstracters-learning-curves-testing', 
    type=str, help='W&B project name')
args = parser.parse_args()

utils.print_section("SET UP")

print(f'received the following arguments: {args}')

# check if GPU is being used
print(tf.config.list_physical_devices())
assert len(tf.config.list_physical_devices('GPU')) > 0

# set up W&B logging
import wandb
wandb.login()

import logging
logger = logging.getLogger("wandb")
logger.setLevel(logging.ERROR)

wandb_project_name = args.wandb_project_name


def create_callbacks(monitor='loss'):
    callbacks = [
        # tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10, mode='auto', restore_best_weights=True),
#         tf.keras.callbacks.ReduceLROnPlateau( monitor='val_loss', factor=0.1, patience=5, verbose=1, mode='auto'),
        wandb.keras.WandbMetricsLogger(log_freq='epoch'),
        # wandb.keras.WandbModelCheckpoint(filepath='models/model_{epoch:02d}', monitor=monitor, mode='auto', save_freq='epoch')
#         wandb.keras.WandbCallback(
#             monitor=monitor, log_weights=log_weights, log_gradients=log_gradients, save_model=save_model, save_graph=True,
#             training_data=train_ds, validation_data=val_ds,
#             labels=class_names, predictions=64, compute_flops=True)
        ]
    return callbacks

from seq2seq_transformer import masked_loss, masked_accuracy

metrics = [masked_accuracy]

loss = masked_loss
create_opt = lambda : tf.keras.optimizers.Adam()

fit_kwargs = {'epochs': 50, 'batch_size': 128}


#region Dataset
def create_sorting_dataset(vocab_size, dim, seqs_length, n_seqs):

    # generate random features for each object
    objects = np.random.normal(size=(vocab_size, dim))

    # generate random permutations of length `seqs_length` out of `vocab_size`
    seqs = np.array([np.random.choice(range(vocab_size), size=seqs_length, replace=False) for _ in range(n_seqs)])
    
    # remove duplicate seqs (although very unlikely)
    _, unique_seq_idxs = np.unique(seqs, axis=0, return_inverse=True)
    seqs = seqs[unique_seq_idxs]

    # create object sequences
    object_seqs = objects[seqs]
    
    sorted_seqs = np.sort(seqs, axis=1)

    arg_sort = np.argsort(seqs, axis=1)

    
    # add `START_TOKEN` to beginning of sorting 
    START_TOKEN = seqs_length
    start_tokens = np.array([START_TOKEN] * len(arg_sort))[np.newaxis].T
    arg_sort = np.hstack([start_tokens, arg_sort])

    return objects, seqs, sorted_seqs, arg_sort, object_seqs,

vocab_size = args.vocab_size
dim = args.dim
seqs_length = args.seqs_length
START_TOKEN = seqs_length
n_seqs = 10_0000

objects, seqs, sorted_seqs, arg_sort, object_seqs = create_sorting_dataset(vocab_size, dim, seqs_length, n_seqs)

target = arg_sort[:, :-1]
labels = arg_sort[:, 1:]

test_size = 0.2
val_size = 0.1

seqs_train, seqs_test, sorted_seqs_train, sorted_seqs_test, object_seqs_train, object_seqs_test, target_train, target_test, labels_train, labels_test = train_test_split(
    seqs, sorted_seqs, object_seqs, target, labels, test_size=0.2)
seqs_train, seqs_val, sorted_seqs_train, sorted_seqs_val, object_seqs_train, object_seqs_val, target_train, target_val, labels_train, labels_val = train_test_split(
    seqs_train, sorted_seqs_train, object_seqs_train, target_train, labels_train, test_size=val_size/(1-test_size))

source_train, source_val, source_test = object_seqs_train, object_seqs_val, object_seqs_test

# endregion

# region evaluation code
def evaluate_seq2seq_model(model, source_test, target_test, labels_test, start_token=START_TOKEN, print_=False):
    
    n = len(source_test)
    output = np.zeros(shape=(n, (seqs_length+1)), dtype=int)
    output[:,0] = start_token
    for i in range(seqs_length):
        predictions = model((source_test, output[:, :-1]), training=False)
        predictions = predictions[:, i, :]
        predicted_id = tf.argmax(predictions, axis=-1)
        output[:,i+1] = predicted_id

    per_card_acc = (np.mean(output[:,1:] == labels_test))
    acc_per_position = [np.mean(output[:, i+1] == labels_test[:, i]) for i in range(seqs_length)]
    seq_acc = np.mean(np.all(output[:,1:]==labels_test, axis=1))
    masked_acc = masked_accuracy(labels_test, model([source_test, target_test]))

    if print_:
        print('per-card accuracy: %.2f%%' % (100*per_card_acc))
        print('full sequence accuracy: %.2f%%' % (100*seq_acc))
        print('masked_accuracy (teacher-forcing):  %.2f%%' % (100*masked_acc))


    return_dict = {
        'per-card accuracy': per_card_acc, 'full sequence accuracy': seq_acc,
        'masked_accuracy': masked_acc, 'acc_by_position': acc_per_position
        }

    return return_dict

def log_to_wandb(model, evaluation_dict):
    acc_by_position_table = wandb.Table(
        data=[(i, acc) for i, acc in enumerate(evaluation_dict['acc_by_position'])], 
        columns=["position", "per-card accuracy at position"])

    evaluation_dict['acc_by_position'] = wandb.plot.line(
        acc_by_position_table, "position", "per-card accuracy at position",
        title="Per-Card Accuracy By Position")

    wandb.log(evaluation_dict)

max_train_size = args.max_train_size
train_size_step = args.train_size_step
min_train_size = args.min_train_size
train_sizes = np.arange(min_train_size, max_train_size+1, step=train_size_step)

num_trials = args.num_trials # num of trials per train set size

print(f'will evaluate learning curve for `train_sizes` from {min_train_size} to {max_train_size} in increments of {train_size_step}.')
print(f'will run {num_trials} trials for each of the {len(train_sizes)} training set sizes for a total of {num_trials * len(train_sizes)} trials')

# endregion

def evaluate_learning_curves(create_model, group_name, 
    source_train=source_train, target_train=target_train, labels_train=labels_train,
    source_val=source_val, target_val=target_val, labels_val=labels_val,
    source_test=source_test, target_test=target_test, labels_test=labels_test,
    train_sizes=train_sizes, num_trials=num_trials):

    for train_size in tqdm(train_sizes, desc='train size'):

        for trial in trange(num_trials, desc='trial', leave=False):
            run = wandb.init(project=wandb_project_name, group=group_name, name=f'train size = {train_size}; trial = {trial}',
                            config={'train size': train_size, 'trial': trial, 'group': group_name})
            model = create_model()

            X_train = source_train[:train_size], target_train[:train_size]
            y_train = labels_train[:train_size]
            X_val = source_val, target_val
            y_val = labels_val

            history = model.fit(X_train, y_train, validation_data=(X_val, y_val), verbose=0, callbacks=create_callbacks(), **fit_kwargs)

            eval_dict = evaluate_seq2seq_model(model, source_test, target_test, labels_test, print_=False)
            log_to_wandb(model, eval_dict)
            wandb.finish(quiet=True)

            del model

# endregion


# region Argsort Transformer
utils.print_section("ARGSORT TRANSFORMER")


def create_model():
    argsort_transformer = arg_sort_models.ArgsortTransformer(
    num_layers=2, num_heads=2, dff=64, 
    # input_vocab_size=vocab_size+1, target_vocab_size=vocab_size+1, embedding_dim=128)
    input_vocab_size=vocab_size, target_vocab_size=seqs_length+1, embedding_dim=64)

    argsort_transformer.compile(loss=loss, optimizer=create_opt(), metrics=metrics)
    argsort_transformer((source_train[:32], target_train[:32]));

    return argsort_transformer


evaluate_learning_curves(create_model, group_name='argsort_transformer')

# endregion

# region Symbolic Abstracter
utils.print_section("ARGSORT SENSORY-CONNECTED ABSTRACTER")


def create_model():
    argsort_model = arg_sort_models.ArgsortSeq2SeqSensoryConnectedAbstracter(
    num_layers=2, num_heads=2, dff=64, 
    # input_vocab_size=vocab_size+1, target_vocab_size=vocab_size+1, embedding_dim=128)
    input_vocab_size=vocab_size, target_vocab_size=seqs_length+1, embedding_dim=64)

    argsort_model.compile(loss=loss, optimizer=create_opt(), metrics=metrics)
    argsort_model((source_train[:32], target_train[:32]));

    return argsort_model


evaluate_learning_curves(create_model, group_name='argsort_sensory-connected_abstracter')

# endregion
