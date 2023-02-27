# LEARNING CURVES AND ABSTRACTER GENERALIZATION: RANDOM OBJECT SORTING WITH SEQUENCE-TO-SEQUENCE ABSTRACTERS
# We generate random objects (as gaussian vectors) and associate an ordering to them.
# We train abstracter models to learn how to sort these objects
# To test the generalization of abstracters, we first train one on another object-sorting task, 
# then fix the abstracter module's weights and train the encoder/decoder
# The models do 'argsorting', meaning they predict the argsort of the sequnce.

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import argparse

import tensorflow as tf

import sklearn.metrics
from sklearn.model_selection import train_test_split

import sys; sys.path.append('../'); sys.path.append('../..')
import seq2seq_abstracter_models
import utils

# region SETUP

seed = 314159

# kwargs for all the models
sc_abstracter_kwargs = dict(
    num_layers=2, num_heads=2, dff=64, 
    input_vocab='vector', target_vocab=seqs_length+1,
    output_dim=seqs_length, embedding_dim=64)
transformer_kwargs = dict(
    num_layers=2, num_heads=2, dff=64, 
    input_vocab='vector', target_vocab=seqs_length+1, 
    output_dim=seqs_length, embedding_dim=64)

# parse script arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, help='the model to evaluate learning curves on')
parser.add_argument('--pretraining_mode', type=str, help='whether and how to pre-train on pre-training task')
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
        callbacks.append(tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10, mode='auto', restore_best_weights=True))

    return callbacks

from seq2seq_transformer import TeacherForcingAccuracy
teacher_forcing_acc_metric = TeacherForcingAccuracy(ignore_class=None)
metrics = [teacher_forcing_acc_metric]


loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, ignore_class=None, name='sparse_categorical_crossentropy')
create_opt = lambda : tf.keras.optimizers.Adam()

fit_kwargs = {'epochs': args.n_epochs, 'batch_size': 128}

#region Dataset

eval_task_data = np.load(args.eval_task_data_path, allow_pickle=True).item()

objects, seqs, sorted_seqs, object_seqs, target, labels, start_token = (eval_task_data['objects'], eval_task_data['seqs'], eval_task_data['sorted_seqs'], eval_task_data['object_seqs'], \
    eval_task_data['target'], eval_task_data['labels'], eval_task_data['start_token'])

test_size = 0.2
val_size = 0.1

seqs_train, seqs_test, sorted_seqs_train, sorted_seqs_test, object_seqs_train, object_seqs_test, target_train, target_test, labels_train, labels_test = train_test_split(
    seqs, sorted_seqs, object_seqs, target, labels, test_size=test_size, random_state=seed)
seqs_train, seqs_val, sorted_seqs_train, sorted_seqs_val, object_seqs_train, object_seqs_val, target_train, target_val, labels_train, labels_val = train_test_split(
    seqs_train, sorted_seqs_train, object_seqs_train, target_train, labels_train, test_size=val_size/(1-test_size), random_state=seed)

seqs_length = seqs.shape[1]

source_train, source_val, source_test = object_seqs_train, object_seqs_val, object_seqs_test

# endregion

# region evaluation code
def evaluate_seq2seq_model(model, source_test, target_test, labels_test, start_token, print_=False):
    
    n = len(source_test)
    output = np.zeros(shape=(n, (seqs_length+1)), dtype=int)
    output[:,0] = start_token
    for i in range(seqs_length):
        predictions = model((source_test, output[:, :-1]), training=False)
        predictions = predictions[:, i, :]
        predicted_id = tf.argmax(predictions, axis=-1)
        output[:,i+1] = predicted_id

    elementwise_acc = (np.mean(output[:,1:] == labels_test))
    acc_per_position = [np.mean(output[:, i+1] == labels_test[:, i]) for i in range(seqs_length)]
    seq_acc = np.mean(np.all(output[:,1:]==labels_test, axis=1))


    teacher_forcing_acc = teacher_forcing_acc_metric(labels_test, model([source_test, target_test]))
    teacher_forcing_acc_metric.reset_state()

    if print_:
        print('element-wise accuracy: %.2f%%' % (100*elementwise_acc))
        print('full sequence accuracy: %.2f%%' % (100*seq_acc))
        print('teacher-forcing accuracy:  %.2f%%' % (100*teacher_forcing_acc))


    return_dict = {
        'elementwise_accuracy': elementwise_acc, 'full_sequence_accuracy': seq_acc,
        'teacher_forcing_accuracy': teacher_forcing_acc, 'acc_by_position': acc_per_position
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
first_trial = args.first_trial

print(f'will evaluate learning curve for `train_sizes` from {min_train_size} to {max_train_size} in increments of {train_size_step}.')
print(f'will run {num_trials} trials for each of the {len(train_sizes)} training set sizes for a total of {num_trials * len(train_sizes)} trials')

def evaluate_learning_curves(create_model, group_name, 
    source_train=source_train, target_train=target_train, labels_train=labels_train,
    source_val=source_val, target_val=target_val, labels_val=labels_val,
    source_test=source_test, target_test=target_test, labels_test=labels_test,
    train_sizes=train_sizes, num_trials=num_trials):

    for train_size in tqdm(train_sizes, desc='train size'):

        for trial in trange(start_trial, start_trial + num_trials, desc='trial', leave=False):
            run = wandb.init(project=wandb_project_name, group=group_name, name=f'train size = {train_size}; trial = {trial}',
                            config={'train size': train_size, 'trial': trial, 'group': group_name})
            model = create_model()

            X_train = source_train[:train_size], target_train[:train_size]
            y_train = labels_train[:train_size]
            X_val = source_val, target_val
            y_val = labels_val

            history = model.fit(X_train, y_train, validation_data=(X_val, y_val), verbose=0, callbacks=create_callbacks(), **fit_kwargs)

            eval_dict = evaluate_seq2seq_model(model, source_test, target_test, labels_test, start_token, print_=False)
            log_to_wandb(model, eval_dict)
            wandb.finish(quiet=True)

            del model

# endregion


# region define models and model set up code

# pre-training set up
if 'pretraining' in args.pretraining_mode:

    # load pre-training task data
    pretraining_task_data = np.load(args.eval_task_data_path, allow_pickle=True).item()

    object_seqs_pretraining, target_pretraining, labels_pretraining, start_token_pretraining = (pretraining_task_data['object_seqs'], \
        pretraining_task_data['target'], pretraining_task_data['labels'], pretraining_task_data['start_token'])

    test_size = 0.2
    val_size = 0.1

    (object_seqs_train_pretraining, object_seqs_test_pretraining, target_train_pretraining, target_test_pretraining, 
        labels_train_pretraining, labels_test_pretraining) = train_test_split(
        object_seqs_pretraining, target_pretraining, labels_pretraining, test_size=test_size, random_state=seed)
    (object_seqs_train_pretraining, object_seqs_val_pretraining, target_train_pretraining, target_val_pretraining, 
        labels_train_pretraining, labels_val_pretraining) = train_test_split(
        object_seqs_pretraining, target_pretraining, labels_pretraining, test_size=val_size/(1-test_size), random_state=seed)

    (source_train_pretraining, source_val_pretraining, source_test_pretraining) = (object_seqs_train_pretraining,
        object_seqs_val_pretraining, object_seqs_test_pretraining)

    X_train = (source_train_pretraining[:args.pretraining_train_size], target_train_pretraining[:args.pretraining_train_size])
    y_train = labels_train_pretraining[:args.pretraining_train_size]
    X_val = (source_val_pretraining[:args.pretraining_train_size], target_val_pretraining[:args.pretraining_train_size])
    y_val = labels_val_pretraining[:args.pretraining_train_size]

    

# transformer
if args.model == 'transformer':
    if args.pretraining_mode == 'none':
        def create_model():
            argsort_model = seq2seq_abstracter_models.Transformer(
                **transformer_kwargs)

            argsort_model.compile(loss=loss, optimizer=create_opt(), metrics=metrics)
            argsort_model((source_train[:32], target_train[:32]));

            return argsort_model
        
        group_name = 'Transformer'
    else:
        raise NotImplementedError('Only pretraining_mode = `none`is supported for transformers')

# sensory-connected abstracter (standard evaluation)
elif args.model == 'sc-abstracter':
    if args.pretraining_mode == 'none':
        def create_model():
            argsort_model = seq2seq_abstracter_models.Seq2SeqSensoryConnectedAbstracter(
                **sc_abstracter_kwargs)

            argsort_model.compile(loss=loss, optimizer=create_opt(), metrics=metrics)
            argsort_model((source_train[:32], target_train[:32]));

            return argsort_model
        
        group_name = 'S-C Abstracter'
    
    # if evaluating generalization via pre-training
    elif args.pretraining_mode in ['pretraining_fix_abstracter', 'pretraining_fix_abstracter_decoder']:
        
        # fit model on pre-training task
        pretrained_model = seq2seq_abstracter_models.Seq2SeqSensoryConnectedAbstracter(
            **sc_abstracter_kwargs)

        pretrained_model.compile(loss=loss, optimizer=create_opt(), metrics=metrics)
        pretrained_model((source_train_pretraining[:32], target_train_pretraining[:32]));

        utils.print_section('Fitting Model on Pre-training Task')
        run = wandb.init(project=wandb_project_name, group='Pre-training Task, S-C Abstracter', 
            config={'train size': len(source_train_pretraining), 'group': 'Pre-Training Task; SC-Abstracter'})
        history = pretrained_model.fit(X_train, y_train, validation_data=(X_val, y_val), verbose=0, callbacks=create_callbacks(), **fit_kwargs)
        eval_dict = evaluate_seq2seq_model(pretrained_model, source_test_pretraining, target_test_pretraining, labels_test_pretraining,
            start_token_pretraining,  print_=False)
        log_to_wandb(pretrained_model, eval_dict)
        wandb.finish(quiet=True)

        if args.pretraining_mode == 'pretraining_fix_abstracter':

            def create_model():
                argsort_model = seq2seq_abstracter_models.Seq2SeqSensoryConnectedAbstracter(
                    **sc_abstracter_kwargs)

                argsort_model.compile(loss=loss, optimizer=create_opt(), metrics=metrics)
                argsort_model((source_train[:32], target_train[:32]));

                argsort_model.abstracter.set_weights(pretrained_model.abstracter.weights)
                argsort_model.abstracter.trainable = False

                return argsort_model

            group_name = 'S-C Abstracter (Pre-Trained/Fixed Abstracter)'
        
        elif args.pretraining_mode == 'pretraining_fix_abstracter_decoder':
            def create_model():
                argsort_model = seq2seq_abstracter_models.Seq2SeqSensoryConnectedAbstracter(
                    **sc_abstracter_kwargs)

                argsort_model.compile(loss=loss, optimizer=create_opt(), metrics=metrics)
                argsort_model((source_train[:32], target_train[:32]));

                argsort_model.abstracter.set_weights(pretrained_model.abstracter.weights)
                argsort_model.abstracter.trainable = False

                argsort_model.decoder.set_weights(pretrained_model.decoder.weights)
                argsort_model.decoder.trainable = False

                return argsort_model

            group_name = 'S-C Abstracter (Pre-Trained/Fixed Abstracter/Decoder)'
    else:
        raise ValueError(f'`pretraining_mode` {args.pretraining_mode} is invalid')

else:
    raise ValueError(f'`model` argument {args.model} is invalid')


# endregion


# region Evaluate Learning Curves

utils.print_section("EVALUATING LEARNING CURVES")
evaluate_learning_curves(create_model, group_name=group_name)

# endregion
