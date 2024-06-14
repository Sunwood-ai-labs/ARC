import copy
import json
import os
import pandas as pd
import numpy as np
from pathlib import Path

from arclib.dsl import Task

if os.path.exists('/kaggle'):
    data_path = Path('/kaggle/input/abstraction-and-reasoning-challenge/')
else:
    data_path = Path('./abstraction-and-reasoning-challenge/')

training_path = data_path / 'training'
evaluation_path = data_path / 'evaluation'
test_path = data_path / 'test'
paths = {'train': training_path, 'eval': evaluation_path, 'test': test_path}


def evaluate_predict_func(func):
    train_tasks = get_tasks('train')
    eval_tasks = get_tasks('eval')
    print('Evaluating train')
    train_score, train_string = evaluate_func_on_tasks(func, train_tasks)
    print('Evaluating eval')
    eval_score, eval_string = evaluate_func_on_tasks(func, eval_tasks)

    print('Train score:', train_score, ';', train_string)
    print('Eval score:', eval_score, ';', eval_string)

    return train_score , eval_score


def evaluate_func_on_tasks(func, tasks):
    preds = [func(task) for task in tasks]
    score, string = score_predictions(tasks, preds)
    return score, string


def score_predictions(tasks, predictions):
    n = 0
    tp = 0
    for task, task_preds in zip(tasks, predictions):
        task_tp = score_task_predictions(task, task_preds)
        tp += task_tp
        n += len(task.test_pairs)
        if task_tp > 0:
            print('True prediction:', task.idx)
    return (n - tp)/float(n), str(tp) + '/' + str(n)


def score_task_predictions(task, task_preds, data='test'):
    if data == 'test':
        examples = task.test_pairs
    elif data == 'train':
        examples = task.pairs
    else:
        raise Exception("Data parameter can be 'train' or 'test' ")

    tp = 0
    for example, example_preds in zip(examples, task_preds):
        example_score = score_example_predictions(example, example_preds)
        tp += example_score
    return tp


def score_example_predictions(example, example_preds):
    input_, output = example
    true = output
    if type(example_preds) == np.ndarray:
        example_preds = [example_preds]
    for pred in example_preds:
        if pred.shape == true.shape:
            if (pred == true).all():
                return 1
    return 0


def get_string(pred):
    pred = pred.astype(int)
    str_pred = str([list(row) for row in pred])
    str_pred = str_pred.replace(', ', '')
    str_pred = str_pred.replace('[[', '|')
    str_pred = str_pred.replace('][', '|')
    str_pred = str_pred.replace(']]', '|')
    return str_pred


def get_tasks(dataset='train'):
    path = paths[dataset]
    fns = sorted(os.listdir(path))
    tasks = []
    for idx, fn in enumerate(fns):
        fp = path / fn
        with open(fp, 'r') as f:
            task = Task(json.load(f), idx)
            tasks.append(task)
    return tasks


def submit(predict):
    submission = pd.read_csv(data_path / 'sample_submission.csv', index_col='output_id')
    test_fns = os.listdir(test_path)
    for fn in test_fns:
        fp = test_path / fn
        with open(fp, 'r') as f:
            task = Task(json.load(f))
        output_id = fn.split('.')[-2] + '_0'
        preds = predict(task)
        string_preds = [get_string(pred) for pred in preds[:3]]
        pred = ' '.join(string_preds)
        submission.loc[output_id, 'output'] = pred
    submission.to_csv('submission.csv')


def sets_intersection(sets_list):
    x = sets_list[0]
    for s in sets_list[1:]:
        x = x.intersection(s)
    return x


def sets_union(sets_list):
    x = set()
    for s in sets_list:
        x = x.union(s)
    return x


def arrays_equal(outputs, predicted_outputs):
    for out, pred in zip(outputs, predicted_outputs):
        if (out != pred).any():
            return False
    return True


def change_inputs(task, inputs_, data='train'):
    task_ = copy.deepcopy(task)
    if data == 'train':
        task_.inputs = inputs_
        task_.pairs = [(inp, out) for inp, out in zip(inputs_, task_.outputs)]
    elif data == 'test':
        task_.test_inputs = inputs_
    return task_


def change_outputs(task, outputs_, data='train'):
    task_ = copy.deepcopy(task)
    if data == 'train':
        task_.outputs = outputs_
        task_.pairs = [(inp, out) for inp, out in zip(task_.inputs, task_.outputs)]
    elif data == 'test':
        task_.test_outputs = outputs_
    return task_


def change_inputs_outputs(task, inputs_, outputs_, data='train'):
    if data == 'train':
        task_ = copy.deepcopy(task)
        task_.pairs = [(inp, out) for inp, out in zip(inputs_, outputs_)]
        task_.inputs = inputs_
        task_.outputs = outputs_
    return task_


def apply_array_func(task, array_func, where='both', data='both'):
    # applys function to all task train inputs and train outputs
    if where == 'both':
        inputs_ = [array_func(input_) for input_ in task.inputs]
        outputs_ = [array_func(output) for output in task.outputs]
        task_ = change_inputs_outputs(task, inputs_, outputs_)
        test_inputs_ = [array_func(input_) for input_ in task.test_inputs]
        if task.test_outputs[0]:
            test_outputs_ = [array_func(output) for output in task.test_outputs]
        else:
            test_outputs_ = task.test_outputs
        task_ = change_inputs_outputs(task_, test_inputs_, test_outputs_, data='test')
    elif where == 'inputs':
        inputs_ = [array_func(input_) for input_ in task.inputs]
        task_ = change_inputs(task, inputs_)
        test_inputs_ = [array_func(input_) for input_ in task.test_inputs]
        task_ = change_inputs(task_, test_inputs_,  data='test')
    elif where == 'outputs':
        outputs_ = [array_func(output) for output in task.outputs]
        task_ = change_outputs(task, outputs_)

    return task_