import os
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd
import os.path
import shlex
import subprocess

from ditto_helper import to_ditto_format, to_jsonl

def active_learning_ditto(task, al_iterations, sampling_size, base_data_path, labeled_set_path, input_path, output_path, learning_model, learning_rate, max_len, batch_size, epochs, balance):

  model = str(task) + '.pt'
  
  test_data = pd.read_csv(base_data_path+task+'_test')
  to_jsonl(base_data_path+task+'_test', input_path+task+'_test.jsonl')

  labeled_set_raw = None

  oracle = pd.read_csv(base_data_path + task + '_train')
  pool_data = pd.read_csv(base_data_path + task + '_train')

  to_jsonl(base_data_path + task + '_train', input_path+task+'_unlabeled_pool.jsonl')

  f1_scores = []
  precision_scores = []
  recall_scores = []
  labeled_set_size = []

  number_labeled_examples = 0

  # Pick a checkpoint, rename it
  cmd = 'mv *_dev.pt checkpoints/%s' % (model)
  os.system(cmd)

  for i in range(1,al_iterations+1):

    print("AL run: " + str(i))

    # Delete all predictions
    cmd = 'rm %s' % (output_path+task+'_prediction.jsonl')
    os.system(cmd)

    # Predict probabilities

    print('Predict probabilities ...')

    cmd = 'CUDA_VISIBLE_DEVICES=0'
    os.system(cmd)

    cmd = """python matcher.py \
      --task %s \
      --input_path %s \
      --output_path %s \
      --lm %s \
      --use_gpu \
      --fp16 \
      --checkpoint_path checkpoints/""" % (task,
      input_path+task+'_unlabeled_pool.jsonl',
      output_path+task+'_prediction.jsonl', learning_model)

    #os.system(cmd)
    # invoke process
    process = subprocess.Popen(shlex.split(cmd),shell=False,stdout=subprocess.PIPE)

    # Poll process.stdout to show stdout live
    while True:
      output = process.stdout.readline()
      if process.poll() is not None:
        break
      if output:
        print(output.strip())
    rc = process.poll()
    print('Return code: ' + str(rc))

    predictions = pd.read_json(output_path+task+'_prediction.jsonl', lines=True)

    predictions_true = predictions[predictions['match']==1]
    predictions_false = predictions[predictions['match']==0]

    # Select k most uncertain pairs based on match_confidence for active learning
    low_conf_pairs_true = predictions_true['match_confidence'].nsmallest(int(sampling_size/2))
    low_conf_pairs_false = predictions_false['match_confidence'].nsmallest(int(sampling_size/2))
    print('low_conf_pairs_true ' + str(low_conf_pairs_true.shape[0]))
    print('low_conf_pairs_false ' + str(low_conf_pairs_false.shape[0]))

    # Label these pairs with oracle and add them to labeled set
    if labeled_set_raw is not None:
        labeled_set_raw = labeled_set_raw.append(oracle[oracle.index.isin(low_conf_pairs_true.index.tolist())])
        labeled_set_raw = labeled_set_raw.append(oracle[oracle.index.isin(low_conf_pairs_false.index.tolist())])
    else:
        labeled_set_raw = oracle[oracle.index.isin(low_conf_pairs_true.index.tolist())]
        labeled_set_raw = labeled_set_raw.append(oracle[oracle.index.isin(low_conf_pairs_false.index.tolist())])

    print('labeled_set_raw ' + str(labeled_set_raw.shape[0]))
    number_labeled_examples += low_conf_pairs_true.shape[0] + low_conf_pairs_false.shape[0]

    #todo: data augmentation

    to_ditto_format(labeled_set_raw, labeled_set_path+task+'_train.txt')

    # Remove labeled pairs from unlabeled pool
    pool_data = pool_data[~pool_data.index.isin(labeled_set_raw.index.tolist())]
    pool_data.to_csv('temp/'+task+'_unlabeled_pool', index=False)
    to_jsonl('temp/'+task+'_unlabeled_pool', input_path+task+'_unlabeled_pool.jsonl')

    # Delete all models
    cmd = 'rm checkpoints/*.pt'
    os.system(cmd)
    cmd = 'rm *.pt'
    os.system(cmd)

    # Train model on labeled set

    print('Train model on labeled set ...')

    cmd = 'CUDA_VISIBLE_DEVICES=0'
    os.system(cmd)

    cmd = """python train_ditto.py \
      --task %s \
      --batch_size %d \
      --max_len %d \
      --lr %s \
      --n_epochs %d \
      --finetuning \
      --lm %s \
      --fp16 \
      --save_model""" % (task, batch_size, max_len, learning_rate, epochs,
      learning_model)
      
    if balance:
        cmd += ' --balance'

    #os.system(cmd)
    # invoke process
    process = subprocess.Popen(shlex.split(cmd),shell=False,stdout=subprocess.PIPE)

    # Poll process.stdout to show stdout live
    while True:
      output = process.stdout.readline()
      if process.poll() is not None:
        break
      if output:
        print(output.strip())
    rc = process.poll()
    print('Return code: ' + str(rc))

    print("Size labeled set " + str(labeled_set_raw.shape[0]))
    print("Size unlabeled pool " + str(pool_data.shape[0]))

    # Delete all predictions
    cmd = 'rm %s' % (output_path+task+'_test_prediction.jsonl')
    os.system(cmd)

    # Pick a checkpoint, rename it
    cmd = 'mv *_dev.pt checkpoints/%s' % (model)
    os.system(cmd)

    # run prediction on test set

    print('Run prediction on test set ...')

    cmd = 'CUDA_VISIBLE_DEVICES=0'
    os.system(cmd)

    cmd = """python matcher.py \
      --task %s \
      --input_path %s \
      --output_path %s \
      --lm %s \
      --use_gpu \
      --fp16 \
      --checkpoint_path checkpoints/""" % (task, input_path+task+'_test.jsonl',
      output_path+task+'_test_prediction.jsonl', learning_model)

    #os.system(cmd)
    # invoke process
    process = subprocess.Popen(shlex.split(cmd),shell=False,stdout=subprocess.PIPE)

    # Poll process.stdout to show stdout live
    while True:
      output = process.stdout.readline()
      if process.poll() is not None:
        break
      if output:
        print(output.strip())
    rc = process.poll()
    print('Return code: ' + str(rc))

    test_predictions = pd.read_json(output_path+task+'_test_prediction.jsonl', lines=True)

    # calculate scores
    prec, recall, fscore, _ = precision_recall_fscore_support(
            test_data['label'],
            test_predictions['match'],
            average='binary')
    
    f1_scores.append(round(fscore,3))
    precision_scores.append(round(prec,3))
    recall_scores.append(round(recall,3))
    labeled_set_size.append(number_labeled_examples)

  all_scores = pd.DataFrame(
    {'labeled set size': labeled_set_size,
    'f1': f1_scores,
    'precision': precision_scores,
    'recall': recall_scores
    })
  print(all_scores)

  return all_scores
