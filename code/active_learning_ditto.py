import os
from numpy import TooHardError, source
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import pandas as pd
import os.path
import shlex
import subprocess
import glob

from ditto_helper import to_ditto_format, to_jsonl

def active_learning_ditto(task, random_sample, al_iterations, sampling_size, train_data_tl, include_tl_data, tl_weights, base_data_path, labeled_set_path, input_path, output_path, data_augmentation, high_conf_to_ls, da_threshold, learning_model, learning_rate, max_len, batch_size, epochs, balance, da, dk, su, verbose, keep_model):

  model = str(task) + '.pt'
  
  test_data = pd.read_csv(base_data_path+task+'_test')
  to_jsonl(base_data_path+task+'_test', input_path+task+'_test.jsonl')
  to_ditto_format(test_data, labeled_set_path+task+'_test.txt')
  to_ditto_format(test_data, input_path+task+'_test.txt')

  labeled_set_raw = random_sample

  #merge train and validation set, since no explicit validation set is used
  train_data = pd.concat(
    [pd.read_csv(base_data_path + task + '_train'),
    pd.read_csv(base_data_path + task + '_validation')]).reset_index(drop=True)

  oracle = train_data.copy()
  pool_data = train_data.copy()

  f1_scores = []
  precision_scores = []
  recall_scores = []
  labeled_set_size = []
  data_augmentation_labels = []
  all_sample_weights = []

  if random_sample is None:
      number_labeled_examples = 0
  else:
      number_labeled_examples = random_sample.shape[0]
      #remove labeled pairs from unlabeled pool
      pool_data = pool_data[~pool_data.index.isin(labeled_set_raw.index.tolist())]

  to_jsonl(pool_data, input_path+task+'_unlabeled_pool.jsonl')
  to_ditto_format(pool_data, input_path+task+'_unlabeled_pool.txt')

  # Save results for first prediction with initialized model

  print('Run prediction on test set ...')

  # Delete all predictions
  cmd = 'rm %s' % (output_path+task+'_test_prediction.jsonl')
  os.system(cmd)

  cmd = 'CUDA_VISIBLE_DEVICES=0'
  os.system(cmd)

  cmd = """python matcher.py \
    --task %s \
    --input_path %s \
    --output_path %s \
    --lm %s \
    --max_len %d \
    --use_gpu \
    --fp16 \
    --checkpoint_path checkpoints/""" % (task, input_path+task+'_test.txt',
    output_path+task+'_test_prediction.jsonl', learning_model, max_len)

  if dk:
    cmd += ' --dk %s' % dk
  if su:
    cmd += ' --summarize'  

  #os.system(cmd)
  # invoke process
  process = subprocess.Popen(shlex.split(cmd),shell=False,stdout=subprocess.PIPE)

  # Poll process.stdout to show stdout live
  while True:
      output = process.stdout.readline()
      if process.poll() is not None:
          break
      if output:
          if verbose:
              print(output.strip())
  rc = process.poll()
  print('Return code: ' + str(rc))

  test_predictions = pd.read_json(output_path+task+'_test_prediction.jsonl', lines=True)

  # calculate scores
  prec, recall, fscore, _ = precision_recall_fscore_support(
          test_data['label'],
          test_predictions['match'],
          average='binary')

  print('f1: ' + str(round(fscore,3)))
  print('precision: ' + str(round(prec,3)))
  print('recall: ' + str(round(recall,3)))

  f1_scores.append(round(fscore,3))
  precision_scores.append(round(prec,3))
  recall_scores.append(round(recall,3))
  labeled_set_size.append(number_labeled_examples)
  all_sample_weights.append([0,0])

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
      --max_len %d \
      --use_gpu \
      --fp16 \
      --checkpoint_path checkpoints/""" % (task,
      input_path+task+'_unlabeled_pool.txt',
      output_path+task+'_prediction.jsonl', learning_model, max_len)
    if dk:
        cmd += ' --dk %s' % dk
    if su:
        cmd += ' --summarize'  
    #os.system(cmd)
    # invoke process
    process = subprocess.Popen(shlex.split(cmd),shell=False,stdout=subprocess.PIPE)

    # Poll process.stdout to show stdout live
    while True:
        output = process.stdout.readline()
        if process.poll() is not None:
            break
        if output:
            if verbose:
                print(output.strip())
    rc = process.poll()
    print('Return code: ' + str(rc))

    predictions = pd.read_json(output_path+task+'_prediction.jsonl', lines=True)
    
    pool_with_predictions = pool_data.copy()
    pool_with_predictions[['match','match_confidence']] = predictions[['match','match_confidence']]

    predictions_true = pool_with_predictions[pool_with_predictions['match']==1]
    predictions_false = pool_with_predictions[pool_with_predictions['match']==0]

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

    labeled_set_raw = labeled_set_raw.drop_duplicates()
    #test
    print('labeled_set_raw ' + str(labeled_set_raw.shape[0]))
    print('index pairs true: ' + str(low_conf_pairs_true.index.tolist()))
    print('index pairs false: ' + str(low_conf_pairs_false.index.tolist()))
    print('index labeled set raw: ' + str(labeled_set_raw.index.tolist()))
    number_labeled_examples += low_conf_pairs_true.shape[0] + low_conf_pairs_false.shape[0]
    
    # TEST: save all files
    #labeled_set_raw.to_csv("labeled_set_" + str(i), index=True)
    #low_conf_pairs_true.to_csv("low_conf_pairs_true_" + str(i), index=True)
    #low_conf_pairs_false.to_csv("low_conf_pairs_false_" + str(i), index=True)
    #predictions.to_csv("predictions_" + str(i), index=True)
    #pool_data.to_csv("pool_data_" + str(i), index=True)
    #oracle.to_csv("oracle_" + str(i), index=True)

    # data augmentation
    if data_augmentation:
        
        if da_threshold == 0:
            # Select k pairs with lowest entropy for data augmentation
            high_conf_pairs_true = predictions_true['match_confidence'].nlargest(int(sampling_size/2))
            high_conf_pairs_false = predictions_false['match_confidence'].nlargest(int(sampling_size/2))
        else:
            # select pairs with high probability for data augmentation
            high_conf_pairs_true = predictions_true[predictions_true['match_confidence']>=da_threshold]
            # select pairs with low probability for data augmentation
            high_conf_pairs_false = predictions_false[predictions_false['match_confidence']>=(da_threshold)]
            print("Data Augmentation True: " + str(high_conf_pairs_true.shape[0]))
            print("Data Augmentation False: " + str(high_conf_pairs_false.shape[0]))
            
        # Use prediction as label #TODO
        data_augmentation_true = pool_data[pool_data.index.isin(high_conf_pairs_true.index.tolist())]
        data_augmentation_true['label'] = 1
        data_augmentation_false = pool_data[pool_data.index.isin(high_conf_pairs_false.index.tolist())]
        data_augmentation_false['label'] = 0

        # Add them to labeled set (based on flag)
        if high_conf_to_ls:
            labeled_set_raw = labeled_set_raw.append([data_augmentation_true,data_augmentation_false])
            labeled_set_temp = labeled_set_raw
        else:
            labeled_set_temp = labeled_set_raw.append([data_augmentation_true,data_augmentation_false])
        
        # Calculate noisy in high confidence examples
        da_examples = pd.concat([data_augmentation_true,data_augmentation_false])
        da_examples = da_examples.merge(oracle[['id','label']],how='left',on='id',suffixes=('', '_oracle'))
        if da_examples.shape[0] > 0:
            tn, fp, fn, tp = confusion_matrix(da_examples['label_oracle'],da_examples['label'],labels=[0,1]).ravel()
        else:
            tn = 0
            fp = 0
            fn = 0
            tp = 0
        data_augmentation_labels.append([tn, fp, fn, tp])
    
    else:
        labeled_set_temp = labeled_set_raw

        y = labeled_set_temp['label']
        try:
            labeled_set_temp, validation_set_temp, _, _ = train_test_split(labeled_set_temp, y,
                                                stratify=y, 
                                                test_size=0.25)
        except ValueError:
            labeled_set_temp, validation_set_temp, _, _ = train_test_split(labeled_set_temp, y,
                                                stratify=None,
                                                test_size=0.25)
        #labeled_set_temp.to_csv('labeled_set', index=False)
        validation_set_temp.to_csv('validation_set', index=False)

    if tl_weights != None:
      if tl_weights == 'calc':
          source_weight = labeled_set_temp.shape[0] / (train_data_tl.shape[0] + labeled_set_temp.shape[0])
          target_weight = train_data_tl.shape[0] / (train_data_tl.shape[0] + labeled_set_temp.shape[0])
      else:
          source_weight = tl_weights[0]
          target_weight = tl_weights[1]

    if include_tl_data:
        size_source = train_data_tl.shape[0]
        size_target = labeled_set_temp.shape[0]
        print("Labeled set size: " + str(labeled_set_temp.shape[0]))
        # TL: if source data is included
        labeled_set_temp = train_data_tl.append(labeled_set_temp)
        print("Size combined set: " + str(labeled_set_temp.shape[0]))

    labeled_set_temp.to_csv('labeled_set', index=False)

    to_ditto_format(labeled_set_temp, labeled_set_path+task+'_train.txt')
    to_ditto_format(validation_set_temp, labeled_set_path+task+'_validation.txt')

    # Remove labeled pairs from unlabeled pool    
    pool_data = pool_data[~pool_data.index.isin(labeled_set_raw.index.tolist())]
    pool_data.to_csv('temp/'+task+'_unlabeled_pool', index=False)
    to_jsonl('temp/'+task+'_unlabeled_pool', input_path+task+'_unlabeled_pool.jsonl')
    to_ditto_format('temp/'+task+'_unlabeled_pool', input_path+task+'_unlabeled_pool.txt')

    # Delete all models
    cmd = 'rm *%s*.pt' % (task)
    os.system(cmd)

    if keep_model == False:
      # Delete all models
      cmd = 'rm checkpoints/%s' % (model)
      os.system(cmd)

    # Delete files from last run
    files_su = glob.glob(labeled_set_path+task+'*su*')
    files_dk = glob.glob(labeled_set_path+task+'*dk*')
    files_to_delete = files_su + files_dk
    files_to_delete = set(files_to_delete)
    for file in files_to_delete:
        os.remove(file)

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
      --run_id %d \
      --save_model""" % (task, batch_size, max_len, learning_rate, epochs,
      learning_model, i)
    if keep_model:
      cmd += ' --bert_path checkpoints/%s' % model
    if da:
      cmd += ' --da %s' % da
    if dk:
      cmd += ' --dk %s' % dk
    if su:
      cmd += ' --summarize'
    if balance:
      cmd += ' --balance'
    if include_tl_data:
      cmd += ' --size_source %d' % size_source
      cmd += ' --size_target %d' % size_target
      cmd += ' --weight_source %f' % source_weight
      cmd += ' --weight_target %f' % target_weight

    #os.system(cmd)
    # invoke process
    process = subprocess.Popen(shlex.split(cmd),shell=False,stdout=subprocess.PIPE)

    # Poll process.stdout to show stdout live
    while True:
        output = process.stdout.readline()
        if process.poll() is not None:
            break
        if output:
            if verbose:
                print(output.strip())
    rc = process.poll()
    print('Return code: ' + str(rc))

    print("Size labeled set " + str(labeled_set_raw.shape[0]))
    print("Size unlabeled pool " + str(pool_data.shape[0]))
    print("Size labeled set temp " + str(labeled_set_temp.shape[0]))
    print("Size validation set temp " + str(validation_set_temp.shape[0]))


    # Delete all predictions
    cmd = 'rm %s' % (output_path+task+'_test_prediction.jsonl')
    os.system(cmd)

    # Pick a checkpoint, rename it
    cmd = 'rm checkpoints/%s' % (model)
    os.system(cmd)
    cmd = 'mv *%s*_dev.pt checkpoints/%s' % (task, model)
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
      --max_len %d \
      --use_gpu \
      --fp16 \
      --checkpoint_path checkpoints/""" % (task, input_path+task+'_test.txt',
      output_path+task+'_test_prediction.jsonl', learning_model, max_len)

    if dk:
      cmd += ' --dk %s' % dk
    if su:
      cmd += ' --summarize'  

    #os.system(cmd)
    # invoke process
    process = subprocess.Popen(shlex.split(cmd),shell=False,stdout=subprocess.PIPE)

    # Poll process.stdout to show stdout live
    while True:
        output = process.stdout.readline()
        if process.poll() is not None:
            break
        if output:
            if verbose:
                print(output.strip())
    rc = process.poll()
    print('Return code: ' + str(rc))

    test_predictions = pd.read_json(output_path+task+'_test_prediction.jsonl', lines=True)

    # calculate scores
    prec, recall, fscore, _ = precision_recall_fscore_support(
            test_data['label'],
            test_predictions['match'],
            average='binary')

    print('f1: ' + str(round(fscore,3)))
    print('precision: ' + str(round(prec,3)))
    print('recall: ' + str(round(recall,3)))

    f1_scores.append(round(fscore,3))
    precision_scores.append(round(prec,3))
    recall_scores.append(round(recall,3))
    labeled_set_size.append(number_labeled_examples)
    if include_tl_data:
        all_sample_weights.append([source_weight,target_weight])

  all_scores = pd.DataFrame(
    {'labeled set size': labeled_set_size,
    'f1': f1_scores,
    'precision': precision_scores,
    'recall': recall_scores
    })

  if data_augmentation:
        all_scores['da labels']=data_augmentation_labels

  if include_tl_data:
      all_scores['sample weights']=all_sample_weights

  print(all_scores)

  return all_scores
