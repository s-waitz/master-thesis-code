import os
from scipy.sparse.construct import random
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
import pandas as pd
from datetime import date
import os.path
import shlex
import subprocess
import numpy as np
import glob
#import torch
#import json
from collections import OrderedDict
#from Snippext_public.snippext.model import MultiTaskNet

from ditto_helper import to_ditto_format, to_jsonl
from active_learning_ditto import active_learning_ditto

def run_al_ditto(task, num_runs, al_iterations, sampling_size, save_results_path, base_data_path, labeled_set_path, keep_model=True, transfer_learning_dataset=None, include_tl_data=False, tl_weights=None, init_random_sample=False, data_augmentation=False, high_conf_to_ls=False, da_threshold=0, input_path='input/', output_path='output/', learning_model='roberta', learning_rate='3e-5', max_len=256, batch_size=32, epochs_tl=None, epochs=40, reduce_epochs=False, balance=False, da='del', dk='general', su=False, verbose=False):

    # Delete files from last run
    files_su = glob.glob(labeled_set_path+task+'*su*')
    files_dk = glob.glob(labeled_set_path+task+'*dk*')
    files_to_delete = files_su + files_dk
    files_to_delete = set(files_to_delete)
    for file in files_to_delete:
        os.remove(file)

    # Delete all models for this task
    cmd = 'rm *%s*.pt' % (task)
    cmd = 'rm *%s*.pt' % (transfer_learning_dataset)
    os.system(cmd)

    model = str(task) + '.pt'

    if transfer_learning_dataset != None:
        init_method='Transfer Learning'
        init='tl'
    else:
        init_method = 'Random Sample ' + str(init_random_sample)
        init = 'rs' + str(init_random_sample)

    # filename results
    day = date.today().strftime('%Y%m%d')
    x=1
    
    while True:
        
        experiment_name = 'al_{}_runs{}_ss{}_init_{}_da_{}_hc_{}_thresh_{}_epochs{}_batch{}_{}_{}'.format(
            task,al_iterations,sampling_size,init,data_augmentation,high_conf_to_ls,da_threshold,epochs,batch_size,day,x)

        save_results_file = save_results_path + experiment_name + '_results.csv'
        
        #path_al_model = save_results_path + experiment_name + '_al_model.pth'
            
        #path_tl_model = save_results_path + experiment_name + '_tl_model.pth'

        if os.path.isfile(save_results_file):
            # increase iterator if file already exists
            x+=1
        else:
            break
    
    for run in range(1,num_runs+1):

        print("Run: " + str(run))
        
        # Load datasets
        train_data = pd.read_csv(base_data_path + task + '_train')
        validation_data = pd.read_csv(base_data_path + task + '_validation')
        test_data = pd.read_csv(base_data_path + task + '_test')

        to_ditto_format(test_data, labeled_set_path+task+'_test.txt')
        
        # Initialize model
        if transfer_learning_dataset != None:

            random_sample = None

            train_data_tl = pd.read_csv(base_data_path + transfer_learning_dataset + '_train')

            if epochs_tl == None:
                epochs_tl = epochs

            # Train model on transfer learning dataset

            print('Initialize model with transfer learning ...')

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
              --save_model""" % (transfer_learning_dataset, batch_size, max_len, learning_rate, epochs_tl,
              learning_model)
            if da:
                cmd += ' --da %s' % da
            if dk:
                cmd += ' --dk %s' % dk
            if su:
                cmd += ' --summarize'
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
                    if verbose:
                        print(output.strip())
            rc = process.poll()
            print('Return code: ' + str(rc))

            # Move model to checkpoints
            cmd = 'mv *%s*_dev.pt checkpoints/%s' % (transfer_learning_dataset, model)
            os.system(cmd)

            # workaround to enable transfer learning
            #configs = json.load(open('configs.json'))
            #configs = {conf['name'] : conf for conf in configs}
            #config = configs[task]
            #m = MultiTaskNet([config], 'cuda', True, lm=learning_model)
            # load state dict
            #state_dict = torch.load('checkpoints/' + str(model), map_location=lambda storage, loc: storage)
            # change state dict for transfer learning
            #state_dict = OrderedDict({x.replace(transfer_learning_dataset, task): v for x, v in state_dict.items()})
            #m.load_state_dict(state_dict)
            # save changed state dict
            #torch.save(state_dict, 'checkpoints/' + str(model))
            #torch.save(m.state_dict(), 'checkpoints/' + str(model))

        else:

            train_data_tl = None

            all_data = pd.concat([train_data, validation_data]).reset_index(drop=True)
            random_sample = all_data.sample(n=init_random_sample, weights=None, axis=None)
            
            y = random_sample['label']
            try:
                random_train_data, random_validation_data, _, _ = train_test_split(random_sample, y,
                                                    stratify=y, 
                                                    test_size=0.25)
            except ValueError:
                random_train_data, random_validation_data, _, _ = train_test_split(random_sample, y,
                                                    stratify=None,
                                                    test_size=0.25)

            to_ditto_format(random_train_data, labeled_set_path+task+'_train.txt')
            to_ditto_format(random_validation_data, labeled_set_path+task+'_validation.txt')

            # Train model on random sample

            print('Initialize model on random sample ...')

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
              --save_model""" % (task, batch_size, max_len, learning_rate,
              epochs, learning_model)
            if da:
                cmd += ' --da %s' % da
            if dk:
                cmd += ' --dk %s' % dk
            if su:
                cmd += ' --summarize'
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
                    if verbose:
                        print(output.strip())
            rc = process.poll()
            print('Return code: ' + str(rc))

            # Move model to checkpoints
            cmd = 'mv *%s*_dev.pt checkpoints/%s' % (task, model)
            os.system(cmd)
       
        results_al = active_learning_ditto(task, random_sample, al_iterations, sampling_size,
                                           train_data_tl, include_tl_data, tl_weights, base_data_path, labeled_set_path,
                                           input_path, output_path, data_augmentation, 
                                           high_conf_to_ls, da_threshold, learning_model, 
                                           learning_rate, max_len, batch_size, epochs, reduce_epochs,
                                           balance, da, dk, su, verbose, keep_model)
        if run == 1:
            results = pd.DataFrame()
            # build final results dataframe and save results
            results['labeled set size']=results_al['labeled set size']
            results['Task']=task
            results['Initialization Method']=init_method
            results['Transfer Learning Dataset']=transfer_learning_dataset
            results['Keep Model']=keep_model
            results['AL Runs']=al_iterations
            results['Sampling Size']=sampling_size
            results['Learning Model']=learning_model
            results['Learning Rate']=learning_rate
            results['Max Length']=max_len
            results['Epochs']=epochs
            results['Batch Size']=batch_size
            results['Data Augmentation']=data_augmentation
            results['High.Conf.LS']=high_conf_to_ls
            results['DA Threshold']=da_threshold
            results['DA']=da
            results['DK']=dk
            results['SU']=su
            if include_tl_data:
                results['sample weights'] = results_al['sample weights']

        results['Run ' + str(run) + ': f1'] = results_al['f1']
        results['Run ' + str(run) + ': precision'] = results_al['precision']
        results['Run ' + str(run) + ': recall'] = results_al['recall']
        if data_augmentation:
            results['Run ' + str(run) + ': da labels'] = results_al['da labels']

        if run > 1:
            all_f1 = np.vstack((all_f1,results_al['f1']))
            all_precision = np.vstack((all_precision,results_al['precision']))
            all_recall = np.vstack((all_recall,results_al['recall']))
        else:
            all_f1 = results_al['f1']
            all_precision = results_al['precision']
            all_recall = results_al['recall']

        # save intermediate results
        results.to_csv(save_results_file, index=False)

    # calculate mean and standard deviation
    mean_f1 = np.mean(all_f1,axis=0)
    std_f1 = np.std(all_f1,axis=0)
    mean_precision = np.mean(all_precision,axis=0)
    std_precision = np.std(all_precision,axis=0)
    mean_recall = np.mean(all_recall,axis=0)
    std_recall = np.std(all_recall,axis=0)

    results['F1 Mean'] = np.round(mean_f1,3)
    results['F1 Std'] = np.round(std_f1,3)
    results['Precision Mean'] = np.round(mean_precision,3)
    results['Precision Std'] = np.round(std_precision,3)
    results['Recall Mean'] = np.round(mean_recall,3)
    results['Recall Std'] = np.round(std_recall,3)

    results.to_csv(save_results_file, index=False)

    return results



def run_pl_ditto(task, save_results_file, base_data_path, ditto_data_path, train_size=None, input_path='input/', output_path='output/', learning_model='roberta', learning_rate='3e-5', max_len=256, batch_size=32, epochs=2, balance=False, da='del', dk='general', su=True, verbose=False):
    
    # Delete files from last run
    files_su = glob.glob(ditto_data_path+task+'*su*')
    files_dk = glob.glob(ditto_data_path+task+'*dk*')
    files_to_delete = files_su + files_dk
    files_to_delete = set(files_to_delete)
    for file in files_to_delete:
        os.remove(file)
    
    # Delete all models
    cmd = 'rm *%s*.pt' % (task)
    os.system(cmd)

    model = str(task) + '.pt'

    if train_size:
        # select samples
        train_data = pd.read_csv(base_data_path + task + '_train')
        validation_data = pd.read_csv(base_data_path + task + '_validation')
        all_data = pd.concat([train_data, validation_data]).reset_index(drop=True)
        sample_data = all_data.sample(n=train_size, weights=None, axis=None)
        y = sample_data['label']
        train_data, validation_data, _, _ = train_test_split(sample_data, y,
                                            stratify=y, 
                                            test_size=0.25)
        
        test_data = pd.read_csv(base_data_path + task + '_test')
    
    else:
        train_data = pd.read_csv(base_data_path + task + '_train')
        validation_data = pd.read_csv(base_data_path + task + '_validation')
        test_data = pd.read_csv(base_data_path + task + '_test')
 
    to_ditto_format(train_data, ditto_data_path+task+'_train.txt')
    to_ditto_format(validation_data, ditto_data_path+task+'_validation.txt')
    to_ditto_format(test_data, ditto_data_path+task+'_test.txt')
    to_ditto_format(test_data, input_path+task+'_test.txt')
    to_jsonl(test_data, input_path+task+'_test.jsonl')

    # Train model

    print('Train model ...')

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
    if da:
        cmd += ' --da %s' % da
    if dk:
        cmd += ' --dk %s' % dk
    if su:
        cmd += ' --summarize'        
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
            if verbose:
                print(output.strip())
    rc = process.poll()
    print('Return code: ' + str(rc))

    # Delete all predictions
    cmd = 'rm %s' % (output_path+task+'_test_prediction.jsonl')
    os.system(cmd)

    # Pick a checkpoint, rename it
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
    
    try:
        results_pl = pd.read_csv(save_results_file)
    except:
        results_pl = pd.DataFrame(columns = ['Task','Method','Date','Train Size', 'F1',
                                        'Precision','Recall','Learning Model','Learning Rate',
                                        'Max Length','Epochs','Batch Size','DA','DK','SU','Balance'])

    results_pl = results_pl.append({'Task':task,'Method':'Passive Learing','Date':date.today().strftime("%d.%m.%Y"),
                   'Train Size':train_size,'F1':round(fscore,3),'Precision':round(prec,3),'Recall':round(recall,3),
                   'Learning Model':learning_model, 'Learning Rate':learning_rate, 'Max Lenght':max_len,
                   'Epochs':epochs,'Batch Size':batch_size,'DA':da,'DK':dk,'SU':su,'Balance':balance},ignore_index=True)
                   
    results_pl.to_csv(save_results_file, index=False)

    return results_pl