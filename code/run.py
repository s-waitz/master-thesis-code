from active_learning import *

import deepmatcher as dm
import pandas as pd
import numpy as np
import time
from datetime import date
import os.path

from sklearn.metrics import precision_recall_fscore_support

def run_al(dataset, num_runs, al_iterations, sampling_size, save_results_path, transfer_learning_dataset=None, init_random_sample=False, ignore_columns=('source_id','target_id'), file_path='', data_augmentation=False, high_conf_to_ls=False, da_threshold=0, attr_summarizer='rnn', attr_comparator='abs-diff', embeddings='fasttext.en.bin', epochs=20, batch_size=20, lr_decay=0.8, embeddings_cache_path='~/.vector_cache'):
    
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
        
        experiment_name = 'al_{}_runs{}_ss{}_init_{}_da_{}_hc_{}_thresh_{}_model_{}_epochs{}_batch{}_lrdecay{}_{}_{}'.format(
            dataset,al_iterations,sampling_size,init,data_augmentation,high_conf_to_ls,da_threshold,attr_summarizer,epochs,batch_size,lr_decay,day,x)

        save_results_file = save_results_path + experiment_name + '_results.csv'
        
        path_al_model = save_results_path + experiment_name + '_al_model.pth'
            
        path_tl_model = save_results_path + experiment_name + '_tl_model.pth'

        if os.path.isfile(save_results_file):
            # increase iterator if file already exists
            x+=1
        else:
            break
    
    for run in range(1,num_runs+1):

        # Load datasets
        train_data = pd.read_csv(file_path + dataset + '_train')
        validation_data = pd.read_csv(file_path + dataset + '_validation')
        test_data = pd.read_csv(file_path + dataset + '_test')

        model = dm.MatchingModel(attr_summarizer=attr_summarizer, attr_comparator=attr_comparator)
        
        # Initialize model
        if transfer_learning_dataset != None:

            # calculate positive negative ratio
            train_data_tl = pd.read_csv(file_path + transfer_learning_dataset + '_train')
            pn_ratio_tl = round((train_data_tl['label'].shape[0] - train_data_tl['label'].sum()) / train_data_tl['label'].sum())
            if pn_ratio_tl == 0:
                pn_ratio_tl = 1
            
            print('Positve Negative Ratio TL: ' + str(pn_ratio_tl))
            
            train_tl, validation_tl, test_tl = dm.data.process(
                path=file_path,
                train=transfer_learning_dataset + '_train',
                validation=transfer_learning_dataset + '_validation',
                test=transfer_learning_dataset + '_test',
                ignore_columns=ignore_columns,
                left_prefix='left_',
                right_prefix='right_',
                label_attr='label',
                id_attr='id',
                cache=False,
                embeddings=embeddings,
                embeddings_cache_path=embeddings_cache_path)

            model.run_train(
                train_tl,
                validation_tl,
                epochs=epochs,
                batch_size=batch_size,
                best_save_path=path_tl_model,
                pos_neg_ratio=pn_ratio_tl)

        else:
            random_train_data = train_data.sample(n=init_random_sample, weights=None, axis=None)
            random_train_data.to_csv('random_train_set', index=False)
            random_validation_data = validation_data.sample(n=int(init_random_sample/3), weights=None, axis=None)
            random_validation_data.to_csv('random_validation_set', index=False)

            # calculate positive negative ratio
            pn_ratio_init = round((random_train_data['label'].shape[0] - random_train_data['label'].sum()) / random_train_data['label'].sum())
            if pn_ratio_init == 0:
                pn_ratio_init = 1
            
            print('Positve Negative Ratio TL: ' + str(pn_ratio_init))

            random_train_set, random_validation_set, _ = dm.data.process(
                path='',
                train='random_train_set',
                validation='random_validation_set',
                test=file_path + dataset + '_train',
                ignore_columns=ignore_columns,
                left_prefix='left_',
                right_prefix='right_',
                label_attr='label',
                id_attr='id',
                cache=None,
                embeddings=embeddings,
                embeddings_cache_path=embeddings_cache_path)

            model.run_train(
                random_train_set,
                random_validation_set,
                epochs=epochs,
                batch_size=batch_size,
                best_save_path='init_model.pth',
                pos_neg_ratio=pn_ratio_init)

        results_al = active_learning(train_data, validation_data, test_data, init_method,
            al_iterations, sampling_size, model, ignore_columns, file_path, data_augmentation,
            high_conf_to_ls, da_threshold, epochs, batch_size, lr_decay, embeddings, path_al_model,
            attr_summarizer, attr_comparator)

        if run == 1:
            results = pd.DataFrame()
            # build final results dataframe and save results
            results['labeled set size']=results_al['labeled set size']
            results['Dataset']=dataset
            results['Initialization Method']=init_method
            results['Transfer Learning Dataset']=transfer_learning_dataset
            results['AL Runs']=al_iterations
            results['Sampling Size']=sampling_size
            results['Attribute Summarizer']=attr_summarizer
            results['Attribute Comparator']=attr_comparator
            results['Embeddings']=embeddings
            results['Epochs']=epochs
            results['Batch Size']=batch_size
            results['LR Decay']=lr_decay
            results['Data Augmentation']=data_augmentation
            results['High.Conf.LS']=high_conf_to_ls
            results['DA Threshold']=da_threshold

        results['Run ' + str(run) + ': f1'] = results_al['f1']
        results['Run ' + str(run) + ': precision'] = results_al['precision']
        results['Run ' + str(run) + ': recall'] = results_al['recall']
        results['Run ' + str(run) + ': pos_neg_ratio'] = results_al['pos_neg_ratio']

        if run > 1:
            all_f1 = np.vstack((all_f1,results_al['f1']))
            all_precision = np.vstack((all_precision,results_al['precision']))
            all_recall = np.vstack((all_recall,results_al['recall']))
        else:
            all_f1 = results_al['f1']
            all_precision = results_al['precision']
            all_recall = results_al['recall']

    # calculate mean and standard deviation
    mean_f1 = np.mean(all_f1,axis=0)
    std_f1 = np.std(all_f1,axis=0)
    mean_precision = np.mean(all_precision,axis=0)
    std_precision = np.std(all_precision,axis=0)
    mean_recall = np.mean(all_recall,axis=0)
    std_recall = np.std(all_recall,axis=0)

    results['F1 Mean'] = round(mean_f1,3)
    results['F1 Std'] = round(std_f1,3)
    results['Precision Mean'] = round(mean_precision,3)
    results['Precision Std'] = round(std_precision,3)
    results['Recall Mean'] = round(mean_recall,3)
    results['Recall Std'] = round(std_recall,3)

    results.to_csv(save_results_file, index=False)

    return results


def run_pl(dataset, save_results_file, train_size=None, ignore_columns=('source_id','target_id'), file_path='', attr_summarizer='rnn', attr_comparator='abs-diff', embeddings='fasttext.en.bin', epochs=20, batch_size=20, path_pl_model='pl_model.pth', embeddings_cache_path='~/.vector_cache'):

    # Load datasets

    if train_size:
        # select samples
        train_data = pd.read_csv(file_path + dataset + '_train').sample(n=train_size, weights=None, axis=None)
        validation_data = pd.read_csv(file_path + dataset + '_validation').sample(n=int(train_size/3), weights=None, axis=None)
        test_data = pd.read_csv(file_path + dataset + '_test').sample(n=int(train_size/3), weights=None, axis=None)
        
        # rewrite samples to csv
        train_data.to_csv(dataset + '_train', index=False)
        validation_data.to_csv(dataset + '_validation', index=False)
        test_data.to_csv(dataset + '_test', index=False)

        # reset file path
        file_path = ''
    else:
        train_data = pd.read_csv(file_path + dataset + '_train')
        validation_data = pd.read_csv(file_path + dataset + '_validation')
        test_data = pd.read_csv(file_path + dataset + '_test')

    # calculate positive negative ratio
    pn_ratio = round((train_data['label'].shape[0] - train_data['label'].sum()) / train_data['label'].sum())
    if pn_ratio == 0:
        pn_ratio = 1

    model = dm.MatchingModel(attr_summarizer=attr_summarizer, attr_comparator=attr_comparator)

    # process datasets
    train_set, validation_set, test_set = dm.data.process(
        path=file_path,
        train=dataset + '_train',
        validation=dataset + '_validation',
        test=dataset + '_test',
        ignore_columns=ignore_columns,
        left_prefix='left_',
        right_prefix='right_',
        label_attr='label',
        id_attr='id',
        cache=False,
        embeddings=embeddings,
        embeddings_cache_path=embeddings_cache_path)

    start_time = time.time()

    model.run_train(
        train_set,
        validation_set,
        epochs=epochs,
        batch_size=batch_size,
        best_save_path=path_pl_model,
        pos_neg_ratio=pn_ratio)

    end_time = time.time()

    # Evaluate model
    prec, recall, fscore, _ = precision_recall_fscore_support(
        test_data['label'],
        np.where(model.run_prediction(test_set)['match_score'] >= 0.5, 1, 0),
        average='binary')

    try:
        results_pl = pd.read_csv(save_results_file)
    except:
        results_pl = pd.DataFrame(columns = ['Dataset','Method','Date','Train Size', 'F1',
                                        'Precision','Recall','Runtime','Attr.Summarizer',
                                        'Attr.Comparator','Embeddings','Epochs',
                                        'Batch Size','Pos.Neg.Ratio'])

    results_pl = results_pl.append({'Dataset':dataset,'Method':'Passive Learing','Date':date.today().strftime("%d.%m.%Y"),
                   'Train Size':train_size,'F1':round(fscore,3),'Precision':round(prec,3),'Recall':round(recall,3),'Runtime':round(end_time - start_time,3),
                   'Attr.Summarizer':attr_summarizer,'Attr.Comparator':attr_comparator,
                   'Embeddings':embeddings,'Epochs':epochs,'Batch Size':batch_size,
                   'Pos.Neg.Ratio':pn_ratio}, ignore_index=True)
                   
    results_pl.to_csv(save_results_file, index=False)

    return results_pl