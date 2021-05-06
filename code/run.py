from active_learning import *

import deepmatcher as dm
import pandas as pd
import time
from datetime import date

from sklearn.metrics import precision_recall_fscore_support

def run_al(dataset, num_runs, sampling_size, save_results_file, transfer_learning_dataset=None, init_random_sample=False, ignore_columns=('source_id','target_id'), file_path='', data_augmentation=True, high_conf_to_ls=False, attr_summarizer='rnn', attr_comparator='abs-diff', embeddings='fasttext.en.bin', epochs=20, batch_size=16, pos_neg_ratio=1, path_tl_model='tl_model.pth', path_al_model='al_model.pth', embeddings_cache_path='~/.vector_cache'):
    
    # Load datasets
    train_data = pd.read_csv(file_path + dataset + '_train')
    validation_data = pd.read_csv(file_path + dataset + '_validation')
    test_data = pd.read_csv(file_path + dataset + '_test')

    model = dm.MatchingModel(attr_summarizer=attr_summarizer, attr_comparator=attr_comparator)
    
    # Initialize model
    if transfer_learning_dataset != None:
        init_method='Transfer Learning'
        
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
            pos_neg_ratio=pos_neg_ratio)

    else:
        init_method = 'Random Sample ' + str(init_random_sample)
        random_train_data = train_data.sample(n=init_random_sample, weights=None, axis=None)
        random_train_data.to_csv('random_train_set', index=False)
        random_validation_data = validation_data.sample(n=int(init_random_sample/3), weights=None, axis=None)
        random_validation_data.to_csv('random_validation_set', index=False)

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
            pos_neg_ratio=pos_neg_ratio)

    results_al = active_learning(train_data, validation_data, test_data, init_method,
        num_runs, sampling_size, model, ignore_columns, file_path, data_augmentation,
        high_conf_to_ls, epochs, batch_size, embeddings, pos_neg_ratio, path_al_model)

    # build final results dataframe and save results
    results_al['Dataset']=dataset
    results_al['Initialization Method']=init_method
    results_al['Transfer Learning Dataset']=transfer_learning_dataset
    results_al['AL Runs']=num_runs
    results_al['Sampling Size']=sampling_size
    results_al['Attribute Summarizer']=attr_summarizer
    results_al['Attribute Comparator']=attr_comparator
    results_al['Embeddings']=embeddings
    results_al['Epochs']=epochs
    results_al['Batch Size']=batch_size
    results_al['Pos.Neg.Ratio']=pos_neg_ratio
    results_al['Data Augmentation']=data_augmentation
    results_al['High.Conf.LS']=high_conf_to_ls
    results_al.to_csv(save_results_file, index=False)

    return results_al


def run_pl(dataset, save_results_file, train_size=None, ignore_columns=('source_id','target_id'), file_path='', attr_summarizer='rnn', attr_comparator='abs-diff', embeddings='fasttext.en.bin', epochs=20, batch_size=16, pos_neg_ratio=1, path_pl_model='pl_model.pth', embeddings_cache_path='~/.vector_cache'):

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
        pos_neg_ratio=pos_neg_ratio)

    end_time = time.time()

    # Evaluate model
    prec, recall, fscore, support = precision_recall_fscore_support(
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
                   'Train Size':train_size,'F1':fscore,'Precision':prec,'Recall':recall,'Runtime':(end_time - start_time),
                   'Attr.Summarizer':attr_summarizer,'Attr.Comparator':attr_comparator,
                   'Embeddings':embeddings,'Epochs':epochs,'Batch Size':batch_size,
                   'Pos.Neg.Ratio':pos_neg_ratio}, ignore_index=True)
                   
    results_pl.to_csv(save_results_file, index=False)

    return results_pl