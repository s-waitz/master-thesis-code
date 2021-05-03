from active_learning import *

import deepmatcher as dm
import pandas as pd
import time
from datetime import date

from sklearn.metrics import precision_recall_fscore_support

def run_al(dataset, num_runs, sampling_size, save_results_file, transfer_learning_dataset=None, ignore_columns=('source_id','target_id'), file_path='', high_conf_to_ls=False, attr_summarizer='rnn', attr_comparator='abs-diff', embeddings='fasttext.en.bin', epochs=20, batch_size=16, pos_neg_ratio=1, path_tl_model='tl_model.pth', path_al_model='al_model.pth', embeddings_cache_path='~/.vector_cache'):
    
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
        init_method = 'Random Initialization'

        train_data.to_csv('train_set', index=False)

        train_set = dm.data.process(
            path='',
            train='train_set',
            ignore_columns=ignore_columns,
            left_prefix='left_',
            right_prefix='right_',
            label_attr='label',
            id_attr='id',
            cache=None,
            embeddings=embeddings,
            embeddings_cache_path=embeddings_cache_path)

        model.initialize(train_set)

        # workaround
        model.meta.__dict__.update({'lowercase': True})
        model.meta.__dict__.update({'tokenize': 'nltk'})
        model.meta.__dict__.update({'include_lengths': True})
        model.__dict__.update({'epoch':0})

        # init_method='200 Random Examples'
        # labeled_set = pd.read_csv(file_path+dataset+'_train').sample(n=200, weights=None, axis=None)
        # labeled_set.to_csv(file_path+'labeled_set',index=False)
        # labeled_set_train, validation_set, test_set = dm.data.process(
        #     path=file_path,
        #     train='labeled_set',
        #     validation=dataset+'_validation',
        #     test=dataset+'_test',
        #     ignore_columns=ignore_columns,
        #     left_prefix='left_',
        #     right_prefix='right_',
        #     label_attr='label',
        #     id_attr='id',
        #     cache=False,
        #     embeddings=embeddings,
        #     embeddings_cache_path=embeddings_cache_path)
        # model.run_train(
        #     labeled_set_train,
        #     validation_set,
        #     epochs=epochs,
        #     batch_size=batch_size,
        #     best_save_path=path_tl_model,
        #     pos_neg_ratio=pos_neg_ratio)

    results_pl = active_learning(train_data, validation_data, test_data,
        num_runs, sampling_size, model, ignore_columns, file_path, high_conf_to_ls,
        epochs, batch_size, embeddings, pos_neg_ratio, path_al_model)

    # build final results dataframe and save results
    results_pl['Dataset']=dataset
    results_pl['Initialization Method']=init_method
    results_pl['Transfer Learning Dataset']=transfer_learning_dataset
    results_pl['AL Runs']=num_runs
    results_pl['Sampling Size']=sampling_size
    results_pl['Attribute Summarizer']=attr_summarizer
    results_pl['Attribute Comparator']=attr_comparator
    results_pl['Embeddings']=embeddings
    results_pl['Epochs']=epochs
    results_pl['Batch Size']=batch_size
    results_pl['Pos.Neg.Ratio']=pos_neg_ratio
    results_pl.to_csv(save_results_file, index=False)

    return results_pl


def run_pl(dataset, save_results_file, ignore_columns=('source_id','target_id'), file_path='', attr_summarizer='rnn', attr_comparator='abs-diff', embeddings='fasttext.en.bin', epochs=20, batch_size=16, pos_neg_ratio=1, path_pl_model='pl_model.pth', embeddings_cache_path='~/.vector_cache'):

    # Load datasets
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
        results_pl = pd.DataFrame(columns = ['Dataset','Method','Date','F1',
                                        'Precision','Recall','Runtime','Attr.Summarizer',
                                        'Attr.Comparator','Embeddings','Epochs',
                                        'Batch Size','Pos.Neg.Ratio'])

    results_pl.append({'Dataset':dataset,'Method':'Passive Learing','Date':date.today().strftime("%d.%m.%Y"),
                   'F1':fscore,'Precision':prec,'Recall':recall,'Runtime':(end_time - start_time),
                   'Attr.Summarizer':attr_summarizer,'Attr.Comparator':attr_comparator,
                   'Embeddings':embeddings,'Epochs':epochs,'Batch Size':batch_size,
                   'Pos.Neg.Ratio':pos_neg_ratio}, ignore_index=True)
                   
    results_pl.to_csv(save_results_file, index=False)

    return results_pl