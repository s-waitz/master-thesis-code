from active_learning import *

import deepmatcher as dm
import pandas as pd

def run_al(dataset, num_runs, sampling_size, save_results_file, transfer_learning_dataset=None, file_path='', high_conf_to_ls=False, attr_summarizer='hybrid', embeddings='glove.42B.300d', train_epochs=20, train_batch_size=16):
    
    # Load datasets
    train_data = pd.read_csv(file_path + dataset + '_train')
    validation_data = pd.read_csv(file_path + dataset + '_validation')
    test_data = pd.read_csv(file_path + dataset + '_test')

    model = dm.MatchingModel(attr_summarizer=attr_summarizer)

    # Initialize model
    if transfer_learning_dataset != None:
        init_method='Transfer Learning'
        # todo
    else:
        init_method='Random'
        # todo
        # only for the moment
        init_sample = train_data.sample(n=200, weights=None, axis=None)
        init_sample.to_csv('init_sample',index=False)
        init_set_train, init_set_validation, init_set_test = dm.data.process(
            path=file_path,
            train='init_sample',
            validation=file_path + dataset + '_validation',
            test=file_path + dataset + '_test',
            ignore_columns=('source_id', 'target_id'),
            left_prefix='left_',
            right_prefix='right_',
            label_attr='label',
            id_attr='id',
            cache=False,
            embeddings=embeddings)
        # initial training
        # only until other init procedures are implemented
        model.run_train(
            init_set_train,
            init_set_validation,
            epochs=10,
            batch_size=16,
            best_save_path='init_model.pth',
            pos_neg_ratio=3)

    results_al = active_learning(train_data, validation_data, test_data, num_runs, sampling_size, model,
        file_path=file_path, high_conf_to_ls=high_conf_to_ls, train_epochs=train_epochs, train_batch_size=train_batch_size, embeddings=embeddings)

    # build final results dataframe and save results
    results_al['Dataset']=dataset
    results_al['Initialization Method']=init_method
    results_al['Transfer Learning Dataset']=transfer_learning_dataset
    results_al['AL Runs']=num_runs
    results_al['Sampling Size']=sampling_size
    results_al['Attribute Summarizer']=attr_summarizer
    results_al['Embeddings']=embeddings
    results_al['Epochs']=train_epochs
    results_al['Batch Size']=train_batch_size
    results_al.to_csv(save_results_file, index=False)

    return results_al