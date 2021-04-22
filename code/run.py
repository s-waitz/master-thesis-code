from active_learning import *

import deepmatcher as dm
import pandas as pd

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
        pass

    results_al = active_learning(train_data, validation_data, test_data,
        num_runs, sampling_size, model, ignore_columns, file_path, high_conf_to_ls,
        epochs, batch_size, embeddings, pos_neg_ratio, path_al_model)

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
    results_al.to_csv(save_results_file, index=False)

    return results_al