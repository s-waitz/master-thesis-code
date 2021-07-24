import pandas as pd
import numpy as np
import copy

from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

import deepmatcher as dm

def active_learning(train_data, validation_data, test_data, init_method, random_sample, train_data_tl, include_tl_data, tl_weights, al_iterations, sampling_size, model, ignore_columns, file_path, data_augmentation, high_conf_to_ls, da_threshold, train_epochs, train_batch_size, lr_decay, embeddings, path_al_model, attr_summarizer, attr_comparator, split_validation, keep_model):
    """    
        Args:
        train_data (pd.DataFrame): ...
        validation_data (pd.DataFrame): ...
        test_data (pd.DataFrame): ...
        al_iterations (int): ...
        sampling_size (int): ...
        model (dm.MatchingModel): ...
        file_path (str, optional): ... Defaults to ''.
        high_conf_to_ls: ... Defaults to True.
        add_pairs (bool, optional): ... Defaults to True.
        train_epochs (int, optional): ... Defaults to 20.
        train_batch_size (int, optional): ... Defaults to 16.
        embeddings (string, optional): http://pages.cs.wisc.edu/~sidharth/deepmatcher/data.html. Defaults to 'fasttext.en.bin'.
    Returns:
        pd.DataFrame: Dataframe containing results for every active learning run.
    """
    def round_int(num):
        if num == float("inf") or num == float("-inf"):
            return 0
        return int(round(num))
        
    # labeled set from random initialization, if tl None    
    labeled_set_raw = random_sample

    # write data to csv for later processing
    if split_validation == False:
        validation_data.to_csv('validation_set', index=False)
    else:
        #merge train and validation set, since no explicit validation set is used
        train_data = pd.concat([train_data, validation_data]).reset_index(drop=True)
    test_data.to_csv('test_set', index=False)

    oracle = train_data.copy()
    pool_data = train_data.copy()

    if split_validation == False:
        validation_set, test_set = dm.data.process(
            path='',
            validation='validation_set',
            test='test_set', 
            ignore_columns=ignore_columns,
            left_prefix='left_',
            right_prefix='right_',
            label_attr='label',
            id_attr='id',
            cache=None,
            embeddings=embeddings)
    else:
        test_set = dm.data.process(
            path='',
            test='test_set', 
            ignore_columns=ignore_columns,
            left_prefix='left_',
            right_prefix='right_',
            label_attr='label',
            id_attr='id',
            cache=None,
            embeddings=embeddings)

    f1_scores = []
    precision_scores = []
    recall_scores = []
    labeled_set_size = []
    pos_neg_ratios = []
    data_augmentation_labels = []
    all_sample_weights = []

    if random_sample is None:
        number_labeled_examples = 0
    else:
        number_labeled_examples = random_sample.shape[0]
        #remove labeled pairs from unlabeled pool
        pool_data = pool_data[~pool_data['id'].isin(labeled_set_raw['id'].tolist())]


    # Save results for first prediction with initialized model
    # todo
    prec, recall, fscore, _ = precision_recall_fscore_support(
        test_data['label'],
        np.where(model.run_prediction(test_set)['match_score'] >= 0.5, 1, 0),
        average='binary')

    f1_scores.append(round(fscore,3))
    precision_scores.append(round(prec,3))
    recall_scores.append(round(recall,3))
    labeled_set_size.append(number_labeled_examples)
    pos_neg_ratios.append(0)
    data_augmentation_labels.append([0,0,0,0])
    all_sample_weights.append([0,0])

    # (1)
    for i in range(1,al_iterations+1):


        print("AL run: " + str(i))

        print('Ignore: ')
        print(ignore_columns)

        # process unlabeled pool for deepmatcher
        #pool_data.to_csv('unlabeled_pool', index=False)

        #unlabeled_pool = dm.data.process_unlabeled(
        #    path='unlabeled_pool',
        #    trained_model=model,
        #    ignore_columns=ignore_columns)
        pool_data.to_csv('train_set', index=False)
        train_set = dm.data.process(
            path='',
            train='train_set',
            ignore_columns=ignore_columns,
            left_prefix='left_',
            right_prefix='right_',
            label_attr='label',
            id_attr='id',
            cache=None,
            embeddings=embeddings)

        if i == 1 and init_method == 'Transfer Learning':
            model._reset_embeddings(train_set.vocabs)

        # Predict probabilities
        predictions = model.run_prediction(train_set)

        # Calculate entropy based on probabilities
        predictions['entropy'] = predictions['match_score'].apply(lambda p: -p * np.log(p) - (1 - p) * np.log(1 - p))

        # Split in matches and non-matches (based on probabilities)
        predictions_true = predictions[predictions['match_score']>=0.5]
        predictions_false = predictions[predictions['match_score']<0.5]

        # Select k pairs with highest entropy for active learning
        low_conf_pairs_true = predictions_true['entropy'].nlargest(int(sampling_size/2))
        low_conf_pairs_false = predictions_false['entropy'].nlargest(int(sampling_size/2))
        
        # if now true pairs add false pairs instead
        if low_conf_pairs_true.shape[0] == 0:
            low_conf_pairs_false = low_conf_pairs_false.append(predictions_false['entropy'].nlargest(int(sampling_size/2)))
        # if now false pairs add true pairs instead
        if low_conf_pairs_false.shape[0] == 0:
            low_conf_pairs_true = low_conf_pairs_true.append(predictions_true['entropy'].nlargest(int(sampling_size/2)))

        print('low_conf_pairs_true ' + str(low_conf_pairs_true.shape[0]))
        print('low_conf_pairs_false ' + str(low_conf_pairs_false.shape[0]))
        
        # Label these pairs with oracle and add them to labeled set
        labeled_set_new = oracle[oracle['id'].isin(low_conf_pairs_true.index.tolist())]
        labeled_set_new= labeled_set_new.append(oracle[oracle['id'].isin(low_conf_pairs_false.index.tolist())])
        
        if labeled_set_raw is not None:
            labeled_set_raw = labeled_set_raw.append(labeled_set_new)
        else:
            labeled_set_raw = labeled_set_new
            
        print('labeled_set_raw ' + str(labeled_set_raw.shape[0]))
        print('labeled_set_new ' + str(labeled_set_new.shape[0]))
        number_labeled_examples += low_conf_pairs_true.shape[0] + low_conf_pairs_false.shape[0]

        if data_augmentation:
            
            if da_threshold == 0:
                # Select k pairs with lowest entropy for data augmentation
                high_conf_pairs_true = predictions_true['entropy'].nsmallest(int(sampling_size/2))
                high_conf_pairs_false = predictions_false['entropy'].nsmallest(int(sampling_size/2))
            else:
                # select pairs with high probability for data augmentation
                high_conf_pairs_true = predictions[predictions['match_score']>=da_threshold]
                # select pairs with low probability for data augmentation
                high_conf_pairs_false = predictions[predictions['match_score']<=(1-da_threshold)]
                print("Data Augmentation True: " + str(high_conf_pairs_true.shape[0]))
                print("Data Augmentation False: " + str(high_conf_pairs_false.shape[0]))
                
            # Use prediction as label
            data_augmentation_true = pool_data[pool_data['id'].isin(high_conf_pairs_true.index.tolist())]
            data_augmentation_true['label'] = 1
            data_augmentation_false = pool_data[pool_data['id'].isin(high_conf_pairs_false.index.tolist())]
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

        #remove labeled pairs from unlabeled pool
        #pool_data =
        #pool_data[~pool_data['id'].isin(labeled_set_raw['id'].tolist())]
        pool_data = pool_data[~pool_data['id'].isin(labeled_set_new['id'].tolist())]
        
        # calculate positive negative ratio
        pn_ratio = round_int((labeled_set_temp['label'].shape[0] - labeled_set_temp['label'].sum()) / labeled_set_temp['label'].sum())

        print('Positve Negative Ratio: ' + str(pn_ratio))
        if pn_ratio == 0:
            pn_ratio = 1

        # split labeled set into training and validation
        if split_validation == False:

            # process labeled set for deepmatcher
            #labeled_set_temp.to_csv('labeled_set', index=False)
            pass

        else:

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

        if tl_weights == None:
            sample_weights = None
        else:
            if tl_weights == 'calc':
                source_weight = labeled_set_temp.shape[0] / (train_data_tl.shape[0] + labeled_set_temp.shape[0])
                target_weight = train_data_tl.shape[0] / (train_data_tl.shape[0] + labeled_set_temp.shape[0])
            else:
                source_weight = tl_weights[0]
                target_weight = tl_weights[1]

            #source data weights
            source_weights = dict(zip(train_data_tl['id'], [[source_weight]] * train_data_tl.shape[0]))
            #target data weights
            target_weights = dict(zip(labeled_set_temp['id'], [[target_weight]] * labeled_set_temp.shape[0]))
            #create dictionary with sample weights
            sample_weights = dict(source_weights)
            sample_weights.update(target_weights)
            print('Source Weight: ' + str(source_weight))
            print('Target Weight: ' + str(target_weight))

        if include_tl_data:
            print("Labeled set size: " + str(labeled_set_temp.shape[0]))
            print("Labeled set size: " + str(labeled_set_temp.shape[0]))
            # TL: if source data is included
            labeled_set_temp = train_data_tl.append(labeled_set_temp)
            # reorder columns
            labeled_set_temp = labeled_set_temp[test_data.columns.tolist()]
            print("Columns combined set: " + str(labeled_set_temp.columns))
            print("Size combined set: " + str(labeled_set_temp.shape[0]))

        labeled_set_temp.to_csv('labeled_set', index=False)

        # ignore columns only in first iteration
        if i == 1 and init_method == 'Transfer Learning':
            ignore_columns=('source_id','target_id')

        labeled_set,validation_set,test_set = dm.data.process(
            path='',
            train='labeled_set',
            validation='validation_set',
            test='test_set',
            ignore_columns=ignore_columns,
            left_prefix='left_',
            right_prefix='right_',
            label_attr='label',
            id_attr='id',
            cache=None,
            embeddings=embeddings)

        # new model in each iteration
        if keep_model == False:
            model = dm.MatchingModel(attr_summarizer=attr_summarizer, attr_comparator=attr_comparator)
        else:
            model._reset_embeddings(labeled_set.vocabs)
        
        # new optimizer in each iteration
        optimizer = dm.optim.Optimizer(lr_decay=lr_decay)

        # Train model on labeled set 
        model.run_train(
            labeled_set,
            validation_set,
            epochs=train_epochs, 
            batch_size=train_batch_size,
            best_save_path=path_al_model,
            optimizer=optimizer,
            pos_neg_ratio=pn_ratio,
            sample_weights=sample_weights)
            
        print("Size labeled set " + str(labeled_set_raw.shape[0]))
        print("Size unlabeled pool " + str(pool_data.shape[0]))
        print("Size labeled set temp " + str(labeled_set_temp.shape[0]))
        print("Size labeled set new " + str(labeled_set_new.shape[0]))
        if split_validation == True:
            print("Size validation set temp " + str(validation_set_temp.shape[0]))

        # Save results
        # todo
        prec, recall, fscore, _ = precision_recall_fscore_support(
            test_data['label'],
            np.where(model.run_prediction(test_set)['match_score'] >= 0.5, 1, 0),
            average='binary')

        f1_scores.append(round(fscore,3))
        precision_scores.append(round(prec,3))
        recall_scores.append(round(recall,3))
        labeled_set_size.append(number_labeled_examples)
        pos_neg_ratios.append(pn_ratio)
        if include_tl_data:
            all_sample_weights.append([source_weight,target_weight])
        

    all_scores = pd.DataFrame(
        {'labeled set size': labeled_set_size,
        'f1': f1_scores,
        'precision': precision_scores,
        'recall': recall_scores,
        'pos_neg_ratio': pos_neg_ratios
        })
 
    if data_augmentation:
        all_scores['da labels']=data_augmentation_labels

    if include_tl_data:
        all_scores['sample weights']=all_sample_weights

    return all_scores
        # Go to (1)

    