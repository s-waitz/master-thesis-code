import pandas as pd
import numpy as np

from sklearn.metrics import precision_recall_fscore_support

import deepmatcher as dm

def active_learning(train_data, validation_data, test_data, num_runs, sampling_size, model, file_path='', high_conf_to_ls=True, train_epochs=20, train_batch_size=16, embeddings='fasttext.en.bin'):
    """    
        Args:
        train_data (pd.DataFrame): ...
        validation_data (pd.DataFrame): ...
        test_data (pd.DataFrame): ...
        num_runs (int): ...
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

    labeled_set_raw = None

    oracle = train_data.copy()
    pool_data = train_data.drop('label',axis=1)

    # write data to csv for later processing
    validation_data.to_csv(file_path + 'validation_set', index=False)
    test_data.to_csv(file_path + 'test_set', index=False)

    validation_set, test_set = dm.data.process(
        path=file_path,
        validation='validation_set',
        test='test_set',
        ignore_columns=('source_id', 'target_id'),
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

    # (1)
    for i in range(num_runs):


        print("AL run: " + str(i))

        # process unlabeled pool for deepmatcher
        pool_data.to_csv(file_path + 'unlabeled_pool', index=False)

        unlabeled_pool = dm.data.process_unlabeled(
            path=file_path + 'unlabeled_pool',
            trained_model=model,
            ignore_columns=('source_id', 'target_id'))

        # Predict probabilities
        predictions = model.run_prediction(unlabeled_pool)

        # Calculate entropy based on probabilities
        predictions['entropy'] = predictions['match_score'].apply(lambda p: -p * np.log(p) - (1 - p) * np.log(1 - p))

        # Split in matches and non-matches (based on probabilities)
        predictions_true = predictions[predictions['match_score']>=0.5]
        predictions_false = predictions[predictions['match_score']<0.5]

        # Select k pairs with highest entropy for active learning
        low_conf_pairs_true = predictions_true['entropy'].nlargest(int(sampling_size/2))
        low_conf_pairs_false = predictions_false['entropy'].nlargest(int(sampling_size/2))
        
        # Label these pairs with oracle and add them to labeled set
        if labeled_set_raw is not None:
            labeled_set_raw = labeled_set_raw.append(oracle[oracle['id'].isin(low_conf_pairs_true.index.tolist())])
            labeled_set_raw = labeled_set_raw.append(oracle[oracle['id'].isin(low_conf_pairs_false.index.tolist())])
        else:
            labeled_set_raw = oracle[oracle['id'].isin(low_conf_pairs_true.index.tolist())]
            labeled_set_raw = labeled_set_raw.append(oracle[oracle['id'].isin(low_conf_pairs_false.index.tolist())])

        # Select k pairs with lowest entropy for data augmentation
        high_conf_pairs_true = predictions_true['entropy'].nsmallest(int(sampling_size/2))
        high_conf_pairs_false = predictions_false['entropy'].nsmallest(int(sampling_size/2))
        
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

        #remove labeled pairs from unlabeled pool
        pool_data = pool_data[~pool_data['id'].isin(labeled_set_raw['id'].tolist())]
        
        # process labeled set for deepmatcher
        labeled_set_temp.to_csv('labeled_set', index=False)

        labeled_set = dm.data.process(
            path=file_path,
            train='labeled_set',
            ignore_columns=('source_id', 'target_id'),
            left_prefix='left_',
            right_prefix='right_',
            label_attr='label',
            id_attr='id',
            cache=None,
            embeddings=embeddings)

        # Train model on labeled set 
        model.run_train(
            labeled_set,
            validation_set,
            epochs=train_epochs, 
            batch_size=train_batch_size,
            best_save_path='dm_model.pth',
            pos_neg_ratio=3)
            
        print("Size labeled set " + str(labeled_set_raw.shape[0]))
        print("Size unlabeled pool " + str(pool_data.shape[0]))

        # Save results
        # todo
        prec, recall, fscore, support = precision_recall_fscore_support(
            test_data['label'],
            np.where(model.run_prediction(test_set)['match_score'] >= 0.5, 1, 0),
            average='binary')

        f1_scores.append(fscore)
        precision_scores.append(prec)
        recall_scores.append(recall)
        labeled_set_size.append(labeled_set_raw.shape[0])

    all_scores = pd.DataFrame(
        {'f1': f1_scores,
        'precision': precision_scores,
        'recall': recall_scores,
        'labeled set size': labeled_set_size
        })

    return all_scores
        # Go to (1)