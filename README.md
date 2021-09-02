# Combining Deep Learning and Active Learning for Entity Resolution

This repository contains the code for the master thesis "Combining Deep Learning and Active Learning for Entity".

## Deepmatcher

#### Installation

To run the experiments, the Deepmatcher package needs to be installed. It is recommended to install the adjusted version directly from GitHub, which contains changes to enable transfer learning with weighted instances.

```shell
pip install --upgrade --force-reinstall -e git+https://github.com/s-waitz/deepmatcher.git@master#egg=deepmatcher
```

Further required packages are listed in "master-thesis-code/requirements.txt".
Copy the files run.py and active_learning.py from the folder "master-thesis-code/code/" to the working directory and import the functions.
```shell
import run
from run import *
```
#### Experiments

To reproduce the Deepmatcher experiments, please use the notebooks in the folder "master-thesis-code/notebooks/experiments_deepmatcher/". There is one notebook per dataset and one notebook for the baseline experiments. The following code snippet shows an example for the method al_tl in the setting DM3:

```shell
run_al(dataset='dbpedia_viaf',
       num_runs=3,
       al_iterations=20,
       sampling_size=100,
       epochs=20,
       batch_size=20,
       split_validation=True,
       attr_summarizer='rnn',
       save_results_path='output_path_for_results/',
       transfer_learning_dataset='dbpedia_dnb',
       data_augmentation=False,
       high_conf_to_ls=False,
       keep_model=True,
       file_path='input_path_with_datasets/',
       ignore_columns=('source_id','target_id'))
```
Both datasets (source and target), in this case dbpedia_viaf and dbpedia_dnb, have to be in the same folder which is passed to the function via file_path. The datasets in the folder "master-thesis-code/data/" are already prepared for the use with deepmatcher. The parameter settings for all other methods and datasets can be taken from the respective experiment notebooks and from the file run.py. To reproduce the experiments with the settings DM1 and DM2, following parameter changes are necessary:
DM1: keep_model = False, split_validation=False; DM2: keep_model = False, split_validation=True

## Ditto

#### Installation

It is recommended to copy the files run_ditto.py, active_learning_ditto.py and ditto_helper.py into the directory "/ditto_changed/ditto/" and to start the experiments directly from this directory.
To install the required packages for Ditto, the developers recommend the following:
```shell
conda install -c conda-forge nvidia-apex
pip install -r requirements.txt
python -m spacy download en_core_web_lg
```
However, during the experiments of this thesis this did not work, and the following commands had to be used:

```shell
pip install -r requirements.txt
python -m spacy download en_core_web_lg
import nltk
nltk.download('stopwords')
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
pip uninstall apex
cd apex
rm -rf build (if it exists)
python setup.py install --cuda_ext --cpp_ext
```
I recommend to start with the first approach, and if it doesn't work, try the second approach.

Further recquired packages are listed in "master-thesis-code/requirements.txt".
Before starting the experiments, each task has to be configured in configs.json. The tasks from the master thesis are already defined here, but the data paths need to be adjusted.
```shell
  {
    "name": "dbpedia_viaf",
    "task_type": "classification",
    "vocab": ["0", "1"],
    "trainset": "/path/dbpedia_viaf_train.txt",
    "validset": "/path/dbpedia_viaf_validation.txt",
    "testset": "/path/dbpedia_viaf_test.txt"
  }
  ```
#### Experiments

To reproduce the Ditto experiments please use the notebooks in the folder "/notebooks/experiments_ditto/". There is one notebook per dataset and one notebook for the baseline experiments. The following code snippet shows an example for the method al_tl in the setting DM3:

```shell
run_al_ditto(task='dbpedia_viaf',
            num_runs=3,
            al_iterations=20,
            sampling_size=100,
            epochs=60,
            epochs_tl=50,
            reduce_epochs=[1000,20],
            batch_size=20,
            learning_model='roberta',
            save_results_path='output_path_for_results/',
            base_data_path='input_path_with_datasets/',
            labeled_set_path='path_from_configs.json/',
            transfer_learning_dataset='dbpedia_dnb',
            data_augmentation=False,
            high_conf_to_ls=False,
            da='span_del',
            dk='general',
            keep_model=False,               
            su=False,
            balance=True,
            verbose=True)
```
The functions use the same input data as Deepmatcher (base_data_path should be identical to file_path for Deepmatcher). Before running the experiments, the files (train, validation and test) should also be converted into the ditto input format with the following command from ditto_helper.py:
```shell
to_ditto_format(input_file, output_file)
```
The "output_file" has to be identical with the path that is defined in configs.json.
The parameter labeled_set_path should be set to  the path that is defined in configs.json and contain the train, validation and test set for the respective task. The parameter settings for all other methods and datasets can be taken from the respective experiment notebooks and from the file run_ditto.py.
Further information about Ditto can be found here: https://github.com/megagonlabs/ditto