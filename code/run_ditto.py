import os
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd
from datetime import date
import os.path
import shlex
import subprocess

from ditto_helper import to_ditto_format, to_jsonl
from active_learning_ditto import active_learning_ditto

def run_al_ditto(task, num_runs, al_iterations, sampling_size, save_results_path, base_data_path, labeled_set_path, transfer_learning_dataset=None, init_random_sample=False, data_augmentation=False, high_conf_to_ls=False, da_threshold=0, input_path='input/', output_path='output/', batch_size=32, epochs=2):

    # Delete all models
    cmd = 'rm *.pt'
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
        
        # Initialize model
        if transfer_learning_dataset != None:

          pass

        else:
            random_train_data = train_data.sample(n=init_random_sample, weights=None, axis=None)
            to_ditto_format(random_train_data, labeled_set_path+task+'_train.txt')

            # Train model on random sample

            print('Initialize model on random sample ...')

            cmd = 'CUDA_VISIBLE_DEVICES=0'
            os.system(cmd)

            cmd = """python train_ditto.py \
              --task %s \
              --batch_size %d \
              --max_len 256 \
              --lr 3e-5 \
              --n_epochs %d \
              --finetuning \
              --lm distilbert \
              --fp16 \
              --balance \
              --save_model""" % (task, batch_size, epochs)

            #os.system(cmd)
            # invoke process
            process = subprocess.Popen(shlex.split(cmd),shell=False,stdout=subprocess.PIPE)

            # Poll process.stdout to show stdout live
            while True:
              output = process.stdout.readline()
              if process.poll() is not None:
                break
              if output:
                print(output.strip())
            rc = process.poll()

       
        results_al = active_learning_ditto(task, al_iterations, sampling_size,
                                           base_data_path, labeled_set_path,
                                           input_path, output_path,
                                           batch_size, epochs)
        # todo all results