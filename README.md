# A controller-peripheral architecture and costly energy principle for learning

### Description
This is a complementary repo to <link> which focuses on DNN model finetuning which then used as part of the proposed architecture in this [research paper](https://www.biorxiv.org/content/10.1101/2023.01.16.524194v1).

### Structure and main components
This repo follows a fairly generic ML-project organisation:
1. `main.py` is the top executable that runs the finetuning process.
<br \> Depending on the `--mode` specified (see Example usage), the script will either be running training by first preparing datasets from raw data and
training models and saving them to disk
```python
if mode == 'train':
    dataset_dir = f'stimuli/{preprocessed_dir}/{model_name}/' \
              f'{layer}_reprs/task{stimulus_set}'
    # if no dataset, create dataset 
    if not os.path.exists(dataset_dir):
        print('[Check] Preparing dataset..')
        data.execute(config)

    print('[Check] Start training..')
    train.execute(config_version)
```
or the script will be evaluating trained models on tests
```python
elif mode == 'eval':
    evaluations.execute(
        config_version, 
        full_test, 
        heldout_test)
```

2. `models.py` contains model definitions of the candidate DNNs.
3. `train.py` contains training logics for finetuning specified models.
4. `evaluations.py` contains evaluation routines for trained models.
5. `utils.py` contains general utilites and data loading mechanisms.
6. `data.py` contains code for raw data preprocessing and dataset preparation. Prepared datasets will be saved to disk.
7. `config_generator.py` produces different model configurations in batch.
8. `keras_custom` contains customised TF/keras functionalities, specifically
```
.
├── keras_custom
│   ├── generators          # base classes with customised components of TF generators which are called by data loaders in `utils.py`
│   ├── callbacks.py        # customised TF callbacks.
│   └── regularizers.py     # customised TF regularizers.
```

### Example usage
1. Run finetuning on task2 on GPU-0
```
python main.py --mode train --task 2 --gpu 0
```

### Attribution
```
@article {Luo2023.01.16.524194,
    author = {Xiaoliang Luo and Robert M. Mok and Brett D. Roads and Bradley C. Love},
    title = {A controller-peripheral architecture and costly energy principle for learning},
    elocation-id = {2023.01.16.524194},
    year = {2023},
    doi = {10.1101/2023.01.16.524194},
    publisher = {Cold Spring Harbor Laboratory},
}
```
