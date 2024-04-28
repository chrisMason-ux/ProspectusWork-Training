import os

# import sys
from dataclasses import dataclass, field
import json
# import logging
import os
from typing import Optional
# import re
# from sklearn import preprocessing
import numpy as np
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoConfig,
    Trainer,
    EvalPrediction,
    set_seed
)
from transformers.training_args import TrainingArguments

from multimodal_transformers.data import load_data_from_folder
from multimodal_transformers.model import TabularConfig, AutoModelWithTabular

from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, \
    confusion_matrix, precision_recall_curve
from scipy.special import softmax

from tensorflow.python.ops.math_ops import arg_max  # This is being depreciated

import result_processsing as rp

# ************************** Imports **************************


os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
FULL_DATASET = "final_dataset.csv"


def filter_dataset(filters):
    """ Applies filters to the dataset """
    df = pd.read_csv(FULL_DATASET)

    # Filter rows where 'validity' is False, this is done because we don't want to remove our True statements from
    # The dataset
    validity_false_df = df[df['validity'] == False]

    # Apply other filters only on rows where 'validity' is False
    for column, filter_list in filters.items():
        for value, include in filter_list:
            if include:
                validity_false_df = validity_false_df[validity_false_df[column] == value]
            else:
                validity_false_df = validity_false_df[validity_false_df[column] != value]

    # Concatenate the filtered rows with rows where 'validity' is True
    filtered_df = pd.concat([validity_false_df, df[df['validity'] == True]])

    # Save the filtered data to a new CSV file
    filtered_df.to_csv("in_use_dataset.csv",
                       index=False)  # Set index=False to exclude row indices in the output CSV


def run_test(filters, category_filter):
    # Filters:
    #
    # filters = {'creator': [('human', True), ('chatgpt', True)]}
    # category_filter = ['numbers']

    ###########################################

    dataset_creator_filter = filters

    #                  ['True_fleshes', 'True_smog_index',
    #                 'True_flesch_kincaid_grade', 'True_coleman_liau_index',
    #                 'True_automated_readability_index', 'True_dale_chall_readability_score',
    #                 'True_difficult_words', 'True_linsear_write_formula',
    #                 'True_difficult_words.1', 'True_gunning_fog',
    #                 'characters', 'words', 'avg_word_length', 'numbers',
    #                 'unique_word_ratio', 'sentiment_index', 'hatred_index', 'support',
    #                 'opposed', 'neutral']


    filter_dataset(filters)


    # Test Settings

    dataset_in_use = "in_use_dataset.csv"

    numerical_inputs = ['True_fleshes', 'True_smog_index',
                        'True_flesch_kincaid_grade', 'True_coleman_liau_index',
                        'True_automated_readability_index', 'True_dale_chall_readability_score',
                        'True_difficult_words', 'True_linsear_write_formula',
                        'True_difficult_words.1', 'True_gunning_fog',
                        'characters', 'words', 'avg_word_length', 'numbers',
                        'unique_word_ratio', 'sentiment_index', 'hatred_index', 'support',
                        'opposed', 'neutral']

    numerical_inputs = list(set(numerical_inputs) - set(category_filter))  # This filters out the columns

    cat_inputs = ['True_text_standard']
    # cat_inputs = ['creator', 'True_text_standard']
    text_inputs = ['statement']

    # logging.basicConfig(level=logging.INFO)
    os.environ['COMET_MODE'] = 'DISABLED'

    # Initial Variable Creation

    data_df = pd.read_csv(dataset_in_use)
    data_df.dropna(inplace=True)
    data_df['Label'] = data_df['validity']
    data_df['Label'] = data_df['Label'].astype(int)
    train_df, val_df, test_df = np.split(data_df.sample(frac=1), [int(.8 * len(data_df)), int(.9 * len(data_df))])

    # Informational Logging

    print(len(data_df['Label']))
    print(data_df['Label'].value_counts())

    print('Num examples train-val-test')
    print(len(train_df), len(val_df), len(test_df))

    # Load Columns / Categories

    numerical_cols = numerical_inputs

    train_numerical_cols = []

    for item in numerical_cols:
        train_numerical_cols.append([train_df[item].values])

    val_numerical_cols = []

    for item in numerical_cols:
        val_numerical_cols.append([val_df[item].values])

    test_numerical_cols = []

    for item in numerical_cols:
        test_numerical_cols.append([test_df[item].values])

    """ Creation of random csv files, if this is not done, the code will error out"""
    train_df.to_csv('train.csv')
    val_df.to_csv('val.csv')
    test_df.to_csv('test.csv')

    # Set-Up Columns
    def return_list(x, numerical_inputs):
        """ This is used below to create lists needed for np.vstack functions"""
        test = []

        for i in range(len(numerical_inputs)):
            test.append(x[i])

        return x

    x = np.vstack(train_numerical_cols)
    train_numerical_cols = np.column_stack(return_list(x, numerical_inputs))

    x = np.vstack(val_numerical_cols)
    val_numerical_cols = np.column_stack(return_list(x, numerical_inputs))

    x = np.vstack(test_numerical_cols)
    test_numerical_cols = np.column_stack(return_list(x, numerical_inputs))

    # Creating Relevant Classes

    @dataclass
    class ModelArguments:
        """
      Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
      """

        model_name_or_path: str = field(
            metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
        )
        config_name: Optional[str] = field(
            default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
        )
        tokenizer_name: Optional[str] = field(
            default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
        )
        cache_dir: Optional[str] = field(
            default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
        )

    @dataclass
    class MultimodalDataTrainingArguments:
        """
      Arguments pertaining to how we combine tabular features
      Using `HfArgumentParser` we can turn this class
      into argparse arguments to be able to specify them on
      the command line.
      """

        data_path: str = field(metadata={
            'help': 'the path to the csv file containing the dataset'
        })
        column_info_path: str = field(
            default=None,
            metadata={
                'help': 'the path to the json file detailing which columns are text, categorical, numerical, and the label'
            })

        column_info: dict = field(
            default=None,
            metadata={
                'help': 'a dict referencing the text, categorical, numerical, and label columns'
                        'its keys are text_cols, num_cols, cat_cols, and label_col'
            })

        categorical_encode_type: str = field(default='ohe',
                                             metadata={
                                                 'help': 'sklearn encoder to use for categorical data',
                                                 'choices': ['ohe', 'binary', 'label', 'none']
                                             })
        numerical_transformer_method: str = field(default='yeo_johnson',
                                                  metadata={
                                                      'help': 'sklearn numerical transformer to preprocess numerical data',
                                                      'choices': ['yeo_johnson', 'box_cox', 'quantile_normal', 'none']
                                                  })
        task: str = field(default="classification",
                          metadata={
                              "help": "The downstream training task",
                              "choices": ["classification", "regression"]
                          })

        mlp_division: int = field(default=4,
                                  metadata={
                                      'help': 'the ratio of the number of '
                                              'hidden dims in a current layer to the next MLP layer'
                                  })
        combine_feat_method: str = field(default='individual_mlps_on_cat_and_numerical_feats_then_concat',
                                         metadata={
                                             'help': 'method to combine categorical and numerical features, '
                                                     'see README for all the method'
                                         })
        mlp_dropout: float = field(default=0.1,
                                   metadata={
                                       'help': 'dropout ratio used for MLP layers'
                                   })
        numerical_bn: bool = field(default=True,
                                   metadata={
                                       'help': 'whether to use batchnorm on numerical features'
                                   })
        use_simple_classifier: str = field(default=True,
                                           metadata={
                                               'help': 'whether to use single layer or MLP as final classifier'
                                           })
        mlp_act: str = field(default='relu',
                             metadata={
                                 'help': 'the activation function to use for finetuning layers',
                                 'choices': ['relu', 'prelu', 'sigmoid', 'tanh', 'linear']
                             })
        gating_beta: float = field(default=0.2,
                                   metadata={
                                       'help': "the beta hyperparameters used for gating tabular data "
                                               "see https://www.aclweb.org/anthology/2020.acl-main.214.pdf"
                                   })

        def __post_init__(self):
            assert self.column_info != self.column_info_path
            if self.column_info is None and self.column_info_path:
                with open(self.column_info_path, 'r') as f:
                    self.column_info = json.load(f)

    # Column Initializations

    text_cols = text_inputs
    cat_cols = cat_inputs
    numerical_cols = numerical_cols

    column_info_dict = {
        'text_cols': text_cols,
        'num_cols': numerical_cols,
        'cat_cols': cat_cols,
        'label_col': 'Label',
        'label_list': [1, 0]
    }

    model_args = ModelArguments(
        model_name_or_path='Davlan/bert-base-multilingual-cased-ner-hrl'
    )

    data_args = MultimodalDataTrainingArguments(
        data_path='',  # Removed "/content/"
        combine_feat_method='mlp_on_concatenated_cat_and_numerical_feats_then_concat',
        column_info=column_info_dict,
        task='classification'
    )

    # Refer to the following link to modify (combine_feat_method)
    # https://github.com/georgian-io/Multimodal-Toolkit

    # Training Settings

    training_args = TrainingArguments(
        output_dir="/content/logs/model_name",
        logging_dir="/content/logs/runs",
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        per_device_train_batch_size=50,
        num_train_epochs=10,
        evaluation_strategy='epoch',
        logging_steps=20,
        eval_steps=20
    )

    set_seed(training_args.seed)

    # training_args = TrainingArguments(
    #     output_dir="/content/logs/model_name",
    #     logging_dir="/content/logs/runs",
    #     overwrite_output_dir=True,
    #     do_train=True,
    #     do_eval=True,
    #     per_device_train_batch_size=50,
    #     num_train_epochs=10,
    #     evaluation_strategy='epoch',
    #     logging_steps=20,
    #     eval_steps=20
    # )
    #
    # set_seed(training_args.seed)

    # Tokenizer Set-up

    tokenizer_path_or_name = model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path
    print('Specified tokenizer: ', tokenizer_path_or_name)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path_or_name,
        cache_dir=model_args.cache_dir,
    )

    # Dataset Loading

    # Get Datasets
    train_dataset, val_dataset, test_dataset = load_data_from_folder(
        data_args.data_path,
        data_args.column_info['text_cols'],
        tokenizer,
        label_col=data_args.column_info['label_col'],
        label_list=data_args.column_info['label_list'],
        categorical_cols=data_args.column_info['cat_cols'],
        numerical_cols=data_args.column_info['num_cols'],
        sep_text_token_str=tokenizer.sep_token,
    )

    # Continued

    train_dataset.labels = train_df['Label'].values
    test_dataset.labels = test_df['Label'].values
    val_dataset.labels = val_df['Label'].values

    # Logging

    val_df['Label'].value_counts()

    num_labels = len(np.unique(train_dataset.labels))
    print(num_labels)

    # Configuration

    train_dataset.numerical_feats = train_numerical_cols
    val_dataset.numerical_feats = val_numerical_cols
    test_dataset.numerical_feats = test_numerical_cols

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    tabular_config = TabularConfig(num_labels=num_labels,
                                   cat_feat_dim=train_dataset.cat_feats.shape[1],
                                   numerical_feat_dim=train_dataset.numerical_feats.shape[1],
                                   **vars(data_args))
    config.tabular_config = tabular_config

    # Model Set-Up

    model = AutoModelWithTabular.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir
    )

    # Method Creations
    def compute_metrics_mycode(p: EvalPrediction):
        y_true = p.label_ids
        y_prob = softmax(list(p.predictions[0]), axis=1)[:, 1]

        # Calculate ROC AUC
        roc_auc = roc_auc_score(y_true, y_prob)

        # Calculate PR AUC
        pr_auc = average_precision_score(y_true, y_prob)

        # Calculate ROC curve and find the best threshold
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        optimal_idx = np.argmax(tpr - fpr)
        threshold_roc_auc = thresholds[optimal_idx]

        # Calculate Precision-Recall curve and find the best threshold
        precision, recall, thresholds_pr = precision_recall_curve(y_true, y_prob)
        optimal_idx_pr = np.argmax(2 * precision * recall / (precision + recall + 1e-9))
        threshold_pr_auc = thresholds_pr[optimal_idx_pr]

        # Calculate confusion matrix
        y_pred = (y_prob >= threshold_roc_auc).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        # Calculate recall, precision, and F1 score
        recall_score = tp / (tp + fn)
        precision_score = tp / (tp + fp)
        f1_score = 2 * (precision_score * recall_score) / (precision_score + recall_score + 1e-9)

        result = {'roc_auc': roc_auc,
                  'threshold': threshold_roc_auc,
                  'pr_auc': pr_auc,
                  'recall': recall_score,
                  'precision': precision_score, 'f1': f1_score,
                  'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp
                  }

        return result

    # Trainer Set-Up

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics_mycode)

    # Program

    trainer.train()

    # Data Collection

    # Actual Labels
    actual = test_dataset.labels
    actual_len = len(test_dataset.labels)

    # Continued

    pred = trainer.predict(test_dataset)
    predc = pred.predictions
    predc = arg_max(predc[0], dimension=1)

    # Data Creation

    read_test = pd.read_csv("test.csv", encoding='utf-8-sig')  # Removed /content/
    read_test["pred_label"] = predc
    read_test.to_csv('results.csv', encoding='utf-8-sig')

    # Continued

    def calculate_accuracy(array1, array2):
        if len(array1) != len(array2):
            return -1
        correct_predictions = 0
        for i in range(len(array1)):
            if array1[i] == array2[i]:
                correct_predictions += 1
        accuracy = correct_predictions / len(array1) * 100
        return accuracy

    acc = calculate_accuracy(actual, predc)
    if acc == -1:
        print("The arrays have different length")
    else:
        print(f'the Test_ACC = {acc}')

    # Continued

    """ Use this code to see the statements used """
    # test_data = pd.read_csv('test.csv')

    eval_results = trainer.evaluate(test_dataset)

    # Print the evaluation results
    print(eval_results)

    # This processes and stores the results into a file called "complete_result.csv"
    rp.add_results(read_test, dataset_creator_filter, category_filter)


""" List of Colummns """
#                  ['True_fleshes', 'True_smog_index',
#                 'True_flesch_kincaid_grade', 'True_coleman_liau_index',
#                 'True_automated_readability_index', 'True_dale_chall_readability_score',
#                 'True_difficult_words', 'True_linsear_write_formula',
#                 'True_difficult_words.1', 'True_gunning_fog',
#                 'characters', 'words', 'avg_word_length', 'numbers',
#                 'unique_word_ratio', 'sentiment_index', 'hatred_index', 'support',
#                 'opposed', 'neutral']

""" Example Filters """
# filter = {'creator': [('human', False), ('chatgpt', False)]}
#        -- This controls for rows where creator != human and creator != chatgpt

# filter = {'creator': [('human', True)]}
#        -- This controls for rows where creator == human

# category_filter = ['numbers']
#       -- This removes the numbers column from use

if __name__ == "__main__":

    filters = [{'creator': [('chatgpt', True)]}
               ]

    category_filters = [['numbers', 'avg_word_length', 'words', 'characters', 'unique_word_ratio',
                         'support', 'opposed', 'True_automated_readability_index',
                         'True_dale_chall_readability_score']]

    for filter_1 in filters:
        for filter_2 in category_filters:
            run_test(filter_1, filter_2)
