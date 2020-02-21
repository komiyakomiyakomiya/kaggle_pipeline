# %%
import os
from io import BytesIO

from googletrans import Translator
from google.cloud import storage
import numpy as np
import pandas as pd


cwd = os.path.dirname(os.path.abspath(__file__))2yy
print(cwd)

# if '/Users/' in __file__:
# from dotenv import load_dotenv
# load_dotenv('{}/../.env'.format(cwd))


class Data(object):
    def __init__(self):
        pass

    def load(self):
        TRAIN_PATH = 'https://raw.githubusercontent.com/taichihaya/event/master/modeling_data.csv'
        TEST_PATH = 'https://raw.githubusercontent.com/taichihaya/event/master/competition_data.csv'

        train = pd.read_csv(TRAIN_PATH, error_bad_lines=False)
        test = pd.read_csv(TEST_PATH, error_bad_lines=False)
        return train, test

    def load_gcs(self):
        PROJECT_NAME = 'aipsample'
        BUCKET_NAME = 'aipsamplebucket'
        TRAIN = 'input/train.csv'
        TEST = 'input/test.csv'
        client = storage.Client(PROJECT_NAME)
        bucket = client.get_bucket(BUCKET_NAME)
        blob_train = storage.Blob(TRAIN, bucket)
        blob_test = storage.Blob(TEST, bucket)
        data_train = blob_train.download_as_string()
        data_test = blob_test.download_as_string()
        train = pd.read_csv(BytesIO(data_train))
        test = pd.read_csv(BytesIO(data_test))
        return train, test

    def processing(self):
        train, test = self.load()

        translator = Translator()
        translate_dict = {
            col_name: translator.translate(col_name).text for col_name in train.columns}

        train_x = train.copy()
        train_x.drop(columns=['解約', 'ID'], axis=1, inplace=True)
        # lgbm用にカラム名を英語化
        train_x.rename(columns=translate_dict, inplace=True)
        train_y = train['解約']

        test_x = test.copy()
        test_x.drop(columns=['解約', 'ID'], axis=1, inplace=True)
        # lgbm用にカラム名を英語化
        test_x.rename(columns=translate_dict, inplace=True)
        test_y = test['解約']

        return train_x, train_y, test_x, test_y

    def nn_processing(self):
        train, test = self.load()

        translator = Translator()
        translate_dict = {
            col_name: translator.translate(col_name).text for col_name in train.columns}

        train_x = train.copy()
        train_x.drop(columns=['解約', 'ID'], axis=1, inplace=True)
        # lgbm用にカラム名を英語化
        train_x.rename(columns=translate_dict, inplace=True)
        train_y = train['解約']

        test_x = test.copy()
        test_x.drop(columns=['解約', 'ID'], axis=1, inplace=True)
        # lgbm用にカラム名を英語化
        test_x.rename(columns=translate_dict, inplace=True)
        test_y = test['解約']

        return train_x, train_y, test_x, test_yx


if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_colwidth', 800)

    data = Data()
    train_x, train_y, test_x, test_y = data.processing()
    display(test_x['Payment Method'].unique())

# %%
