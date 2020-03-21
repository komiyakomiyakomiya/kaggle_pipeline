# %%
import os
from io import BytesIO
import pdb

from google.cloud import storage
import numpy as np
import pandas as pd


cwd = os.path.dirname(os.path.abspath(__file__))
key_path = '{}/../credentials.json'.format(cwd)
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = key_path


class Data(object):
    def __init__(self):
        pass

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


if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_colwidth', 800)

    data = Data()
# %%
