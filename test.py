# %%
import pandas as pd
from google.cloud import storage
from io import BytesIO


PROJECT_NAME = 'titanic-prediction'
BUCKET_NAME = 'titanic-prediction-mlengine'
FILE_NAME = 'input/train.csv'

client = storage.Client(PROJECT_NAME)
bucket = client.get_bucket(BUCKET_NAME)
blob = storage.Blob(FILE_NAME, bucket)
data = blob.download_as_string()
df = pd.read_csv(BytesIO(data))
print(df)


# %%
import os
os.makedirs('./toilet')

# %%
