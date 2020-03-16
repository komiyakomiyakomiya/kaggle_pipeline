# %%
import os
import pickle

from google.cloud import bigquery
from IPython.display import display
import pandas as pd


# 自ファイルのディレクトリを取得
cwd = os.path.dirname(os.path.abspath(__file__))
# 環境変数に設定
key_path = f'{cwd}/./credentials.json'
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = key_path

project_id = 'kaggle-titanic-bq'
dataset = 'dlg'

client = bigquery.Client(project=project_id)
tables_obj = client.list_tables(dataset=dataset)
tables = [table.table_id for table in tables_obj]
print(tables)


df_dict = {}
for table in tables:
    sql = f"""
    select *
    from {dataset}.{table}
    """

    df = client.query(sql).to_dataframe()
    df_dict[table] = df
    display(df_dict[table])


# %%
