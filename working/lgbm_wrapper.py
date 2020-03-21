# %%
import pdb
import pickle

import lightgbm as lgb
import pandas as pd


class LGBMWrapper(object):
    def __init__(self):
        self.model = None

    def fit(self, tr_x, tr_y, va_x, va_y):
        lgb_train = lgb.Dataset(tr_x, tr_y)
        lgb_valid = lgb.Dataset(va_x, va_y)

        params = {'objective': 'binary',
                  'seed': 71,
                  'verbose': 0}

        # params  = {
        #     'objective': 'regression',
        #     'learning_rate': 0.1, # 学習率
        #     'max_depth': -1, # 木の数 (負の値で無制限)
        #     'num_leaves': 9, # 枝葉の数
        #     'metric': ('mean_absolute_error', 'mean_squared_error', 'rmse'),
        #     'drop_rate': 0.15,
        #     'verbose': 0
        # }
        # メトリック https://lightgbm.readthedocs.io/en/latest/Parameters.html#metric-parameters

        self.model = lgb.train(params=params,
                               train_set=lgb_train,
                               valid_sets=[lgb_train, lgb_valid],
                               num_boost_round=10000,
                               early_stopping_rounds=10,  # 検証スコアが10ラウンド改善しないまでトレーニング
                               verbose_eval=-1)

        # importanceを表示する
        importance = pd.DataFrame(self.model.feature_importance(
        ), index=tr_x.columns, columns=['importance'])
        importance = importance.sort_values('importance', ascending=False)
        display(importance)

    def save(self):
        pass
        # model_dir_path = '../output/'
        # file_name = 'model_lgbm.pkl'
        # with open(model_dir_path + file_name, 'wb') as f:
        #     pickle.dump(self.model, f)

    def predict(self, x):
        pred_proba = self.model.predict(x)
        return pred_proba
