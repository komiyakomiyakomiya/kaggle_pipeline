# %%
import pdb
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split


class Valid(object):
    def __init__(self):
        pass

    def hold_out(self, model_wrapper, train_x, train_y, test_x):
        tr_x, va_x, tr_y, va_y = train_test_split(train_x,
                                                  train_y,
                                                  test_size=0.25,
                                                  shuffle=True,
                                                  random_state=71)
        model_wrapper.fit(tr_x, tr_y, va_x, va_y)
        model_wrapper.save()
        pred_hold_out = model_wrapper.predict(train_x)
        pred_test = model_wrapper.predict(test_x)
        return pred_hold_out, pred_test

    def k_fold(self, model_wrapper, train_x, train_y, test_x):
        preds_list_valid = []
        preds_list_test = []
        index_list_valid = []

        kf = KFold(n_splits=4, shuffle=True, random_state=71)

        for i, (tr_idx, va_idx) in enumerate(kf.split(train_x, train_y), 1):
            print('fold{} start'.format(i))
            tr_x = train_x.iloc[tr_idx]
            tr_y = train_y.iloc[tr_idx]
            va_x = train_x.iloc[va_idx]
            va_y = train_y.iloc[va_idx]

            model_wrapper.fit(tr_x, tr_y, va_x, va_y)

            pred_valid = model_wrapper.predict(va_x)
            preds_list_valid.append(pred_valid)
            pred_test = model_wrapper.predict(test_x)
            preds_list_test.append(pred_test)
            index_list_valid.append(va_idx)
            print('fold{} end\n'.format(i))

        index_list_valid = np.concatenate(index_list_valid, axis=0)
        preds_list_valid = np.concatenate(preds_list_valid, axis=0)
        order = np.argsort(index_list_valid)
        pred_cv = preds_list_valid[order]
        pred_test_mean = np.mean(preds_list_test, axis=0)
        # trは特徴量にするのでmeanする必要ない
        return pred_cv, pred_test_mean

    def stratified_k_fold(self, model_wrapper, train_x, train_y, test_x):
        preds_list_valid = []
        preds_list_test = []
        index_list_valid = []

        kf = StratifiedKFold(n_splits=4, shuffle=True, random_state=71)

        for i, (tr_idx, va_idx) in enumerate(kf.split(train_x, train_y), 1):
            print('fold{} start'.format(i))
            tr_x = train_x.iloc[tr_idx]
            tr_y = train_y.iloc[tr_idx]
            va_x = train_x.iloc[va_idx]
            va_y = train_y.iloc[va_idx]

            model_wrapper.fit(tr_x, tr_y, va_x, va_y)

            pred_valid = model_wrapper.predict(va_x)
            preds_list_valid.append(pred_valid)
            pred_test = model_wrapper.predict(test_x)
            preds_list_test.append(pred_test)
            index_list_valid.append(va_idx)
            print('fold{} end\n'.format(i))

        index_list_valid = np.concatenate(index_list_valid, axis=0)
        preds_list_valid = np.concatenate(preds_list_valid, axis=0)
        order = np.argsort(index_list_valid)
        pred_cv = preds_list_valid[order]
        pred_test_mean = np.mean(preds_list_test, axis=0)
        # trは特徴量にするのでmeanする必要ない
        return pred_cv, pred_test_mean

    def group_k_fold(self, model_wrapper, train_x, train_y, test_x, group_col):
        preds_list_valid = []
        preds_list_test = []
        index_list_valid = []

        # KFoldクラスを用いる
        # GroupKFoldクラスはshuffleとrandom_stateが指定できないため
        kf = KFold(n_splits=2, shuffle=True, random_state=71)
        # group_colの値の種類で分割
        group_col = train_x[group_col]
        unique_values = group_col.unique()

        print(kf.split(unique_values))

        for i, (tr_idx_group, va_idx_group) in enumerate(kf.split(unique_values), 1):
            print('fold{} start'.format(i))

            tr_groups = unique_values[tr_idx_group]
            va_groups = unique_values[va_idx_group]

            is_tr = group_col.isin(tr_groups)
            is_va = group_col.isin(va_groups)

            tr_x = train_x[is_tr]
            tr_y = train_y[is_tr]
            va_x = train_x[is_va]
            va_y = train_y[is_va]

            model_wrapper.fit(tr_x, tr_y, va_x, va_y)

            pred_valid = model_wrapper.predict(va_x)
            preds_list_valid.append(pred_valid)
            pred_test = model_wrapper.predict(test_x)
            preds_list_test.append(pred_test)
            index_list_valid.append(va_x.index)
            print('fold{} end\n'.format(i))

        index_list_valid = np.concatenate(index_list_valid, axis=0)
        preds_list_valid = np.concatenate(preds_list_valid, axis=0)
        order = np.argsort(index_list_valid)
        pred_cv = preds_list_valid[order]
        pred_test_mean = np.mean(preds_list_test, axis=0)
        # trは特徴量にするのでmeanする必要ない

        return pred_cv, pred_test_mean


# %%
