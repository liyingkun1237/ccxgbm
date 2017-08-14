os.getcwd()
os.chdir(r'C:\Users\liyin\Desktop\20170620_tn\0620_base')

train = pd.read_csv('train_base14.csv')
test = pd.read_csv('test_base14.csv')

x_col = [x for x in train.columns if x not in ['target', 'contract_id', 'tg_location1']]
y_col = 'target'
lg_train = lgb.Dataset(train[x_col], train[y_col].values)
lg_test = lgb.Dataset(test[x_col], test[y_col].values)

# 网格搜索
param_grid = {

    'num_leaves': [31, 63],
    'learning_rate': [0.1],
    'min_split_gain': [2],
    'min_child_weight': [5],

    'subsample': [0.9, 0.5],
    'colsample_bytree': [0.9, 0.5],
    'reg_lambda': [3000, 2000, 10000],
}

estimator = lgb.LGBMClassifier(n_estimators=300)

gbm = GridSearchCV(estimator, param_grid, cv=5, n_jobs=1, scoring='roc_auc', verbose=10)

gbm.fit(train[x_col], train[y_col])

re = gbm.cv_results_
dd = pd.DataFrame(re)

ddd = dd[['mean_test_score', 'mean_train_score', 'params', 'rank_test_score']]
ddd

dd.to_csv('lightgbm_param.csv', index=False)

gbm.scorer_
gbm.best_score_
gbm.best_index_

params = gbm.best_params_

#####
bst = lgb.train(params,
                lg_train,
                num_boost_round=300,
                valid_sets=[lg_test, lg_train],  # eval training data
                valid_names=['test', 'train'],

                )

dir(bst)

# 重要变量
xx = pd.DataFrame({'Feature_Name': bst.feature_name(),
                   'gain': bst.feature_importance(importance_type='gain')})

xx.sort('gain', ascending=False)
xx.query('gain >0')

##模型预测
pd.Series(bst.predict(test[x_col])).describe()
import matplotlib.pyplot as plt
import numpy as np


def plot_ks_line(y_true, y_pred, title='ks-line', detail=False):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)

    plt.plot(tpr, label='tpr-line')
    plt.plot(fpr, label='fpr-line')
    plt.plot(tpr - fpr, label='KS-line')
    # 设置x的坐标轴为0-1范围
    plt.xticks(np.arange(0, len(tpr), len(tpr) // 10), np.arange(0, 1.1, 0.1))

    # 添加标注
    x0 = np.argmax(tpr - fpr)
    y0 = np.max(tpr - fpr)
    plt.scatter(x0, y0, color='black')  # 显示一个点
    z0 = thresholds[x0]  # ks值对应的阈值
    plt.text(x0 - 2, y0 - 0.12, ('(ks: %.4f,\n th: %.4f)' % (y0, z0)))

    if detail:
        # plt.plot([x0,x0],[0,y0],'b--',label=('thresholds=%.4f'%z0)) #在点到x轴画出垂直线
        # plt.plot([0,x0],[y0,y0],'r--',label=('ks=%.4f'%y0)) #在点到y轴画出垂直线
        plt.plot(thresholds[1:], label='thresholds')
        t0 = thresholds[np.argmin(np.abs(thresholds - 0.5))]
        t1 = list(thresholds).index(t0)
        plt.scatter(t1, t0, color='black')
        plt.plot([t1, t1], [0, t0])
        plt.text(t1 + 2, t0, 'thresholds≈0.5')

        tpr0 = tpr[t1]
        plt.scatter(t1, tpr0, color='black')
        plt.text(t1 + 2, tpr0, ('tpr=%.4f' % tpr0))

        fpr0 = fpr[t1]
        plt.scatter(t1, fpr0, color='black')
        plt.text(t1 + 2, fpr0, ('fpr=%.4f' % fpr0))

    plt.legend(loc='upper left')
    plt.title(title)
    # fig_path = save_figure(plt, title)
    plt.show()
    # plt.close()
    # return fig_path


plot_ks_line(test[y_col], bst.predict(test[x_col]))

plot_ks_line(lg_test.get_label(), bst.predict(lg_test))
