param_dict = {
    'bootstrap': [True, False],
    'max_depth': np.arange(1, 200),
    'max_features': np.arange(0.1, 1, 0.1),
    'min_samples_leaf': np.arange(0.1, 0.5, 0.1),
    'min_samples_split': np.arange(0.1, 1, 0.1),
    'n_estimators': [71],
}

random_search = RandomizedSearchCV(estimator=rfc,
                                   param_distributions=param_dict,
                                   n_iter=100,
                                   cv=10,
                                   # 使用所有的CPU进行训练，默认为1，使用1个CPU
                                   n_jobs=-1,
                                   # 日志打印
                                   verbose=1)

random_search.fit(X_train, y_train)

# 搜索训练后的副产品
superpa = []
for params, score in zip(random_search.cv_results_['params'], random_search.cv_results_['mean_test_score']):
    superpa.append(score)
    logger.info('\t'.join([str(params), str(score)]))

plt.figure(figsize=[20, 5])
plt.plot(range(len(superpa)), superpa)
plt.savefig(r'./pictures/random_search.png',dpi=200,bbox_inches='tight')
plt.show()



print("==" * 30)
print(f"random_search模型的最优参数：{random_search.best_params_}")
print(f"random_search最优模型分数：{random_search.best_score_}")
print(f"random_search最优模型对象：")
pprint(random_search.best_estimator_)