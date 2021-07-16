param_grid = [
    {'bootstrap': [False],
     'max_depth': [9,10,11],
     'max_features': [0.5,0.6,0.7,0.8],
     'min_samples_leaf': [0.2,0.3,0.4],
     'min_samples_split': [0.7,0.8,0.9],
     'n_estimators': [71],
     }
]

grid_search = GridSearchCV(estimator = rfc, param_grid = param_grid,
                          cv = 10, n_jobs = -1, verbose = 1)

grid_search.fit(X_train,y_train)

# 搜索训练后的副产品
superpa = []
for params, score in zip(grid_search.cv_results_['params'], grid_search.cv_results_['mean_test_score']):
    superpa.append(score)
    logger.info('\t'.join([str(params), str(score)]))

plt.figure(figsize=[20, 5])
plt.plot(range(len(superpa)), superpa)
plt.savefig(r'./pictures/grid_search.png',dpi=80,bbox_inches='tight')
plt.show()


print("==" * 30)
print(f"grid_search模型的最优参数：{grid_search.best_params_}")
print(f"grid_search最优模型分数：{grid_search.best_score_}")
print(f"grid_search最优模型对象：")
pprint(grid_search.best_estimator_)