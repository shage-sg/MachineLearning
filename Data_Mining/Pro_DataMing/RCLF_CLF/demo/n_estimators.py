superpa = []
for n_e in range(100):
    # n_jobs设定工作的core的数量等于-1时，表示cpu里的所有core进行工作
    rfc = RandomForestClassifier(n_estimators=n_e+1,n_jobs=-1)
    rfc_super = cross_val_score(rfc,X,Y,cv=10).mean()
    superpa.append(rfc_super)
#打印出：最高精确度取值，因为索引下标从零开始的
# max(superpa))+1指的是森林数目的数量n_estimators
print("Accuracy:",max(superpa))
print("n_estimators:",superpa.index(max(superpa))+1)
print("=="*30)
plt.figure(figsize=[20,5])
plt.plot(range(1,101),superpa)
plt.savefig(fname=r'./pictures/n_estimators.png',bbox_inches='tight',dpi=200)
plt.show()