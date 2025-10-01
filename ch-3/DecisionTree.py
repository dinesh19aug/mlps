import numpy as np
from matplotlib import pyplot as plt
from  sklearn.tree import DecisionTreeClassifier
from common.common import load_data, plot_decision


class DecisionTree:
    def __init__(self, max_depth, random_state=1, criterion='gini'):
        self.max_depth = max_depth
        self.random_state = random_state
        self.criterion = criterion

    def get_model(self)-> DecisionTreeClassifier:
        return DecisionTreeClassifier(max_depth=self.max_depth,
                                      random_state=self.random_state,
                                      criterion=self.criterion)



tree_model = DecisionTree(max_depth=4).get_model()
X_train, X_test, y_train, y_test = load_data()
tree_model.fit(X_train, y_train)

print(f'Accuracy of Decision Tree classifier on training set: {tree_model.score(X_train, y_train)}')
print(f'Accuracy of Decision Tree classifier on test set: {tree_model.score(X_test, y_test)}')

X_combined_std = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))

plot_decision(X=X_combined_std, y=y_combined,
              classifier=tree_model, test_idx=range(105, 150))

plt.xlabel('petal length [cm]')
plt.ylabel('petal width [cm]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()


## Show tree
from sklearn import tree
feature_names = ['Sepal length', 'Sepal width',
'Petal length', 'Petal width']
tree.plot_tree(tree_model,
feature_names=feature_names,
filled=True)
plt.show()
