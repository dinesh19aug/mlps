from matplotlib import pyplot as plt
from sklearn import datasets

import numpy as np
from sklearn.linear_model import LogisticRegression


class LogisticRegressionGD:
    def __init__(self, lr=0.01, n_iter=50, random_state=1):
        self.lr = lr
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.weight  = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.bias = float(0.)
        self.losses = []

        for i in range(self.n_iter):
            # y wx+b
            net_output = self.net_input(X)

            # Apply the sigmoid function
            output = self.activation(net_output)

            #Calculate the error
            errors = y - output
            # Adjust weights
            self.weight += self.lr * 2.0 * X.T.dot(errors)/X.shape[0]
            self.weight += self.lr * 2.0 * errors.mean()

            # update the loss
            loss = (-y.dot(np.log(output))) - ((1-y).dot(np.log(1 - output)))
            self.losses.append(loss)
        return self

    def predict(self, X):
        return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)



    def net_input(self, X):
        return np.dot(X,self.weight) + self.bias

    def activation(self, z):
        return 1/ (1.+ np.exp(-np.clip(z, -250, 250)))


def plot_decision(X, y, classifier, test_idx=None, resolution = 0.02):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = plt.cm.get_cmap('viridis')

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x_min, x_max, resolution),
                           np.arange(x2_min, x2_max, resolution))

    lab = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    lab = lab.reshape(xx1.shape)
    plt.contourf(xx1, xx2, lab, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot examples
    for idx , cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=f'Class {cl}',
                    edgecolor='black')


def plot_using_built_in(multi_class, solver, X_train_std, X_test_std,y_train, y_test):
    X_combined_std = np.vstack((X_train_std, X_test_std))
    y_combined = np.hstack((y_train, y_test))
    lrgd = LogisticRegression(C=102.0, solver=solver, multi_class=multi_class, l1_ratio=0.01, penalty='l2')
    lrgd.fit(X_train_std, y_train)
    plot_decision(X=X_combined_std, y=y_combined,
                  classifier=lrgd, test_idx=range(105, 250))

    plt.xlabel('petal length [standardized]')
    plt.ylabel('petal width [standardized]')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()


from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y)

X_train_std = StandardScaler().fit_transform(X_train)
X_test_std = StandardScaler().fit_transform(X_test)

## Taking only two features
#X_train_dataset_1 = X_train_std[np.isin(y_train, [0, 1])]
#y_train_dataset_1 = y_train[np.isin(y_train, [0, 1])]

#lrgd = LogisticRegressionGD(lr=0.3, n_iter=1000, random_state=1)
#lrgd.fit(X_train_dataset_1, y_train_dataset_1)
#plot_decision(X=X_train_dataset_1, y=y_train_dataset_1,classifier=lrgd)

#'lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'
plot_using_built_in(multi_class='ovr', solver='newton-cg', X_train_std=X_train_std, X_test_std=X_test_std, y_train=y_train, y_test=y_test)
#plot_using_built_in(multi_class='ovr', solver='lbfgs', X_train_std=X_train_std, X_test_std=X_test_std, y_train=y_train, y_test=y_test)
#plot_using_built_in(multi_class='ovr', solver='liblinear', X_train_std=X_train_std, X_test_std=X_test_std, y_train=y_train, y_test=y_test)
#plot_using_built_in(multi_class='ovr', solver='newton-cholesky', X_train_std=X_train_std, X_test_std=X_test_std, y_train=y_train, y_test=y_test)
#plot_using_built_in(multi_class='ovr', solver='sag', X_train_std=X_train_std, X_test_std=X_test_std, y_train=y_train, y_test=y_test)
#plot_using_built_in(multi_class='ovr', solver='saga', X_train_std=X_train_std, X_test_std=X_test_std, y_train=y_train, y_test=y_test)




