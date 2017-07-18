
import torch


class PCA:
    @staticmethod
    def fit(data, ratio=0.9):
        '''

        :param data: FloatTensor
        :return:
        '''
        X = torch.FloatTensor(data)
        u, s, v = X.svd()

        s = s / s.sum()
        sigma = s.cumsum(0)
        index = next(idx for idx, value in enumerate(sigma) if value > ratio) + 1

        sigma = torch.diag(s[:index])
        return torch.mm(u[:, 0:index], sigma)


from sklearn import datasets
import matplotlib.pyplot as plt
import seaborn as sns

iris_dataset = datasets.load_iris()

pca = PCA.fit(iris_dataset.data, 0.9).numpy()

color_mapping = {0: sns.xkcd_rgb['bright purple'], 1: sns.xkcd_rgb['lime'], 2: sns.xkcd_rgb['ochre']}
colors = list(map(lambda x: color_mapping[x], iris_dataset.target))

plt.scatter(pca[:, 0], pca[:, 1], c=colors)


plt.savefig("pca.png")
