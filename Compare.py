import matplotlib.pyplot as plt
import numpy as np

# Creating dataset
np.random.seed(10)

d1 = np.load('four_layers_model_high.npy')
d1h = np.load('four_layers_model_droplayer.npy')
d2 = np.load('four_layers_model_input.npy')
d2h = np.load('four_layers_model_recurrent.npy')

# data = [d1, d2, d3, d4,d4h,d5]

colors = ['#73020C', '#426A8C', '#D94D1A']
colors_setosa = dict(color=colors[0])
colors_versicolor = dict(color=colors[1])
colors_virginica = dict(color=colors[2])

# fig = plt.figure(figsize=(10, 7))
fig, ax = plt.subplots(figsize=(10, 7))

ax.boxplot(d1, positions=[1], sym='',boxprops=colors_virginica, medianprops=colors_virginica, whiskerprops=colors_virginica, capprops=colors_virginica, flierprops=dict(markeredgecolor=colors[2]))
ax.boxplot(d1h, positions=[2], sym='', boxprops=colors_versicolor, medianprops=colors_versicolor, whiskerprops=colors_versicolor, capprops=colors_versicolor, flierprops=dict(markeredgecolor=colors[1]))
ax.boxplot(d2, positions=[3],sym='',boxprops=colors_virginica, medianprops=colors_virginica, whiskerprops=colors_virginica, capprops=colors_virginica, flierprops=dict(markeredgecolor=colors[2]))
ax.boxplot(d2h, positions=[4], sym='', boxprops=colors_versicolor, medianprops=colors_versicolor, whiskerprops=colors_versicolor, capprops=colors_versicolor, flierprops=dict(markeredgecolor=colors[1]))
ax.boxplot(d3, positions=[5],sym='',boxprops=colors_virginica, medianprops=colors_virginica, whiskerprops=colors_virginica, capprops=colors_virginica, flierprops=dict(markeredgecolor=colors[2]))
ax.boxplot(d3h, positions=[6], sym='', boxprops=colors_versicolor, medianprops=colors_versicolor, whiskerprops=colors_versicolor, capprops=colors_versicolor, flierprops=dict(markeredgecolor=colors[1]))
ax.boxplot(d4, positions=[7],sym='',boxprops=colors_virginica, medianprops=colors_virginica, whiskerprops=colors_virginica, capprops=colors_virginica, flierprops=dict(markeredgecolor=colors[2]))
ax.boxplot(d4h, positions=[8], sym='', boxprops=colors_versicolor, medianprops=colors_versicolor, whiskerprops=colors_versicolor, capprops=colors_versicolor, flierprops=dict(markeredgecolor=colors[1]))
ax.boxplot(d5, positions=[9],sym='',boxprops=colors_virginica, medianprops=colors_virginica, whiskerprops=colors_virginica, capprops=colors_virginica, flierprops=dict(markeredgecolor=colors[2]))
ax.boxplot(d5h, positions=[10], sym='', boxprops=colors_versicolor, medianprops=colors_versicolor, whiskerprops=colors_versicolor, capprops=colors_versicolor, flierprops=dict(markeredgecolor=colors[1]))

# Creating plot
plt.title("Layers Adjustment")
plt.grid()
# plt.boxplot(data, sym='', meanline=True)
# show plot
plt.show()