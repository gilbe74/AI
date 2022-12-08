import matplotlib.pyplot as plt
import numpy as np

# Creating dataset
np.random.seed(10)

t1 = np.load('da_tiny_40.npy')
v1 = np.load('da_vanilla_40.npy')
s1 = np.load('da_stacked_40.npy')
# rs1 = np.load('da_keras_stack_100.npy')



# data = [d1, d2, d3, d4,d4h,d5]

colors = ['#73020C', '#426A8C', '#D94D1A']
colors_setosa = dict(color=colors[0])
colors_versicolor = dict(color=colors[1])
colors_virginica = dict(color=colors[2])

# fig = plt.figure(figsize=(10, 7))
fig, ax = plt.subplots(figsize=(10, 7))

ax.set(
    axisbelow=True,  # Hide the grid behind plot objects
    title='Confronto finestra temporale',
    # xlabel='Distribuzione Errore',
    # ylabel='Value',
)

ax.boxplot(t1, positions=[1], sym='+',boxprops=colors_virginica, medianprops=colors_virginica, whiskerprops=colors_virginica, capprops=colors_virginica,flierprops=dict(markeredgecolor=colors[2]),labels=['Tiny'])
ax.boxplot(v1, positions=[2], sym='+', boxprops=colors_versicolor, medianprops=colors_versicolor, whiskerprops=colors_versicolor, capprops=colors_versicolor, flierprops=dict(markeredgecolor=colors[1]), labels=['Vanilla'])
ax.boxplot(s1, positions=[3],sym='+',boxprops=colors_versicolor, medianprops=colors_versicolor, whiskerprops=colors_versicolor, capprops=colors_versicolor, flierprops=dict(markeredgecolor=colors[1]), labels=['MultiStack'])
# ax.boxplot(rs1, positions=[4],sym='+', boxprops=colors_versicolor, medianprops=colors_versicolor, whiskerprops=colors_versicolor, capprops=colors_versicolor, flierprops=dict(markeredgecolor=colors[1]), labels=['ReturnState'])
# ax.boxplot(b1, positions=[5],sym='+', boxprops=colors_versicolor, medianprops=colors_versicolor, whiskerprops=colors_versicolor, capprops=colors_versicolor, flierprops=dict(markeredgecolor=colors[1]), labels=['Binominal'])
# ax.boxplot(v4, positions=[4],sym='+', boxprops=colors_versicolor, medianprops=colors_versicolor, whiskerprops=colors_versicolor, capprops=colors_versicolor, flierprops=dict(markeredgecolor=colors[1]), labels=['Vanilla_160'])

# Creating plot
plt.title("TimeStamp Outliers")
plt.grid()
# plt.boxplot(data, sym='', meanline=True)
# show plot
plt.show()

# import seaborn as sns
# import matplotlib.pyplot as plts
# fig, ax = plts.subplots()
# plts.ion()
# sns.histplot(d1, color="red", label="Errors", kde=True, stat="density", element="poly", linewidth=3, bins=int(180 / 2))
# plts.title("Error Distribution")
# plts.draw()
# plts.pause(0.1)
# plts.show(block=True)

import seaborn as sn
import matplotlib.pyplot as plts
sn.set(style="darkgrid")
# plotting both distibutions on the same figure
# fig = sn.kdeplot(s1, fill=True, color="r", label='20 Tsteps')
# fig = sn.kdeplot(s2, fill=True, color="b", label='40 Tsteps')
fig = sn.kdeplot(t1, fill=True, color="g", label='Tiny')
fig = sn.kdeplot(v1, fill=True, color="r", label='Vanilla')
fig = sn.kdeplot(s1, fill=True, color="b", label='MultiStacked')
# fig = sn.kdeplot(s6, fill=True, color="pink", label='160 Tsteps')
# fig = sn.kdeplot(s7, fill=True, color="r", label='100 Best')
plts.title("Error Distribution Stacked")
plts.legend(prop={'size': 12})
plts.draw()
plts.pause(0.1)
plts.show(block=True)