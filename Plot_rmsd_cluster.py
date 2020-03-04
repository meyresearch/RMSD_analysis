__author__= "Antonia Mey"
__data__="23.3.2016"

r"""
The script is used to sort the RMSD matrix into clusters using a diffusion map
approach and the plot the sorted RMSDs according to cluster. This is part
of Figure 1 of the D3R challenge paper.
"""
###################################
# Imports
##################################

import matplotlib.pylab as plt
import numpy as np
import pickle
import seaborn as sbn
import pyemma
sbn.set(style="white")
sbn.set(style="ticks")

#load RMSD matrix
instream = open("RMSDmatrix.pickle","r")
matrix = pickle.load(instream)
instream.close()
rmsd_entries = np.zeros((len(matrix[0]),len(matrix[0])))
for i in range(len(matrix)):
    for j in range(len(matrix)):
        rmsd_entries[i][j] = matrix[i][j][2]

#do diffusion map approach
epsilon = 0.5
K = np.exp(-rmsd_entries**2/epsilon)
row_sum = np.sum(K, axis=1)
inv_row_sum = 1.0/row_sum
diag_inv_row_sum = np.diag(inv_row_sum)
Q = np.dot(np.dot(diag_inv_row_sum,K),diag_inv_row_sum)
Q_row_sum = np.sum(Q, axis=1)
Q_trans = np.dot(np.diag(1.0/Q_row_sum),Q) # finally the transition matrix I am after

#we established that we want 5 clusters based on the eigenvalue gap.
#We do a pcca analysis on these in order to group clusters.
msm = pyemma.msm.MSM(Q_trans.real)
msm.pcca(5)

cluster_index = msm.metastable_sets
index_list = []
for c in cluster_index:
    for i in c:
        index_list.append(i)
shift=np.array(index_list)
conv_matrix = np.zeros(shape=(rmsd_entries.shape[0], rmsd_entries.shape[0]))
for i in xrange(rmsd_entries.shape[0]):
    for j in xrange(rmsd_entries.shape[0]):
        conv_matrix[i][j]= rmsd_entries[shift[i]][shift[j]]


# In[10]:

sbn.set_context("notebook", font_scale=2, rc={"lines.linewidth": 2.5})
#fig = plt.figure(figsize=(15,15), dpi=300)
fig = plt.figure(figsize=(8,8))
plt.clf()
ax = fig.add_subplot(111)
#ax.set_aspect(1)
cmap = sbn.diverging_palette(10, 220, sep=80, n=7, as_cmap=True)
res = ax.imshow(conv_matrix, cmap=cmap,
                interpolation='nearest')
width = len(conv_matrix)
height = len(conv_matrix[0])
cb = fig.colorbar(res)
#let's save everything as a pdf file.
plt.savefig('rmsd_matrix.pdf', format='pdf')
