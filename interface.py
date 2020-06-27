COLOR_PALETE_PAR = (10, 150, 400)
MAXDISTANCE = 40
CHARSIZE_PLOT = .5
FONTSIZE = 18
MAX_LEN = 16


import matplotlib.pyplot as plt
import math
import numpy as np
import seaborn as sns

CMAP = sns.diverging_palette(COLOR_PALETE_PAR[0], COLOR_PALETE_PAR[1], n=COLOR_PALETE_PAR[2])


def choseChangeThis(c):
    return np.argmax(c)

def feedback(s, g, p, CC, P):
    
    # normalize feedback
    c = (1 - ( g / MAXDISTANCE))
    c = np.clip(c, 0, 1)
    
    # color feedback
    y = 1 - c
    
    n = len(p)
    fig, ax = plt.subplots(1, 1, figsize=(n*CHARSIZE_PLOT, CHARSIZE_PLOT))
    xp = np.arange(len(c))
    
    if CC:
        # select to change character
        t = choseChangeThis(p)
        ax.annotate('Change this', xy=(t+CHARSIZE_PLOT, 0), xycoords='data', xytext=(t+CHARSIZE_PLOT, -1), arrowprops=dict(facecolor='black', shrink=0.0), horizontalalignment='center', verticalalignment='center')
    
    fp = 3
    p = np.round(p, fp)


    sns.heatmap(y[None,:], vmin=0, vmax=1, ax=ax, cmap=CMAP, cbar=False)
    
    for i in range(len(s)):
        if P:
            ax.text(i+0.8, 1.5, '%.3f' % p[i], horizontalalignment='center',  verticalalignment='center', fontsize=10, rotation=35)
        ax.text(i+CHARSIZE_PLOT, CHARSIZE_PLOT, s[i], horizontalalignment='center',  verticalalignment='center', fontsize=FONTSIZE)
        
    ax.set(ylim=(0, 1), yticks=[], xticks=[]);


def plot(S, s, CC=True, P=True, verbose=False):
    
    if len(s) > MAX_LEN:
        print("[WARNING] This model only support passwords with a maximal length of %d" % MAX_LEN)
        return
    
    xs, cp, p, sess = S.meterSingle(s, sess=S.sess)
    cp, ng = cp
    bng = ng.prod()
    feedback(s, ng, cp, CC, P)
    
    if verbose:
        print(cp)
        print(xs)
        print("WORST-CASE GUESS NUMBER: %s MAGNITUDE:%s" % (bng, int(np.floor(np.log10(bng)))))
        