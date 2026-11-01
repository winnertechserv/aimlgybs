import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
import numpy as np, os
NAVY, TEAL, ORANGE, RED, GREY, LIGHT, GREEN, PALE = '#1F497D', '#17575E', '#D9822B', '#C0392B', '#7F7F7F', '#DCE6F1', '#2E8B57', '#FBE9D7'
plt.rcParams.update({'font.size': 10, 'axes.titlesize': 11, 'axes.titleweight': 'bold', 'axes.spines.top': False,
                     'axes.spines.right': False, 'figure.dpi': 100, 'axes.titlecolor': NAVY})
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'png')
REG = []          # (chapter, anchor, file, caption)
def reg(ch, anchor, name, caption):
    REG.append((ch, anchor, name, caption))
def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=160, bbox_inches='tight', facecolor='white'); plt.close(fig)
def diagram(w=8, h=3.2, xlim=(0, 10), ylim=(0, 4)):
    fig, ax = plt.subplots(figsize=(w, h)); ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.axis('off'); return fig, ax
def box(ax, x, y, text, w=1.8, h=0.8, fc=LIGHT, ec=NAVY, fs=9, bold=False, tc='black', round_=0.08):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle=f'round,pad=0.02,rounding_size={round_}',
                                fc=fc, ec=ec, lw=1.3))
    ax.text(x, y, text, ha='center', va='center', fontsize=fs, fontweight='bold' if bold else 'normal', color=tc, wrap=True)
def arrow(ax, p1, p2, text=None, color=NAVY, rad=0.0, fs=8, lw=1.4, toff=(0, 0.18), style='-|>', ls='-'):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=13, color=color, lw=lw, linestyle=ls,
                                 connectionstyle=f'arc3,rad={rad}', shrinkA=2, shrinkB=2))
    if text:
        ax.text((p1[0] + p2[0]) / 2 + toff[0], (p1[1] + p2[1]) / 2 + toff[1], text, ha='center', va='center',
                fontsize=fs, color=color, style='italic')
