"""Plot revised temporal-scheduling experiment results."""
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_theme(style='whitegrid', context='paper', font_scale=1.15)
PALETTE = {'Random': '#e74c3c', 'LRU': '#f39c12', 'LR-Gate': '#2ecc71', 'Lookahead': '#3498db'}
POLICY_ORDER = ['Random', 'LRU', 'LR-Gate', 'Lookahead']
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
FIG_DIR = os.path.expanduser('~/Desktop/papers/qvm_scheduler/figures')
os.makedirs(FIG_DIR, exist_ok=True)


def load(name):
    with open(os.path.join(RESULTS_DIR, name)) as f:
        return json.load(f)


def savefig(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('saved', path)


def fig_order_sensitivity():
    data = load('exp_rq1_order_sensitivity.json')
    orderings = ['clustered', 'interleaved']
    sizes = sorted({r['N'] for r in data})
    width = 0.18
    x = np.arange(len(sizes))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, ordering in zip(axes, orderings):
        rows = [r for r in data if r['ordering'] == ordering]
        agg = defaultdict(list)
        for r in rows:
            agg[(r['N'], r['policy'])].append(r['K'])
        for i, pol in enumerate(POLICY_ORDER):
            means = [np.mean(agg[(n, pol)]) for n in sizes]
            ax.bar(x + i * width, means, width, color=PALETTE[pol], label=pol,
                   edgecolor='white', linewidth=0.5)
        ax.set_title(ordering.capitalize())
        ax.set_xlabel('Virtual qubits (N)')
        ax.set_xticks(x + 1.5 * width)
        ax.set_xticklabels(sizes)
    axes[0].set_ylabel('Cut count (K)')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4, frameon=True,
               bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    savefig(fig, 'fig_order_sensitivity.pdf')


def fig_oracle_gap():
    data = load('exp_rq2_oracle_gap.json')
    workloads = [
        'qaoa_p1', 'qaoa_p2', 'phase_clustered', 'phase_interleaved',
        'ancilla_reuse', 'working_set_shift', 'random_seed42', 'random_seed43'
    ]
    matrix = np.zeros((len(workloads), len(POLICY_ORDER)))
    for i, w in enumerate(workloads):
        for j, p in enumerate(POLICY_ORDER):
            vals = [r['gap_abs'] for r in data if r['workload'] == w and r['policy'] == p]
            matrix[i, j] = np.mean(vals) if vals else np.nan

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    sns.heatmap(matrix, annot=True, fmt='.0f', cmap='YlGnBu_r', cbar_kws={'label': 'Absolute gap to OPT'}, ax=ax,
                xticklabels=POLICY_ORDER, yticklabels=workloads, linewidths=0.5, linecolor='white')
    ax.set_xlabel('Policy')
    ax.set_ylabel('Workload')
    ax.set_title('Exact Oracle Gap on Small Instances')
    savefig(fig, 'fig_oracle_gap.pdf')


def fig_dynamic_workloads():
    data = load('exp_rq4_temporal_workloads.json')
    workloads = ['ancilla_reuse_r2', 'ancilla_reuse_r3', 'working_set_shift_p4', 'working_set_shift_p6', 'qaoa_p3_16q']
    width = 0.18
    x = np.arange(len(workloads))
    fig, ax = plt.subplots(figsize=(10, 4.5))
    agg = defaultdict(list)
    for r in data:
        agg[(r['workload'], r['policy'])].append(r['K'])
    for i, pol in enumerate(POLICY_ORDER):
        means = [np.mean(agg[(w, pol)]) for w in workloads]
        ax.bar(x + i * width, means, width, color=PALETTE[pol], label=pol,
               edgecolor='white', linewidth=0.5)
    ax.set_ylabel('Cut count (K)')
    ax.set_xlabel('Workload')
    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels(['reuse-r2', 'reuse-r3', 'shift-p4', 'shift-p6', 'QAOA p=3'], rotation=15)
    ax.set_title('Dynamic-Circuit-Inspired and Phase-Structured Workloads')
    ax.legend(title='Policy', frameon=True, ncol=4, loc='upper center', bbox_to_anchor=(0.5, 1.22))
    savefig(fig, 'fig_dynamic_workloads.pdf')


if __name__ == '__main__':
    fig_order_sensitivity()
    fig_oracle_gap()
    fig_dynamic_workloads()
