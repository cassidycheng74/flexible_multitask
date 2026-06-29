"""
analysis/analyze_repro.py  (v2)

Reproduction analysis for Driscoll et al. 2024.
Produces:
    1.  Training curves for all 15 tasks
    2.  Final performance bar chart
    3.  DelayGo PCA trajectories        (ring attractor check)
    4.  DelayDM1 PCA trajectories       (line attractor check)
    5.  Shared subspace: DelayGo vs DelayAnti
    6.  All tasks in global PCA space
    7.  Variance matrix                 (Driscoll Fig 3a equivalent)
    8.  Cross-task variance explained   (Driscoll Fig 3b/4d equivalent)
    9.  Context period PCA              (Driscoll Fig 4a equivalent)

Run from repo root:
    python analysis/analyze_repro.py
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Patch
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

MODEL_DIR = (
    'data/all/LeakyRNN/softplus/diag/15_tasks/128_n_rnn/'
    'lr6l2_w6_h6_sig_rec0.05_sig_x0.1_w_rec_coeff1_'
    'fdgo_reactgo_delaygo_fdanti_reactanti_delayanti_'
    'delaydm1_delaydm2_contextdelaydm1_contextdelaydm2_'
    'multidelaydm_dmsgo_dmsnogo_dmcgo_dmcnogo/0'
)

ACTIVITY_DIR = os.path.join(MODEL_DIR, 'activity', '0', 'test')
LOG_PATH     = os.path.join(MODEL_DIR, 'log.json')
SAVE_DIR     = os.path.join(MODEL_DIR, 'figures')
os.makedirs(SAVE_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Task metadata
# ---------------------------------------------------------------------------

TASKS = [
    'fdgo', 'reactgo', 'delaygo',
    'fdanti', 'reactanti', 'delayanti',
    'delaydm1', 'delaydm2',
    'contextdelaydm1', 'contextdelaydm2', 'multidelaydm',
    'dmsgo', 'dmsnogo', 'dmcgo', 'dmcnogo',
]

TASK_LABELS = {
    'fdgo'           : 'Go',
    'reactgo'        : 'RT Go',
    'delaygo'        : 'Dly Go',
    'fdanti'         : 'Anti',
    'reactanti'      : 'RT Anti',
    'delayanti'      : 'Dly Anti',
    'delaydm1'       : 'Dly DM 1',
    'delaydm2'       : 'Dly DM 2',
    'contextdelaydm1': 'Ctx DM 1',
    'contextdelaydm2': 'Ctx DM 2',
    'multidelaydm'   : 'Multi DM',
    'dmsgo'          : 'DMS',
    'dmsnogo'        : 'DNMS',
    'dmcgo'          : 'DMC',
    'dmcnogo'        : 'DNMC',
}

# Task family colors matching Driscoll's style.
TASK_COLORS = {
    'fdgo'           : '#4C72B0',
    'reactgo'        : '#4C72B0',
    'delaygo'        : '#4C72B0',
    'fdanti'         : '#DD8452',
    'reactanti'      : '#DD8452',
    'delayanti'      : '#DD8452',
    'delaydm1'       : '#55A868',
    'delaydm2'       : '#55A868',
    'contextdelaydm1': '#C44E52',
    'contextdelaydm2': '#C44E52',
    'multidelaydm'   : '#C44E52',
    'dmsgo'          : '#8172B2',
    'dmsnogo'        : '#8172B2',
    'dmcgo'          : '#937860',
    'dmcnogo'        : '#937860',
}

# Task family labels for legend.
FAMILY_COLORS = {
    'Go family'       : '#4C72B0',
    'Anti family'     : '#DD8452',
    'DM family'       : '#55A868',
    'Context DM'      : '#C44E52',
    'Match/Category'  : '#8172B2',
    'Cat Match'       : '#937860',
}

# Epoch definitions per task (timestep fractions, approximate).
# These are used to extract epoch-specific hidden states.
# Format: (start_frac, end_frac) as fraction of total trial length.
# Based on Driscoll's task definitions.
EPOCH_FRACS = {
    'fdgo'           : {'fix': (0.0, 0.35), 'stim': (0.35, 0.7),  'resp': (0.7,  1.0)},
    'reactgo'        : {'fix': (0.0, 0.65), 'stim': (0.65, 1.0),  'resp': (0.65, 1.0)},
    'delaygo'        : {'fix': (0.0, 0.28), 'stim': (0.28, 0.52), 'delay': (0.52, 0.78), 'resp': (0.78, 1.0)},
    'fdanti'         : {'fix': (0.0, 0.35), 'stim': (0.35, 0.7),  'resp': (0.7,  1.0)},
    'reactanti'      : {'fix': (0.0, 0.65), 'stim': (0.65, 1.0),  'resp': (0.65, 1.0)},
    'delayanti'      : {'fix': (0.0, 0.28), 'stim': (0.28, 0.52), 'delay': (0.52, 0.78), 'resp': (0.78, 1.0)},
    'delaydm1'       : {'fix': (0.0, 0.2),  'stim1': (0.2, 0.4),  'delay1': (0.4, 0.6), 'stim2': (0.6, 0.8), 'resp': (0.8, 1.0)},
    'delaydm2'       : {'fix': (0.0, 0.2),  'stim1': (0.2, 0.4),  'delay1': (0.4, 0.6), 'stim2': (0.6, 0.8), 'resp': (0.8, 1.0)},
    'contextdelaydm1': {'fix': (0.0, 0.15), 'stim1': (0.15, 0.35),'delay1': (0.35,0.55),'stim2': (0.55,0.75),'resp': (0.75,1.0)},
    'contextdelaydm2': {'fix': (0.0, 0.15), 'stim1': (0.15, 0.35),'delay1': (0.35,0.55),'stim2': (0.55,0.75),'resp': (0.75,1.0)},
    'multidelaydm'   : {'fix': (0.0, 0.15), 'stim1': (0.15, 0.35),'delay1': (0.35,0.55),'stim2': (0.55,0.75),'resp': (0.75,1.0)},
    'dmsgo'          : {'fix': (0.0, 0.2),  'stim1': (0.2, 0.45), 'delay1': (0.45, 0.7),'stim2': (0.7, 1.0), 'resp': (0.7, 1.0)},
    'dmsnogo'        : {'fix': (0.0, 0.2),  'stim1': (0.2, 0.45), 'delay1': (0.45, 0.7),'stim2': (0.7, 1.0), 'resp': (0.7, 1.0)},
    'dmcgo'          : {'fix': (0.0, 0.2),  'stim1': (0.2, 0.45), 'delay1': (0.45, 0.7),'stim2': (0.7, 1.0), 'resp': (0.7, 1.0)},
    'dmcnogo'        : {'fix': (0.0, 0.2),  'stim1': (0.2, 0.45), 'delay1': (0.45, 0.7),'stim2': (0.7, 1.0), 'resp': (0.7, 1.0)},
}


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_log():
    with open(LOG_PATH) as f:
        return json.load(f)


def load_activity():
    state  = np.load(os.path.join(ACTIVITY_DIR, 'state.npz'))
    inputs = np.load(os.path.join(ACTIVITY_DIR, 'trial_input.npz'))
    output = np.load(os.path.join(ACTIVITY_DIR, 'trial_output.npz'))
    return state, inputs, output


def get_epoch_states(h, epoch_key, task):
    """
    Extract hidden states from a specific epoch of a task.

    Args:
        h:         (T, B, N) hidden states
        epoch_key: epoch name string e.g. 'stim', 'delay', 'resp'
        task:      task name string

    Returns:
        epoch_h: (T_epoch, B, N) hidden states for that epoch
    """
    T = h.shape[0]
    fracs = EPOCH_FRACS[task]
    if epoch_key not in fracs:
        # Fall back to last third of trial as response.
        t0, t1 = int(0.67 * T), T
    else:
        t0 = int(fracs[epoch_key][0] * T)
        t1 = int(fracs[epoch_key][1] * T)
    t0 = max(0, t0)
    t1 = min(T, max(t0 + 1, t1))
    return h[t0:t1]


# ---------------------------------------------------------------------------
# PCA helpers
# ---------------------------------------------------------------------------

def fit_pca_on_task(state_data, n_components=3):
    T, B, N = state_data.shape
    flat = state_data.reshape(T * B, N)
    pca  = PCA(n_components=n_components)
    pca.fit(flat)
    proj = pca.transform(flat).reshape(T, B, n_components)
    return pca, proj


def project_onto_pca(state_data, pca):
    T, B, N = state_data.shape
    flat = state_data.reshape(T * B, N)
    proj = pca.transform(flat).reshape(T, B, pca.n_components_)
    return proj


# ---------------------------------------------------------------------------
# Figure 1: Training curves
# ---------------------------------------------------------------------------

def plot_training_curves(log):
    trials = np.array(log['trials']) / 1e6

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    for task in TASKS:
        key = 'perf_' + task
        if key in log:
            ax.plot(trials, log[key],
                    label=TASK_LABELS[task],
                    color=TASK_COLORS[task],
                    alpha=0.85, linewidth=1.5)
    ax.set_xlabel('Trials (millions)', fontsize=12)
    ax.set_ylabel('Performance', fontsize=12)
    ax.set_title('Task Performance over Training', fontsize=13)
    ax.set_ylim([0, 1.05])
    ax.axhline(1.0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.legend(fontsize=7, ncol=2, loc='lower right')

    ax = axes[1]
    for task in TASKS:
        key = 'cost_' + task
        if key in log:
            vals = np.clip(np.array(log[key]), 1e-6, None)
            ax.semilogy(trials, vals,
                        label=TASK_LABELS[task],
                        color=TASK_COLORS[task],
                        alpha=0.85, linewidth=1.5)
    ax.set_xlabel('Trials (millions)', fontsize=12)
    ax.set_ylabel('Cost (log scale)', fontsize=12)
    ax.set_title('Task Cost over Training', fontsize=13)
    ax.legend(fontsize=7, ncol=2, loc='upper right')

    plt.tight_layout()
    path = os.path.join(SAVE_DIR, 'fig_training_curves.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f'Saved: {path}')
    plt.close()


# ---------------------------------------------------------------------------
# Figure 2: Final performance bar chart
# ---------------------------------------------------------------------------

def plot_performance_bar(log):
    fig, ax = plt.subplots(figsize=(12, 4))

    final_perfs, colors, labels = [], [], []
    for task in TASKS:
        key = 'perf_' + task
        if key in log:
            final_perfs.append(log[key][-1])
            colors.append(TASK_COLORS[task])
            labels.append(TASK_LABELS[task])

    x    = np.arange(len(labels))
    bars = ax.bar(x, final_perfs, color=colors, edgecolor='white', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
    ax.set_ylabel('Final Performance', fontsize=12)
    ax.set_title('Final Performance Across All 15 Tasks', fontsize=13)
    ax.set_ylim([0, 1.1])
    ax.axhline(1.0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    for bar, perf in zip(bars, final_perfs):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f'{perf:.2f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    path = os.path.join(SAVE_DIR, 'fig_final_performance.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f'Saved: {path}')
    plt.close()


# ---------------------------------------------------------------------------
# Figure 3: DelayGo PCA (ring attractor check)
# ---------------------------------------------------------------------------

def plot_delaygo_pca(state):
    h = state['delaygo']
    T, B, N = h.shape
    pca, proj = fit_pca_on_task(h, n_components=3)
    var_exp   = pca.explained_variance_ratio_
    n_locs    = 40
    cmap      = cm.hsv

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (pc_x, pc_y) in zip(axes, [(0, 1), (0, 2)]):
        for i in range(min(n_locs, B)):
            color = cmap(i / n_locs)
            traj  = proj[:, i, :]
            ax.plot(traj[:, pc_x], traj[:, pc_y],
                    color=color, alpha=0.6, linewidth=1.2)
            ax.scatter(traj[-1, pc_x], traj[-1, pc_y],
                       color=color, s=20, zorder=5)
        ax.set_xlabel(f'PC{pc_x+1} ({var_exp[pc_x]*100:.1f}%)', fontsize=11)
        ax.set_ylabel(f'PC{pc_y+1} ({var_exp[pc_y]*100:.1f}%)', fontsize=11)
        ax.set_title(f'DelayGo: PC{pc_x+1} vs PC{pc_y+1}\n(color = stimulus angle)', fontsize=11)
        ax.set_aspect('equal')

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=360))
    sm.set_array([])
    plt.colorbar(sm, ax=axes, label='Stimulus angle (degrees)', fraction=0.02, pad=0.04)
    plt.suptitle('Ring Attractor Check: DelayGo Hidden State PCA', fontsize=13, y=1.02)
    plt.tight_layout()
    path = os.path.join(SAVE_DIR, 'fig_delaygo_pca.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f'Saved: {path}')
    plt.close()
    print(f'  DelayGo variance explained: '
          f'PC1={var_exp[0]*100:.1f}%  PC2={var_exp[1]*100:.1f}%  PC3={var_exp[2]*100:.1f}%')


# ---------------------------------------------------------------------------
# Figure 4: DelayDM1 PCA (line attractor check)
# ---------------------------------------------------------------------------

def plot_delaydm1_pca(state):
    h = state['delaydm1']
    T, B, N = h.shape
    pca, proj = fit_pca_on_task(h, n_components=3)
    var_exp   = pca.explained_variance_ratio_
    half      = B // 2

    fig, ax = plt.subplots(figsize=(6, 5))
    for i in range(B):
        color = '#4C72B0' if i < half else '#DD8452'
        traj  = proj[:, i, :]
        ax.plot(traj[:, 0], traj[:, 1], color=color, alpha=0.4, linewidth=0.8)
        ax.scatter(traj[-1, 0], traj[-1, 1], color=color, s=15, zorder=5)

    ax.legend(handles=[
        Patch(facecolor='#4C72B0', label='Stim 1 stronger'),
        Patch(facecolor='#DD8452', label='Stim 2 stronger'),
    ], fontsize=10)
    ax.set_xlabel(f'PC1 ({var_exp[0]*100:.1f}%)', fontsize=11)
    ax.set_ylabel(f'PC2 ({var_exp[1]*100:.1f}%)', fontsize=11)
    ax.set_title('DelayDM1: PC1 vs PC2\n(Line/Point Attractor Check)', fontsize=11)
    plt.tight_layout()
    path = os.path.join(SAVE_DIR, 'fig_delaydm1_pca.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f'Saved: {path}')
    plt.close()
    print(f'  DelayDM1 variance explained: '
          f'PC1={var_exp[0]*100:.1f}%  PC2={var_exp[1]*100:.1f}%  PC3={var_exp[2]*100:.1f}%')


# ---------------------------------------------------------------------------
# Figure 5: Shared subspace DelayGo vs DelayAnti
# ---------------------------------------------------------------------------

def plot_shared_subspace(state):
    h_go   = state['delaygo']
    h_anti = state['delayanti']
    pca, proj_go  = fit_pca_on_task(h_go, n_components=3)
    proj_anti     = project_onto_pca(h_anti, pca)
    var_exp       = pca.explained_variance_ratio_
    n_locs        = 40
    cmap          = cm.hsv

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, (proj, label) in zip(axes, [(proj_go, 'DelayGo'), (proj_anti, 'DelayAnti')]):
        for i in range(min(n_locs, proj.shape[1])):
            color = cmap(i / n_locs)
            traj  = proj[:, i, :]
            ax.plot(traj[:, 0], traj[:, 1], color=color, alpha=0.7, linewidth=1.2)
            ax.scatter(traj[-1, 0], traj[-1, 1], color=color, s=20, zorder=5)
        ax.set_xlabel(f'PC1 ({var_exp[0]*100:.1f}%)', fontsize=11)
        ax.set_ylabel(f'PC2 ({var_exp[1]*100:.1f}%)', fontsize=11)
        ax.set_title(f'{label} in DelayGo PCA space', fontsize=11)
        ax.set_aspect('equal')

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=360))
    sm.set_array([])
    plt.colorbar(sm, ax=axes, label='Stimulus angle (degrees)', fraction=0.02, pad=0.04)
    plt.suptitle('Shared Subspace Check: DelayGo vs DelayAnti\n'
                 '(Both projected onto DelayGo PCA axes)', fontsize=13, y=1.02)
    plt.tight_layout()
    path = os.path.join(SAVE_DIR, 'fig_shared_subspace.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f'Saved: {path}')
    plt.close()


# ---------------------------------------------------------------------------
# Figure 6: All tasks in global PCA space
# ---------------------------------------------------------------------------

def plot_all_tasks_pca(state):
    all_flat = np.concatenate([
        state[t].reshape(-1, state[t].shape[-1]) for t in TASKS], axis=0)
    pca     = PCA(n_components=3)
    pca.fit(all_flat)
    var_exp = pca.explained_variance_ratio_

    ncols = 5
    nrows = (len(TASKS) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 3.5))
    axes = axes.flatten()

    for idx, task in enumerate(TASKS):
        h    = state[task]
        T, B, N = h.shape
        proj = pca.transform(h.reshape(T * B, N)).reshape(T, B, 3)
        ax   = axes[idx]
        cmap = cm.hsv
        for i in range(min(40, B)):
            color = cmap(i / min(40, B))
            ax.plot(proj[:, i, 0], proj[:, i, 1],
                    color=color, alpha=0.5, linewidth=0.8)
        ax.set_title(TASK_LABELS[task], fontsize=10)
        ax.set_xlabel('PC1', fontsize=8)
        ax.set_ylabel('PC2', fontsize=8)
        ax.tick_params(labelsize=7)

    for idx in range(len(TASKS), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(
        f'All Tasks in Global PCA Space\n'
        f'(PC1={var_exp[0]*100:.1f}%  PC2={var_exp[1]*100:.1f}%  PC3={var_exp[2]*100:.1f}%)',
        fontsize=13)
    plt.tight_layout()
    path = os.path.join(SAVE_DIR, 'fig_all_tasks_pca.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f'Saved: {path}')
    plt.close()
    print(f'  Global PCA variance explained: '
          f'PC1={var_exp[0]*100:.1f}%  PC2={var_exp[1]*100:.1f}%  PC3={var_exp[2]*100:.1f}%')


# ---------------------------------------------------------------------------
# Figure 7: Variance matrix  (Driscoll Fig 3a equivalent)
# ---------------------------------------------------------------------------

def plot_variance_matrix(state):
    """
    For each task and each unit, compute the variance of that unit's
    activation across the batch dimension (stimulus conditions).
    Normalize each task column by its maximum.
    Cluster rows (units) by similarity across tasks and plot as heatmap.

    Driscoll Fig 3a shows this matrix sorted by hierarchical clustering,
    revealing blocks of units that are selectively active for particular
    task families.
    """
    print('  Computing unit variance per task...')

    n_units = state[TASKS[0]].shape[2]
    n_tasks = len(TASKS)

    # Build variance matrix: (n_units, n_tasks).
    var_mat = np.zeros((n_units, n_tasks))
    for j, task in enumerate(TASKS):
        h = state[task]   # (T, B, N)
        # Mean over time first, then variance over batch.
        h_mean_time = h.mean(axis=0)   # (B, N)
        var_mat[:, j] = h_mean_time.var(axis=0)   # (N,)

    # Normalize each column by its maximum (as Driscoll does).
    col_max = var_mat.max(axis=0, keepdims=True)
    col_max[col_max == 0] = 1.0
    var_mat_norm = var_mat / col_max

    # Hierarchical clustering on units (rows) by their variance profile.
    print('  Clustering units...')
    row_dist    = pdist(var_mat_norm, metric='correlation')
    row_link    = linkage(row_dist, method='ward')
    # Get leaf ordering from dendrogram (no plot).
    dendro      = dendrogram(row_link, no_plot=True)
    row_order   = dendro['leaves']

    # Sort matrix by cluster order.
    var_sorted = var_mat_norm[row_order, :]

    # Task label colors for x-axis.
    task_label_list = [TASK_LABELS[t] for t in TASKS]
    task_color_list = [TASK_COLORS[t] for t in TASKS]

    fig, axes = plt.subplots(1, 2, figsize=(14, 7),
                              gridspec_kw={'width_ratios': [3, 1]})

    # Main heatmap.
    ax = axes[0]
    im = ax.imshow(var_sorted, aspect='auto', cmap='hot',
                   vmin=0, vmax=1, interpolation='nearest')
    ax.set_xticks(range(n_tasks))
    ax.set_xticklabels(task_label_list, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Units (sorted by clustering)', fontsize=11)
    ax.set_title('Unit Variance Matrix\n(normalized per task, sorted by unit clustering)',
                 fontsize=12)
    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02,
                 label='Normalized variance')

    # Color the x-axis tick labels by task family.
    for tick, color in zip(ax.get_xticklabels(), task_color_list):
        tick.set_color(color)

    # Right panel: mean variance profile per task (summary bar).
    ax2 = axes[1]
    mean_var = var_mat_norm.mean(axis=0)
    bars = ax2.barh(range(n_tasks), mean_var,
                    color=task_color_list, edgecolor='white')
    ax2.set_yticks(range(n_tasks))
    ax2.set_yticklabels(task_label_list, fontsize=9)
    ax2.set_xlabel('Mean normalized variance', fontsize=10)
    ax2.set_title('Mean unit\nvariance per task', fontsize=11)
    ax2.invert_yaxis()

    plt.suptitle('Variance Matrix: Unit Selectivity Across Tasks\n'
                 '(Driscoll Fig 3a equivalent)', fontsize=13, y=1.01)
    plt.tight_layout()
    path = os.path.join(SAVE_DIR, 'fig_variance_matrix.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f'Saved: {path}')
    plt.close()

    # Print sparsity: fraction of entries below 15% of max.
    sparsity = (var_mat_norm < 0.15).mean()
    print(f'  Variance matrix sparsity (entries < 15% max): {sparsity:.3f}')
    print(f'  (Driscoll reports high sparsity as a sign of task-selective units)')


# ---------------------------------------------------------------------------
# Figure 8: Cross-task variance explained  (Driscoll Fig 3b/4d equivalent)
# ---------------------------------------------------------------------------

def plot_cross_task_variance(state):
    """
    For each pair of tasks (A, B):
        - Fit PCA on task A's hidden states (top 2 PCs)
        - Measure how much variance those PCs explain in task B
    Plot as a 15x15 matrix.

    Driscoll uses this to show that task pairs within the same motif family
    explain more variance in each other than task pairs from different families.
    High off-diagonal values within a family = shared subspace.
    """
    print('  Computing cross-task variance explained...')

    n_tasks = len(TASKS)
    var_exp_mat = np.zeros((n_tasks, n_tasks))

    for i, task_a in enumerate(TASKS):
        h_a  = state[task_a]
        T_a, B_a, N = h_a.shape
        flat_a = h_a.reshape(T_a * B_a, N)

        pca_a = PCA(n_components=2)
        pca_a.fit(flat_a)

        for j, task_b in enumerate(TASKS):
            h_b    = state[task_b]
            T_b, B_b, _ = h_b.shape
            flat_b = h_b.reshape(T_b * B_b, N)

            # Variance explained by task A's PCs on task B's data.
            # = 1 - (residual variance / total variance)
            proj_b   = pca_a.transform(flat_b)
            recon_b  = pca_a.inverse_transform(proj_b)
            ss_res   = np.sum((flat_b - recon_b) ** 2)
            ss_tot   = np.sum((flat_b - flat_b.mean(axis=0)) ** 2)
            var_exp_mat[i, j] = max(0, 1 - ss_res / (ss_tot + 1e-10))

    task_label_list = [TASK_LABELS[t] for t in TASKS]

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(var_exp_mat, cmap='viridis', vmin=0, vmax=1,
                   aspect='auto', interpolation='nearest')
    ax.set_xticks(range(n_tasks))
    ax.set_yticks(range(n_tasks))
    ax.set_xticklabels(task_label_list, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(task_label_list, fontsize=9)
    ax.set_xlabel('Task B (data being explained)', fontsize=11)
    ax.set_ylabel('Task A (PCA fitted on A, applied to B)', fontsize=11)
    ax.set_title('Cross-Task Variance Explained\n'
                 'Entry (A,B) = fraction of task B variance explained by top 2 PCs of task A\n'
                 '(Driscoll Fig 3b/4d equivalent)',
                 fontsize=11)
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02,
                 label='Fraction variance explained')

    # Draw family boundary lines.
    boundaries = [3, 6, 8, 11, 13]   # after go, anti, dm, context, dms groups
    for b in boundaries:
        ax.axhline(b - 0.5, color='white', linewidth=1.5, alpha=0.7)
        ax.axvline(b - 0.5, color='white', linewidth=1.5, alpha=0.7)

    plt.tight_layout()
    path = os.path.join(SAVE_DIR, 'fig_cross_task_variance.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f'Saved: {path}')
    plt.close()

    # Print highest cross-task pairs (excluding diagonal).
    print('  Top cross-task variance explained pairs:')
    np.fill_diagonal(var_exp_mat, 0)
    top_idx = np.dstack(np.unravel_index(
        np.argsort(var_exp_mat.ravel())[::-1][:6], var_exp_mat.shape))[0]
    for i, j in top_idx:
        print(f'    {TASK_LABELS[TASKS[i]]:<12} -> {TASK_LABELS[TASKS[j]]:<12}  '
              f'{var_exp_mat[i,j]:.3f}')


# ---------------------------------------------------------------------------
# Figure 9: Context period PCA  (Driscoll Fig 4a equivalent)
# ---------------------------------------------------------------------------

def plot_context_period_pca(state):
    """
    Extract the hidden state at the END of the fixation/context period
    (just before stimulus onset) for all tasks and all trials.
    Project into a shared PCA space and color by task family.

    Driscoll Fig 4a shows that tasks with the same computational motif
    cluster together in context-period state space, before any stimulus
    information has arrived. This means the network has already committed
    to a computational strategy based on the rule input alone.
    """
    print('  Computing context period states...')

    # For each task, take the last few timesteps of the fixation epoch.
    context_states = []   # list of (B, N) arrays
    task_indices   = []

    for i, task in enumerate(TASKS):
        h   = state[task]   # (T, B, N)
        T   = h.shape[0]
        # Use last 10% of the fixation period as the context state.
        t0  = int(EPOCH_FRACS[task]['fix'][0] * T)
        t1  = int(EPOCH_FRACS[task]['fix'][1] * T)
        t_context = max(t0, t1 - max(1, int(0.1 * T)))
        ctx = h[t_context:t1].mean(axis=0)   # (B, N) mean over last few fix steps
        context_states.append(ctx)
        task_indices.extend([i] * ctx.shape[0])

    all_ctx    = np.concatenate(context_states, axis=0)   # (sum_B, N)
    task_idx   = np.array(task_indices)

    # Fit PCA on all context states.
    pca     = PCA(n_components=2)
    pca.fit(all_ctx)
    proj    = pca.transform(all_ctx)   # (sum_B, 2)
    var_exp = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(8, 7))

    # Plot each task's context states as a scatter cloud.
    offset = 0
    for i, task in enumerate(TASKS):
        n = context_states[i].shape[0]
        pts = proj[offset:offset + n]
        ax.scatter(pts[:, 0], pts[:, 1],
                   color=TASK_COLORS[task],
                   label=TASK_LABELS[task],
                   alpha=0.6, s=15, zorder=3)
        # Mark centroid with a larger marker.
        ax.scatter(pts[:, 0].mean(), pts[:, 1].mean(),
                   color=TASK_COLORS[task],
                   s=120, marker='*', edgecolors='black',
                   linewidths=0.8, zorder=5)
        offset += n

    ax.set_xlabel(f'PC1 ({var_exp[0]*100:.1f}%)', fontsize=12)
    ax.set_ylabel(f'PC2 ({var_exp[1]*100:.1f}%)', fontsize=12)
    ax.set_title('Context Period PCA: Neural State Before Stimulus Onset\n'
                 'Tasks with same motif should cluster together\n'
                 '(Driscoll Fig 4a equivalent)',
                 fontsize=11)
    ax.legend(fontsize=7, ncol=2, loc='best',
              markerscale=1.5, framealpha=0.8)

    plt.tight_layout()
    path = os.path.join(SAVE_DIR, 'fig_context_period_pca.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f'Saved: {path}')
    plt.close()

    print(f'  Context PCA variance explained: '
          f'PC1={var_exp[0]*100:.1f}%  PC2={var_exp[1]*100:.1f}%')
    print('  Look for clustering by task family (color) in the figure.')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print('Loading log...')
    log = load_log()

    print('Loading activity files...')
    state, inputs, output = load_activity()

    print('\n--- 1. Training curves ---')
    plot_training_curves(log)

    print('\n--- 2. Performance bar chart ---')
    plot_performance_bar(log)

    print('\n--- 3. DelayGo PCA (ring attractor check) ---')
    plot_delaygo_pca(state)

    print('\n--- 4. DelayDM1 PCA (line attractor check) ---')
    plot_delaydm1_pca(state)

    print('\n--- 5. Shared subspace: DelayGo vs DelayAnti ---')
    plot_shared_subspace(state)

    print('\n--- 6. All tasks in global PCA space ---')
    plot_all_tasks_pca(state)

    print('\n--- 7. Variance matrix (Driscoll Fig 3a) ---')
    plot_variance_matrix(state)

    print('\n--- 8. Cross-task variance explained (Driscoll Fig 3b/4d) ---')
    plot_cross_task_variance(state)

    print('\n--- 9. Context period PCA (Driscoll Fig 4a) ---')
    plot_context_period_pca(state)

    print(f'\nAll figures saved to: {SAVE_DIR}')
    print('Open them in VSCode by clicking the .png files in the file explorer.')