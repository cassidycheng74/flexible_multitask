"""
analysis/analyze_repro.py

Reproduction analysis for Driscoll et al. 2024.
Produces:
    1. Training curves for all 15 tasks
    2. PCA trajectories for delaygo (ring attractor check)
    3. PCA trajectories for delaydm1 (line attractor check)
    4. Overlay of delaygo + delayanti in same PCA space (shared subspace check)
    5. Final performance bar chart for all 15 tasks

Run from repo root:
    python analysis/analyze_repro.py
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.decomposition import PCA

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

# Color families matching Driscoll's figure style.
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


# ---------------------------------------------------------------------------
# Figure 1: Training curves
# ---------------------------------------------------------------------------

def plot_training_curves(log):
    trials = np.array(log['trials']) / 1e6   # convert to millions

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Performance.
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

    # Cost (log scale).
    ax = axes[1]
    for task in TASKS:
        key = 'cost_' + task
        if key in log:
            vals = np.array(log[key])
            vals = np.clip(vals, 1e-6, None)
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

    final_perfs = []
    colors      = []
    labels      = []
    for task in TASKS:
        key = 'perf_' + task
        if key in log:
            final_perfs.append(log[key][-1])
            colors.append(TASK_COLORS[task])
            labels.append(TASK_LABELS[task])

    x = np.arange(len(labels))
    bars = ax.bar(x, final_perfs, color=colors, edgecolor='white', linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
    ax.set_ylabel('Final Performance', fontsize=12)
    ax.set_title('Final Performance Across All 15 Tasks', fontsize=13)
    ax.set_ylim([0, 1.1])
    ax.axhline(1.0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)

    # Annotate bars.
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
# PCA helper
# ---------------------------------------------------------------------------

def fit_pca_on_task(state_data, n_components=3):
    """
    Fit PCA on hidden states of one task.

    Args:
        state_data: (T, B, n_rnn) array
        n_components: number of PCA components

    Returns:
        pca: fitted PCA object
        projected: (T, B, n_components) projected states
    """
    T, B, N = state_data.shape
    # Reshape to (T*B, N) for PCA.
    flat = state_data.reshape(T * B, N)
    pca  = PCA(n_components=n_components)
    pca.fit(flat)
    proj = pca.transform(flat).reshape(T, B, n_components)
    return pca, proj


def project_onto_pca(state_data, pca):
    """Project state_data onto an already-fitted PCA."""
    T, B, N = state_data.shape
    flat = state_data.reshape(T * B, N)
    proj = pca.transform(flat).reshape(T, B, pca.n_components_)
    return proj


# ---------------------------------------------------------------------------
# Figure 3: delaygo PCA trajectories (ring attractor check)
# ---------------------------------------------------------------------------

def plot_delaygo_pca(state):
    """
    Plot PCA trajectories for delaygo task.
    Each trial is colored by stimulus angle.
    A ring in PC1-PC2 space during the delay period = ring attractor.
    """
    h = state['delaygo']   # (T, B, 128)
    T, B, N = h.shape

    pca, proj = fit_pca_on_task(h, n_components=3)
    var_exp = pca.explained_variance_ratio_

    # delaygo test mode: 40 locations x 2 mods = 80 trials.
    # Stimulus angles are evenly spaced 0..2pi for the first 40 trials.
    n_locs  = 40
    angles  = np.linspace(0, 2 * np.pi, n_locs, endpoint=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    cmap = cm.hsv

    # PC1 vs PC2.
    ax = axes[0]
    for i in range(min(n_locs, B)):
        color = cmap(i / n_locs)
        traj  = proj[:, i, :]   # (T, 3)
        ax.plot(traj[:, 0], traj[:, 1], color=color, alpha=0.6, linewidth=1.2)
        # Mark end of trial with a dot.
        ax.scatter(traj[-1, 0], traj[-1, 1], color=color, s=20, zorder=5)

    ax.set_xlabel(f'PC1 ({var_exp[0]*100:.1f}%)', fontsize=11)
    ax.set_ylabel(f'PC2 ({var_exp[1]*100:.1f}%)', fontsize=11)
    ax.set_title('DelayGo: PC1 vs PC2\n(color = stimulus angle)', fontsize=11)
    ax.set_aspect('equal')

    # PC1 vs PC3.
    ax = axes[1]
    for i in range(min(n_locs, B)):
        color = cmap(i / n_locs)
        traj  = proj[:, i, :]
        ax.plot(traj[:, 0], traj[:, 2], color=color, alpha=0.6, linewidth=1.2)
        ax.scatter(traj[-1, 0], traj[-1, 2], color=color, s=20, zorder=5)

    ax.set_xlabel(f'PC1 ({var_exp[0]*100:.1f}%)', fontsize=11)
    ax.set_ylabel(f'PC3 ({var_exp[2]*100:.1f}%)', fontsize=11)
    ax.set_title('DelayGo: PC1 vs PC3', fontsize=11)
    ax.set_aspect('equal')

    # Colorbar for angle.
    sm = plt.cm.ScalarMappable(cmap=cmap,
                                norm=plt.Normalize(vmin=0, vmax=360))
    sm.set_array([])
    plt.colorbar(sm, ax=axes, label='Stimulus angle (degrees)',
                 fraction=0.02, pad=0.04)

    plt.suptitle('Ring Attractor Check: DelayGo Hidden State PCA',
                 fontsize=13, y=1.02)
    plt.tight_layout()
    path = os.path.join(SAVE_DIR, 'fig_delaygo_pca.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f'Saved: {path}')
    plt.close()

    print(f'  DelayGo variance explained: PC1={var_exp[0]*100:.1f}%  '
          f'PC2={var_exp[1]*100:.1f}%  PC3={var_exp[2]*100:.1f}%')


# ---------------------------------------------------------------------------
# Figure 4: delaydm1 PCA trajectories (line attractor check)
# ---------------------------------------------------------------------------

def plot_delaydm1_pca(state):
    """
    Plot PCA trajectories for delaydm1.
    Should show two clusters (one per decision) rather than a ring.
    """
    h = state['delaydm1']   # (T, B, 128)
    T, B, N = h.shape

    pca, proj = fit_pca_on_task(h, n_components=3)
    var_exp = pca.explained_variance_ratio_

    # delaydm1 test mode: 40 locs x 4 strengths = 160 trials.
    # Color by which stimulus is stronger (first or second half).
    n_trials = B
    half     = n_trials // 2

    fig, ax = plt.subplots(figsize=(6, 5))

    for i in range(n_trials):
        color = '#4C72B0' if i < half else '#DD8452'
        traj  = proj[:, i, :]
        ax.plot(traj[:, 0], traj[:, 1],
                color=color, alpha=0.4, linewidth=0.8)
        ax.scatter(traj[-1, 0], traj[-1, 1],
                   color=color, s=15, zorder=5)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#4C72B0', label='Stim 1 stronger'),
        Patch(facecolor='#DD8452', label='Stim 2 stronger'),
    ]
    ax.legend(handles=legend_elements, fontsize=10)
    ax.set_xlabel(f'PC1 ({var_exp[0]*100:.1f}%)', fontsize=11)
    ax.set_ylabel(f'PC2 ({var_exp[1]*100:.1f}%)', fontsize=11)
    ax.set_title('DelayDM1: PC1 vs PC2\n(Line/Point Attractor Check)',
                 fontsize=11)

    plt.tight_layout()
    path = os.path.join(SAVE_DIR, 'fig_delaydm1_pca.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f'Saved: {path}')
    plt.close()

    print(f'  DelayDM1 variance explained: PC1={var_exp[0]*100:.1f}%  '
          f'PC2={var_exp[1]*100:.1f}%  PC3={var_exp[2]*100:.1f}%')


# ---------------------------------------------------------------------------
# Figure 5: delaygo + delayanti shared subspace
# ---------------------------------------------------------------------------

def plot_shared_subspace(state):
    """
    Fit PCA on delaygo, then project delayanti onto the same axes.
    If they share a subspace, delayanti trajectories should also form
    a ring in the same PC space — just rotated by pi.
    """
    h_go   = state['delaygo']    # (T, B, 128)
    h_anti = state['delayanti']  # (T, B, 128)

    # Fit PCA on delaygo only.
    pca, proj_go = fit_pca_on_task(h_go, n_components=3)
    var_exp = pca.explained_variance_ratio_

    # Project delayanti onto the same PCA axes.
    proj_anti = project_onto_pca(h_anti, pca)

    n_locs = 40
    cmap   = cm.hsv

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax_idx, (proj, label, alpha) in enumerate([
        (proj_go,   'DelayGo',   0.7),
        (proj_anti, 'DelayAnti', 0.7),
    ]):
        ax = axes[ax_idx]
        for i in range(min(n_locs, proj.shape[1])):
            color = cmap(i / n_locs)
            traj  = proj[:, i, :]
            ax.plot(traj[:, 0], traj[:, 1],
                    color=color, alpha=alpha, linewidth=1.2)
            ax.scatter(traj[-1, 0], traj[-1, 1],
                       color=color, s=20, zorder=5)
        ax.set_xlabel(f'PC1 ({var_exp[0]*100:.1f}%)', fontsize=11)
        ax.set_ylabel(f'PC2 ({var_exp[1]*100:.1f}%)', fontsize=11)
        ax.set_title(f'{label} in DelayGo PCA space', fontsize=11)
        ax.set_aspect('equal')

    sm = plt.cm.ScalarMappable(cmap=cmap,
                                norm=plt.Normalize(vmin=0, vmax=360))
    sm.set_array([])
    plt.colorbar(sm, ax=axes, label='Stimulus angle (degrees)',
                 fraction=0.02, pad=0.04)

    plt.suptitle('Shared Subspace Check: DelayGo vs DelayAnti\n'
                 '(Both projected onto DelayGo PCA axes)',
                 fontsize=13, y=1.02)
    plt.tight_layout()
    path = os.path.join(SAVE_DIR, 'fig_shared_subspace.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f'Saved: {path}')
    plt.close()


# ---------------------------------------------------------------------------
# Figure 6: All tasks overlaid in shared PCA space
# ---------------------------------------------------------------------------

def plot_all_tasks_pca(state):
    """
    Fit PCA on all task hidden states concatenated, then plot each task
    separately. Shows how different tasks occupy the same neural space.
    """
    # Concatenate all tasks along the batch dimension for a global PCA.
    all_states = []
    for task in TASKS:
        h = state[task]   # (T, B, 128)
        T, B, N = h.shape
        all_states.append(h.reshape(T * B, N))

    all_flat = np.concatenate(all_states, axis=0)
    pca = PCA(n_components=3)
    pca.fit(all_flat)
    var_exp = pca.explained_variance_ratio_

    n_tasks = len(TASKS)
    ncols   = 5
    nrows   = (n_tasks + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(ncols * 3.5, nrows * 3.5))
    axes = axes.flatten()

    for idx, task in enumerate(TASKS):
        h    = state[task]
        T, B, N = h.shape
        proj = pca.transform(h.reshape(T * B, N)).reshape(T, B, 3)

        ax    = axes[idx]
        cmap  = cm.hsv
        n_show = min(40, B)
        for i in range(n_show):
            color = cmap(i / n_show)
            traj  = proj[:, i, :]
            ax.plot(traj[:, 0], traj[:, 1],
                    color=color, alpha=0.5, linewidth=0.8)

        ax.set_title(TASK_LABELS[task], fontsize=10)
        ax.set_xlabel('PC1', fontsize=8)
        ax.set_ylabel('PC2', fontsize=8)
        ax.tick_params(labelsize=7)

    # Hide unused subplots.
    for idx in range(n_tasks, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(
        f'All Tasks in Global PCA Space\n'
        f'(PC1={var_exp[0]*100:.1f}%  PC2={var_exp[1]*100:.1f}%  '
        f'PC3={var_exp[2]*100:.1f}%)',
        fontsize=13)
    plt.tight_layout()
    path = os.path.join(SAVE_DIR, 'fig_all_tasks_pca.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f'Saved: {path}')
    plt.close()

    print(f'  Global PCA variance explained: '
          f'PC1={var_exp[0]*100:.1f}%  '
          f'PC2={var_exp[1]*100:.1f}%  '
          f'PC3={var_exp[2]*100:.1f}%')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print('Loading log...')
    log = load_log()

    print('Loading activity files...')
    state, inputs, output = load_activity()

    print('\n--- Training curves ---')
    plot_training_curves(log)

    print('\n--- Performance bar chart ---')
    plot_performance_bar(log)

    print('\n--- DelayGo PCA (ring attractor check) ---')
    plot_delaygo_pca(state)

    print('\n--- DelayDM1 PCA (line attractor check) ---')
    plot_delaydm1_pca(state)

    print('\n--- Shared subspace: DelayGo vs DelayAnti ---')
    plot_shared_subspace(state)

    print('\n--- All tasks in global PCA space ---')
    plot_all_tasks_pca(state)

    print(f'\nAll figures saved to: {SAVE_DIR}')
    print('Open them in VSCode by clicking the .png files in the file explorer.')