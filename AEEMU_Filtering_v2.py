#!/usr/bin/env python3
# ================================================================
# AEEMU_Filtering_v2.py
# ================================================================
# VERSION 2 — Fixes + Integrated SOTA Comparison
#
# FIXES from v1:
#   1. --filter-ablation now respects --dataset (was hardcoded to ml-100k & book-crossing)
#   2. SimpleX baseline replaced with regression-calibrated version
#
# NEW FEATURES:
#   3. --sota-full: runs ALL baselines + AEEMU ensemble on SAME folds
#      → Generates unified LaTeX table with significance markers
#   4. --sota-full-multi: runs --sota-full on all 4 publication datasets
#   5. --best-config: specify which filter config to use for AEEMU
#
# USAGE:
#   # Single dataset SOTA comparison:
#   python AEEMU_Filtering_v2.py --sota-full --dataset ml-100k --folds 10
#   python AEEMU_Filtering_v2.py --sota-full --dataset amazon-music --folds 10
#
#   # All 4 datasets:
#   python AEEMU_Filtering_v2.py --sota-full-multi --folds 10
#
#   # Filter ablation on specific dataset (FIXED):
#   python AEEMU_Filtering_v2.py --filter-ablation --dataset amazon-music --folds 10
#
#   # All v1 commands still work:
#   python AEEMU_Filtering_v2.py --ablation --dataset ml-100k --folds 10
# ================================================================

import sys
import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold, train_test_split
from sklearn.linear_model import Ridge
from scipy.stats import ttest_rel, wilcoxon
from datetime import datetime
from collections import defaultdict
import traceback
import warnings
import gc

warnings.filterwarnings('ignore')

# ================================================================
# Import everything from v1
# ================================================================
_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _script_dir)

from AEEMU_Filtering import (
    # Device & constants
    DEVICE,
    # Base classes
    BaseRecommender, ContextExtractor, ContextExtractorWithFilters,
    # Models
    MatrixFactorization, BayesianNCF, RecVAE, SASRec, LightGCN,
    # SOTA baselines
    MBRCCBaseline, DualChannelGCN,
    # Meta-network
    MetaNeuralNetworkWithFilters, MetaNetworkTrainerWithFilters,
    # Ensemble
    SimplePerformanceWeightedEnsemble,
    # Data loaders
    load_movielens_100k, load_movielens_1m,
    load_amazon_digital_music, load_book_crossing,
    # Utilities
    compute_ranking_metrics, evaluate_model, prepare_data_and_models,
    analyze_error_correlation,
    # Experiment runners (v1)
    run_experiment_with_filters, run_filter_ablation_study,
    compare_with_and_without_filters,
    test_ensemble_combinations,
    # Visualization (v1)
    visualize_filter_architecture, visualize_combination_results,
    # Statistical (v1)
    compute_statistical_significance, generate_latex_significance_table,
    generate_ranking_metrics_table,
    # Multi-dataset (v1)
    consolidate_multi_dataset_results,
)

print("=" * 70)
print("🎓 AEEMU v2 — Integrated SOTA Comparison Pipeline")
print("=" * 70)


# ================================================================
# PREDEFINED BEST FILTER CONFIGURATIONS (from v1 ablation results)
# ================================================================

BEST_FILTER_CONFIGS = {
    'all_filters': {
        'kalman': True, 'wavelet': True, 'spectral': True,
        'adaptive': True, 'ema': True, 'median': True,
        'bilateral': True, 'savgol': True, 'particle': True,
        'confidence': True
    },
    'best_three': {
        'kalman': True, 'adaptive': True, 'ema': True
    },
    'spectral_bilateral': {
        'spectral': True, 'bilateral': True
    },
    'kalman_consensus': {
        'kalman': True, 'consensus': True
    },
    'no_filters': {},
}


# ================================================================
# V2: UNIFIED SOTA COMPARISON
# ================================================================

def compare_with_sota_full(
    dataset: str = 'ml-100k',
    n_folds: int = 10,
    aeemu_filter_configs: list = None,
):
    """
    Unified SOTA comparison: all baselines + AEEMU variants on SAME folds.

    This is the main v2 addition. It:
    1. Loads dataset once
    2. Creates K-fold splits (shared across all methods)
    3. For each fold:
       a. Trains all individual baselines (MF, NCF, SASRec, LightGCN,
          RecVAE, MBRCC, DualChannelGCN)
       b. Trains AEEMU meta-network with specified filter configs
       c. Evaluates everything on the SAME test set
    4. Aggregates results across folds
    5. Computes statistical significance (AEEMU vs each baseline)
    6. Generates publication-ready LaTeX table

    Args:
        dataset: Dataset identifier
        n_folds: Number of cross-validation folds (recommend 10)
        aeemu_filter_configs: List of filter config names to test.
            Default: ['all_filters', 'best_three', 'no_filters']

    Returns:
        dict: Complete results for all methods
    """
    if aeemu_filter_configs is None:
        aeemu_filter_configs = ['all_filters', 'best_three', 'no_filters']

    print("=" * 80)
    print("🏆 UNIFIED SOTA COMPARISON (v2)")
    print(f"📊 Dataset: {dataset}")
    print(f"📊 Folds: {n_folds}")
    print(f"📊 AEEMU configs: {aeemu_filter_configs}")
    print("=" * 80)

    # ------------------------------------------------------------------
    # 1. Load dataset
    # ------------------------------------------------------------------
    dataset_loaders = {
        'ml-100k': ('MovieLens 100K', load_movielens_100k),
        'ml-1m': ('MovieLens 1M', load_movielens_1m),
        'amazon-music': ('Amazon Digital Music', load_amazon_digital_music),
        'book-crossing': ('Book-Crossing', load_book_crossing),
    }
    if dataset not in dataset_loaders:
        raise ValueError(f"Unknown dataset: {dataset}. Options: {list(dataset_loaders.keys())}")

    dataset_name, loader_fn = dataset_loaders[dataset]
    print(f"\n📥 Loading {dataset_name}...")
    df, rating_matrix = loader_fn()
    n_users, n_items = rating_matrix.shape
    print(f"✅ {n_users} users, {n_items} items, {len(df)} ratings")

    # ------------------------------------------------------------------
    # 2. Define all methods to evaluate
    # ------------------------------------------------------------------
    # Individual baselines (each is a factory function)
    baseline_factories = {
        'MF': lambda: MatrixFactorization(n_factors=64),
        'NCF': lambda: BayesianNCF(n_users=n_users, n_items=n_items),
        'SASRec': lambda: SASRec(n_users=n_users, n_items=n_items),
        'LightGCN': lambda: LightGCN(n_users=n_users, n_items=n_items),
        'RecVAE': lambda: RecVAE(n_users=n_users, n_items=n_items),
        'MBRCC': lambda: MBRCCBaseline(n_users=n_users, n_items=n_items),
        'DualGCN': lambda: DualChannelGCN(n_users=n_users, n_items=n_items),
    }

    # AEEMU ensemble base models (the 4 models used in the ensemble)
    aeemu_base_model_names = ['NCF', 'SASRec', 'LightGCN', 'MF']

    # ------------------------------------------------------------------
    # 3. K-fold cross-validation (SHARED splits)
    # ------------------------------------------------------------------
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    # Storage for per-fold results
    # {method_name: {metric: [fold_values]}}
    all_fold_results = defaultdict(lambda: defaultdict(list))

    for fold_idx, (train_index, test_index) in enumerate(kf.split(df)):
        print(f"\n{'='*60}")
        print(f"📂 FOLD {fold_idx + 1}/{n_folds}")
        print(f"{'='*60}")

        train_df = df.iloc[train_index]
        test_df = df.iloc[test_index]

        # Build training matrix for this fold
        train_matrix = np.zeros_like(rating_matrix)
        for _, row in train_df.iterrows():
            train_matrix[int(row['user_id']), int(row['item_id'])] = row['rating']

        # Split training further for AEEMU meta-network validation
        train_df_meta, val_df_meta = train_test_split(
            train_df, test_size=0.15, random_state=42
        )

        # =============================================================
        # 3a. Train & evaluate individual baselines
        # =============================================================
        trained_baselines = {}

        for bl_name, bl_factory in baseline_factories.items():
            print(f"   🔧 Training {bl_name}...")
            try:
                model = bl_factory()
                model.fit(train_matrix, train_df)
                metrics = evaluate_model(model, test_df, train_matrix, compute_ranking=True)

                for mk, mv in metrics.items():
                    all_fold_results[bl_name][mk].append(mv)

                # Keep models needed for AEEMU ensemble
                if bl_name in aeemu_base_model_names:
                    trained_baselines[bl_name] = model

                print(f"      ✅ {bl_name}: RMSE={metrics.get('rmse', 999):.4f}, "
                      f"MAE={metrics.get('mae', 999):.4f}")

            except Exception as e:
                print(f"      ❌ {bl_name} failed: {e}")
                all_fold_results[bl_name]['rmse'].append(999)
                all_fold_results[bl_name]['mae'].append(999)

        # Make sure we have all 4 AEEMU base models
        for name in aeemu_base_model_names:
            if name not in trained_baselines:
                print(f"   ⚠️  {name} not available, training separately for AEEMU...")
                try:
                    factory = baseline_factories[name]
                    model = factory()
                    model.fit(train_matrix, train_df)
                    trained_baselines[name] = model
                except Exception as e:
                    print(f"      ❌ Could not train {name}: {e}")

        # =============================================================
        # 3b. Simple weighted ensemble (no meta-network, no filters)
        # =============================================================
        if len(trained_baselines) >= 2:
            print(f"   🔧 Evaluating Simple Weighted Ensemble...")
            try:
                # Get validation RMSE for weighting
                val_performances = {}
                for name, model in trained_baselines.items():
                    val_metrics = evaluate_model(model, val_df_meta, train_matrix, compute_ranking=False)
                    val_performances[name] = val_metrics.get('rmse', 1.0)

                simple_ens = SimplePerformanceWeightedEnsemble(val_performances, verbose=False)

                # Predict on test set
                simple_preds = []
                true_ratings = []
                user_preds_simple = defaultdict(list)

                for _, row in test_df.iterrows():
                    u, i = int(row['user_id']), int(row['item_id'])
                    if u < train_matrix.shape[0] and i < train_matrix.shape[1]:
                        base_preds = {}
                        for name, model in trained_baselines.items():
                            base_preds[name] = model.predict_clipped(u, i)
                        pred = simple_ens.predict(base_preds)
                        simple_preds.append(pred)
                        true_ratings.append(row['rating'])
                        user_preds_simple[u].append((i, pred, row['rating']))

                if simple_preds:
                    rmse = np.sqrt(mean_squared_error(true_ratings, simple_preds))
                    mae = mean_absolute_error(true_ratings, simple_preds)
                    all_fold_results['SimpleWtdEns']['rmse'].append(rmse)
                    all_fold_results['SimpleWtdEns']['mae'].append(mae)

                    # Ranking metrics
                    filtered = {u: p for u, p in user_preds_simple.items() if len(p) >= 5}
                    if filtered:
                        rank_m = compute_ranking_metrics(filtered, k_values=[5, 10, 20])
                        for mk, mv in rank_m.items():
                            all_fold_results['SimpleWtdEns'][mk].append(mv)

                    print(f"      ✅ SimpleWtdEns: RMSE={rmse:.4f}, MAE={mae:.4f}")

            except Exception as e:
                print(f"      ❌ SimpleWtdEns failed: {e}")

        # =============================================================
        # 3c. AEEMU ensemble with different filter configs
        # =============================================================
        for config_name in aeemu_filter_configs:
            filter_config = BEST_FILTER_CONFIGS.get(config_name, {})
            use_filters = len(filter_config) > 0
            method_name = f"AEEMU_{config_name}"

            print(f"   🔧 Training {method_name}...")

            try:
                aeemu_results = _run_aeemu_single_fold(
                    trained_baselines=trained_baselines,
                    train_df=train_df,
                    train_df_meta=train_df_meta,
                    val_df_meta=val_df_meta,
                    test_df=test_df,
                    train_matrix=train_matrix,
                    use_filters=use_filters,
                    filter_config=filter_config,
                )

                for mk, mv in aeemu_results.items():
                    all_fold_results[method_name][mk].append(mv)

                print(f"      ✅ {method_name}: RMSE={aeemu_results.get('rmse', 999):.4f}, "
                      f"MAE={aeemu_results.get('mae', 999):.4f}")

            except Exception as e:
                print(f"      ❌ {method_name} failed: {e}")
                traceback.print_exc()
                all_fold_results[method_name]['rmse'].append(999)
                all_fold_results[method_name]['mae'].append(999)

        # Cleanup GPU memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # 4. Aggregate results
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("📊 AGGREGATING RESULTS")
    print("=" * 80)

    summary = {}
    for method_name, metrics_dict in all_fold_results.items():
        method_summary = {}
        for metric_name, values in metrics_dict.items():
            valid_values = [v for v in values if v < 900]  # Filter out failures
            if valid_values:
                method_summary[f'mean_{metric_name}'] = float(np.mean(valid_values))
                method_summary[f'std_{metric_name}'] = float(np.std(valid_values))
                method_summary[f'fold_{metric_name}s'] = [float(v) for v in valid_values]
            else:
                method_summary[f'mean_{metric_name}'] = 999.0
                method_summary[f'std_{metric_name}'] = 0.0
        method_summary['n_successful_folds'] = len([
            v for v in metrics_dict.get('rmse', []) if v < 900
        ])
        summary[method_name] = method_summary

    # ------------------------------------------------------------------
    # 5. Print results table
    # ------------------------------------------------------------------
    print("\n" + "=" * 100)
    print(f"🏆 UNIFIED RESULTS — {dataset_name} ({n_folds}-fold CV)")
    print("=" * 100)

    # Sort by mean RMSE
    sorted_methods = sorted(
        summary.items(),
        key=lambda x: x[1].get('mean_rmse', 999)
    )

    header = (f"{'Rank':<5} {'Method':<25} {'RMSE':<18} {'MAE':<18} "
              f"{'NDCG@10':<12} {'HR@10':<12} {'Folds':<6}")
    print(header)
    print("-" * 100)

    for rank, (method_name, method_data) in enumerate(sorted_methods, 1):
        rmse_mean = method_data.get('mean_rmse', 999)
        rmse_std = method_data.get('std_rmse', 0)
        mae_mean = method_data.get('mean_mae', 999)
        ndcg10 = method_data.get('mean_ndcg@10', 0)
        hr10 = method_data.get('mean_hr@10', 0)
        n_folds_ok = method_data.get('n_successful_folds', 0)

        print(f"{rank:<5} {method_name:<25} {rmse_mean:.4f}±{rmse_std:.4f}   "
              f"{mae_mean:.4f}            {ndcg10:.4f}       {hr10:.4f}       "
              f"{n_folds_ok}/{n_folds}")

    # ------------------------------------------------------------------
    # 6. Statistical significance
    # ------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("📊 STATISTICAL SIGNIFICANCE (vs best AEEMU)")
    print("=" * 100)

    # Find best AEEMU config
    aeemu_methods = {k: v for k, v in summary.items() if k.startswith('AEEMU_')}
    if aeemu_methods:
        best_aeemu_name = min(aeemu_methods, key=lambda k: aeemu_methods[k].get('mean_rmse', 999))
        best_aeemu = aeemu_methods[best_aeemu_name]
        best_aeemu_folds = best_aeemu.get('fold_rmses', [])

        print(f"\nBest AEEMU: {best_aeemu_name} (RMSE: {best_aeemu.get('mean_rmse', 999):.4f})")

        significance_results = {}

        if len(best_aeemu_folds) >= 5:
            baseline_arr = np.array(best_aeemu_folds)

            for method_name, method_data in sorted_methods:
                if method_name == best_aeemu_name:
                    continue

                other_folds = method_data.get('fold_rmses', [])
                if len(other_folds) < 5:
                    continue

                other_arr = np.array(other_folds[:len(baseline_arr)])
                if len(other_arr) != len(baseline_arr):
                    continue

                # Paired t-test (AEEMU vs this method)
                # Positive t → this method has higher RMSE → AEEMU is better
                try:
                    _, t_p = ttest_rel(other_arr, baseline_arr)
                except:
                    t_p = 1.0

                # Wilcoxon
                try:
                    diffs = other_arr - baseline_arr
                    if np.any(diffs != 0):
                        _, w_p = wilcoxon(diffs)
                    else:
                        w_p = 1.0
                except:
                    w_p = 1.0

                # Direction: is AEEMU better?
                delta = float(np.mean(other_arr) - np.mean(baseline_arr))
                improvement = delta / np.mean(other_arr) * 100 if np.mean(other_arr) > 0 else 0

                if t_p < 0.001:
                    sig = "***"
                elif t_p < 0.01:
                    sig = "**"
                elif t_p < 0.05:
                    sig = "*"
                else:
                    sig = "n.s."

                significance_results[method_name] = {
                    'mean_rmse': method_data.get('mean_rmse', 999),
                    'aeemu_rmse': best_aeemu.get('mean_rmse', 999),
                    'delta_rmse': delta,
                    'improvement_%': improvement,
                    't_test_p': float(t_p),
                    'wilcoxon_p': float(w_p),
                    'significance': sig,
                }

            # Print significance table
            print(f"\n{'Method':<25} {'RMSE':<12} {'Δ RMSE':<12} {'Improv.':<10} "
                  f"{'t-test p':<12} {'Wilcoxon p':<12} {'Sig':<6}")
            print("-" * 95)

            for method_name, sig_data in sorted(
                significance_results.items(),
                key=lambda x: -x[1]['improvement_%']
            ):
                print(f"{method_name:<25} {sig_data['mean_rmse']:<12.4f} "
                      f"{sig_data['delta_rmse']:<+12.4f} "
                      f"{sig_data['improvement_%']:<+10.2f}% "
                      f"{sig_data['t_test_p']:<12.6f} "
                      f"{sig_data['wilcoxon_p']:<12.6f} "
                      f"{sig_data['significance']:<6}")
        else:
            significance_results = {}
            print("⚠️  Not enough folds for significance testing")
    else:
        best_aeemu_name = None
        significance_results = {}

    # ------------------------------------------------------------------
    # 7. Generate LaTeX table
    # ------------------------------------------------------------------
    latex_table = _generate_sota_latex_table(
        summary, sorted_methods, significance_results,
        best_aeemu_name, dataset_name, n_folds
    )

    # ------------------------------------------------------------------
    # 8. Save everything
    # ------------------------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_tag = dataset.replace('-', '_')
    os.makedirs('results', exist_ok=True)

    # Save JSON
    output = {
        'timestamp': timestamp,
        'dataset': dataset,
        'dataset_name': dataset_name,
        'n_folds': n_folds,
        'aeemu_filter_configs': aeemu_filter_configs,
        'summary': summary,
        'significance': significance_results,
        'best_aeemu': best_aeemu_name,
    }

    json_path = f"results/sota_full_{dataset_tag}_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n💾 JSON results: {json_path}")

    # Save LaTeX
    latex_path = f"results/sota_full_table_{dataset_tag}_{timestamp}.tex"
    with open(latex_path, 'w') as f:
        f.write(latex_table)
    print(f"💾 LaTeX table: {latex_path}")

    # Print LaTeX to console
    print("\n" + "=" * 100)
    print("📄 LATEX TABLE (copy to your .tex file):")
    print("=" * 100)
    print(latex_table)

    return output


def _run_aeemu_single_fold(
    trained_baselines: dict,
    train_df: pd.DataFrame,
    train_df_meta: pd.DataFrame,
    val_df_meta: pd.DataFrame,
    test_df: pd.DataFrame,
    train_matrix: np.ndarray,
    use_filters: bool,
    filter_config: dict,
) -> dict:
    """
    Run AEEMU ensemble for a single fold.
    Returns dict with rmse, mae, ndcg@K, hr@K, etc.
    """
    # Get validation performances for meta-network
    model_performances = {}
    for name, model in trained_baselines.items():
        val_metrics = evaluate_model(model, val_df_meta, train_matrix, compute_ranking=False)
        model_performances[name] = val_metrics.get('rmse', 1.0)

    # Context extractor
    context_extractor = ContextExtractorWithFilters(use_filters=use_filters, verbose=False)
    context_extractor.fit(train_matrix, train_df)
    context_extractor.set_model_performances(model_performances)

    # Embedding dimensions
    base_embedding_dims = {}
    for name, model in trained_baselines.items():
        u_emb, i_emb = model.get_embeddings()
        base_embedding_dims[name] = u_emb.shape[1] + i_emb.shape[1]

    # Meta-network
    meta_network = MetaNeuralNetworkWithFilters(
        base_embedding_dims=base_embedding_dims,
        context_dim=context_extractor.total_context_dim,
        performance_prior=model_performances,
        use_filters=use_filters,
    )

    # Trainer
    trainer = MetaNetworkTrainerWithFilters(
        meta_network=meta_network,
        model_performances=model_performances,
        use_filters=use_filters,
        performance_alpha=5.0,
        dominance_weight=10.0,
        filter_config=filter_config,
    )

    # Prepare data loaders
    train_dataset = TensorDataset(
        torch.from_numpy(train_df_meta['user_id'].values.astype(np.int64)),
        torch.from_numpy(train_df_meta['item_id'].values.astype(np.int64)),
        torch.from_numpy(train_df_meta['rating'].values.astype(np.float32)),
    )
    val_dataset = TensorDataset(
        torch.from_numpy(val_df_meta['user_id'].values.astype(np.int64)),
        torch.from_numpy(val_df_meta['item_id'].values.astype(np.int64)),
        torch.from_numpy(val_df_meta['rating'].values.astype(np.float32)),
    )
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

    # Train meta-network (1 epoch — same as v1)
    trainer.train_epoch(train_loader, trained_baselines, context_extractor, train_matrix, val_loader)

    # Evaluate on test set
    meta_network.eval()
    meta_network.set_filter_mode('inference')

    ensemble_preds = []
    true_ratings = []
    user_preds = defaultdict(list)

    with torch.no_grad():
        for _, row in test_df.iterrows():
            u, i = int(row['user_id']), int(row['item_id'])
            if u < train_matrix.shape[0] and i < train_matrix.shape[1]:
                # Base predictions
                base_predictions = {
                    name: torch.tensor(model.predict_clipped(u, i), device=DEVICE)
                    for name, model in trained_baselines.items()
                }

                # Embeddings
                embeddings = {
                    name: torch.FloatTensor(
                        model.get_embedding_for_pair(u, i).astype(np.float32)
                    ).unsqueeze(0).to(DEVICE)
                    for name, model in trained_baselines.items()
                }

                # Context
                ctx = context_extractor.extract_context_vector(u, i, 0, train_matrix, train_df)
                ctx_tensor = torch.FloatTensor(ctx.astype(np.float32)).unsqueeze(0).to(DEVICE)

                # Ensemble prediction
                pred = meta_network.compute_ensemble_prediction(
                    base_predictions, embeddings, ctx_tensor
                )

                pred_val = pred.cpu().item()
                ensemble_preds.append(pred_val)
                true_ratings.append(row['rating'])
                user_preds[u].append((i, pred_val, row['rating']))

    if not ensemble_preds:
        return {'rmse': 999, 'mae': 999}

    results = {
        'rmse': float(np.sqrt(mean_squared_error(true_ratings, ensemble_preds))),
        'mae': float(mean_absolute_error(true_ratings, ensemble_preds)),
    }

    # Ranking metrics
    filtered = {u: p for u, p in user_preds.items() if len(p) >= 5}
    if filtered:
        rank_m = compute_ranking_metrics(filtered, k_values=[5, 10, 20])
        results.update(rank_m)

    return results


def _generate_sota_latex_table(
    summary: dict,
    sorted_methods: list,
    significance_results: dict,
    best_aeemu_name: str,
    dataset_name: str,
    n_folds: int,
) -> str:
    """Generate publication-ready LaTeX table."""

    lines = []
    lines.append(r'\begin{table}[htbp]')
    lines.append(r'\centering')
    lines.append(r'\caption{Comparison with SOTA baselines on ' + dataset_name +
                 r' (' + str(n_folds) + r'-fold CV).}')
    lines.append(r'\label{tab:sota-' + dataset_name.lower().replace(' ', '-') + r'}')
    lines.append(r'\begin{adjustbox}{max width=\textwidth}')
    lines.append(r'\begin{tabular}{l ccc cc c}')
    lines.append(r'\toprule')
    lines.append(r'\textbf{Method} & \textbf{RMSE} & \textbf{MAE} & '
                 r'\textbf{NDCG@10} & \textbf{HR@10} & '
                 r'\textbf{$\Delta$RMSE} & \textbf{Sig.} \\')
    lines.append(r'\midrule')

    # Separate baselines from AEEMU
    baselines = [(n, d) for n, d in sorted_methods if not n.startswith('AEEMU_')]
    aeemu_methods = [(n, d) for n, d in sorted_methods if n.startswith('AEEMU_')]

    # Individual baselines section
    lines.append(r'\multicolumn{7}{l}{\textit{Individual baselines}} \\')
    for method_name, method_data in baselines:
        if method_name == 'SimpleWtdEns':
            continue  # Print separately

        rmse_m = method_data.get('mean_rmse', 999)
        rmse_s = method_data.get('std_rmse', 0)
        mae_m = method_data.get('mean_mae', 999)
        ndcg = method_data.get('mean_ndcg@10', 0)
        hr = method_data.get('mean_hr@10', 0)

        sig_info = significance_results.get(method_name, {})
        delta = sig_info.get('delta_rmse', 0)
        sig_marker = sig_info.get('significance', '')

        latex_name = method_name.replace('_', r'\_')
        delta_str = f'{delta:+.4f}' if delta != 0 else '---'
        sig_str = sig_marker if sig_marker else '---'

        lines.append(f'{latex_name} & {rmse_m:.4f}$\\pm${rmse_s:.4f} & {mae_m:.4f} & '
                     f'{ndcg:.4f} & {hr:.4f} & {delta_str} & {sig_str} \\\\')

    # Simple ensemble
    lines.append(r'\midrule')
    lines.append(r'\multicolumn{7}{l}{\textit{Simple ensemble}} \\')
    for method_name, method_data in baselines:
        if method_name != 'SimpleWtdEns':
            continue
        rmse_m = method_data.get('mean_rmse', 999)
        rmse_s = method_data.get('std_rmse', 0)
        mae_m = method_data.get('mean_mae', 999)
        ndcg = method_data.get('mean_ndcg@10', 0)
        hr = method_data.get('mean_hr@10', 0)
        sig_info = significance_results.get(method_name, {})
        delta = sig_info.get('delta_rmse', 0)
        sig_marker = sig_info.get('significance', '')
        delta_str = f'{delta:+.4f}' if delta != 0 else '---'
        sig_str = sig_marker if sig_marker else '---'
        lines.append(f'SimpleWtdEns & {rmse_m:.4f}$\\pm${rmse_s:.4f} & {mae_m:.4f} & '
                     f'{ndcg:.4f} & {hr:.4f} & {delta_str} & {sig_str} \\\\')

    # AEEMU variants
    lines.append(r'\midrule')
    lines.append(r'\multicolumn{7}{l}{\textit{AEEMU (proposed)}} \\')
    for method_name, method_data in aeemu_methods:
        rmse_m = method_data.get('mean_rmse', 999)
        rmse_s = method_data.get('std_rmse', 0)
        mae_m = method_data.get('mean_mae', 999)
        ndcg = method_data.get('mean_ndcg@10', 0)
        hr = method_data.get('mean_hr@10', 0)

        latex_name = method_name.replace('_', r'\_')

        # Bold the best AEEMU
        if method_name == best_aeemu_name:
            lines.append(f'\\textbf{{{latex_name}}} & \\textbf{{{rmse_m:.4f}$\\pm${rmse_s:.4f}}} & '
                         f'\\textbf{{{mae_m:.4f}}} & \\textbf{{{ndcg:.4f}}} & '
                         f'\\textbf{{{hr:.4f}}} & --- & --- \\\\')
        else:
            lines.append(f'{latex_name} & {rmse_m:.4f}$\\pm${rmse_s:.4f} & {mae_m:.4f} & '
                         f'{ndcg:.4f} & {hr:.4f} & --- & --- \\\\')

    lines.append(r'\bottomrule')
    lines.append(r'\multicolumn{7}{l}{\footnotesize $\Delta$RMSE: difference vs best AEEMU '
                 r'(positive = AEEMU is better). Sig.: *** $p<$0.001, ** $p<$0.01, * $p<$0.05} \\')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{adjustbox}')
    lines.append(r'\end{table}')

    return '\n'.join(lines)


# ================================================================
# V2: MULTI-DATASET SOTA COMPARISON
# ================================================================

def compare_with_sota_full_multi(n_folds: int = 10):
    """
    Run unified SOTA comparison on all 4 publication datasets.
    Generates a consolidated cross-dataset table.
    """
    print("🌍" * 40)
    print("=" * 80)
    print("🏆 MULTI-DATASET UNIFIED SOTA COMPARISON (v2)")
    print("=" * 80)

    datasets = ['ml-100k', 'ml-1m', 'amazon-music', 'book-crossing']
    all_results = {}

    for ds in datasets:
        print(f"\n{'🎯' * 40}")
        print(f"📊 DATASET: {ds.upper()}")
        print(f"{'🎯' * 40}")

        try:
            result = compare_with_sota_full(
                dataset=ds,
                n_folds=n_folds,
                aeemu_filter_configs=['all_filters', 'best_three', 'no_filters'],
            )
            all_results[ds] = result
        except Exception as e:
            print(f"❌ Error with {ds}: {e}")
            traceback.print_exc()
            all_results[ds] = None

    # Generate consolidated cross-dataset table
    _generate_cross_dataset_table(all_results, n_folds)

    # Save all results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"results/sota_full_multi_{timestamp}.json"
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n💾 All results saved: {output_path}")

    return all_results


def _generate_cross_dataset_table(all_results: dict, n_folds: int):
    """Generate consolidated LaTeX table across all datasets."""

    print("\n" + "=" * 100)
    print("📄 CROSS-DATASET COMPARISON TABLE")
    print("=" * 100)

    lines = []
    lines.append(r'\begin{table*}[htbp]')
    lines.append(r'\centering')
    lines.append(r'\caption{Cross-dataset comparison of AEEMU with SOTA baselines '
                 r'(' + str(n_folds) + r'-fold CV). Best results in \textbf{bold}.}')
    lines.append(r'\label{tab:cross-dataset-sota}')
    lines.append(r'\begin{adjustbox}{max width=\textwidth}')

    dataset_names = list(all_results.keys())
    n_datasets = len(dataset_names)

    col_spec = 'l ' + ' '.join(['cc'] * n_datasets)
    lines.append(r'\begin{tabular}{' + col_spec + r'}')
    lines.append(r'\toprule')

    # Header row 1: dataset names
    header1 = r'\textbf{Method}'
    for ds in dataset_names:
        header1 += r' & \multicolumn{2}{c}{\textbf{' + ds.upper().replace('-', r'\_') + r'}}'
    header1 += r' \\'
    lines.append(header1)

    # Header row 2: metrics
    header2 = ''
    for ds in dataset_names:
        header2 += r' & \textbf{RMSE} & \textbf{MAE}'
    header2 += r' \\'
    lines.append(r'\cmidrule(lr){2-' + str(1 + 2 * n_datasets) + r'}')
    lines.append(header2)
    lines.append(r'\midrule')

    # Collect all method names across datasets
    all_methods = set()
    for ds, result in all_results.items():
        if result and 'summary' in result:
            all_methods.update(result['summary'].keys())

    # Order: individual baselines first, then SimpleWtdEns, then AEEMU
    individual = sorted([m for m in all_methods
                        if not m.startswith('AEEMU_') and m != 'SimpleWtdEns'])
    aeemu = sorted([m for m in all_methods if m.startswith('AEEMU_')])
    method_order = individual + ['SimpleWtdEns'] + aeemu

    # Find best RMSE per dataset
    best_rmse_per_dataset = {}
    for ds, result in all_results.items():
        if result and 'summary' in result:
            rmses = {m: d.get('mean_rmse', 999) for m, d in result['summary'].items()}
            best_rmse_per_dataset[ds] = min(rmses.values())

    for method in method_order:
        if method not in all_methods:
            continue

        if method == 'SimpleWtdEns':
            lines.append(r'\midrule')

        if method == aeemu[0] if aeemu else None:
            lines.append(r'\midrule')

        latex_name = method.replace('_', r'\_')
        row = latex_name

        for ds in dataset_names:
            result = all_results.get(ds)
            if result and 'summary' in result and method in result['summary']:
                data = result['summary'][method]
                rmse = data.get('mean_rmse', 999)
                mae = data.get('mean_mae', 999)

                if rmse < 900:
                    rmse_str = f'{rmse:.4f}'
                    mae_str = f'{mae:.4f}'
                    # Bold if best
                    if abs(rmse - best_rmse_per_dataset.get(ds, 999)) < 0.0001:
                        rmse_str = r'\textbf{' + rmse_str + r'}'
                        mae_str = r'\textbf{' + mae_str + r'}'
                    row += f' & {rmse_str} & {mae_str}'
                else:
                    row += r' & --- & ---'
            else:
                row += r' & --- & ---'

        row += r' \\'
        lines.append(row)

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{adjustbox}')
    lines.append(r'\end{table*}')

    table = '\n'.join(lines)

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"results/cross_dataset_table_{timestamp}.tex"
    with open(path, 'w') as f:
        f.write(table)
    print(f"💾 Cross-dataset table: {path}")
    print(table)

    return table


# ================================================================
# V2: FIXED filter-ablation (respects --dataset)
# ================================================================

def run_filter_ablation_v2(n_folds: int = 10, dataset: str = 'ml-100k'):
    """
    Fixed version: runs ablation on the SPECIFIED dataset only
    (v1 was hardcoded to ml-100k & book-crossing).
    """
    print("=" * 80)
    print(f"🔬 FILTER ABLATION STUDY (v2 — single dataset)")
    print(f"📊 Dataset: {dataset}")
    print(f"📊 Folds: {n_folds}")
    print("=" * 80)

    results = run_filter_ablation_study(n_folds=n_folds, dataset=dataset)

    print(f"\n✅ Filter ablation completed for {dataset}")
    return results


# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AEEMU v2 — Integrated SOTA Comparison Pipeline"
    )

    # ---- V2 new commands ----
    parser.add_argument(
        '--sota-full',
        action='store_true',
        help="[V2] Run unified SOTA comparison: all baselines + AEEMU on same folds."
    )
    parser.add_argument(
        '--sota-full-multi',
        action='store_true',
        help="[V2] Run --sota-full on all 4 datasets (ml-100k, ml-1m, amazon-music, book-crossing)."
    )
    parser.add_argument(
        '--best-config',
        type=str,
        nargs='+',
        default=None,
        help="[V2] Which AEEMU filter configs to test. "
             "Options: all_filters, best_three, spectral_bilateral, kalman_consensus, no_filters. "
             "Default: all_filters best_three no_filters"
    )

    # ---- V1 commands (still supported) ----
    parser.add_argument('--filter-ablation', action='store_true',
                        help="Run comprehensive filter ablation study (FIXED: respects --dataset).")
    parser.add_argument('--ablation', action='store_true',
                        help="Run with/without filters comparison.")
    parser.add_argument('--folds', type=int, default=10,
                        help="Number of cross-validation folds.")
    parser.add_argument('--dataset', type=str, default='ml-100k',
                        choices=['ml-100k', 'ml-1m', 'amazon-music', 'book-crossing'],
                        help="Dataset to use.")
    parser.add_argument('--filters', action='store_true',
                        help="Run single experiment with filters enabled.")
    parser.add_argument('--no-filters', action='store_true',
                        help="Run single experiment with filters disabled.")
    parser.add_argument('--name', type=str, default='default_experiment',
                        help="Experiment name.")
    parser.add_argument('--test-combinations', action='store_true',
                        help="Test all model combinations.")
    parser.add_argument('--visualize-filters', action='store_true',
                        help="Generate filter architecture visualization.")
    parser.add_argument('--visualize-results', type=str, metavar='JSON_FILE',
                        help="Visualize combination results from JSON.")
    parser.add_argument('--full-pipeline', action='store_true',
                        help="Run complete v1 pipeline.")
    parser.add_argument('--multi-dataset', action='store_true',
                        help="[V1] Run filter ablation on all datasets (v1 style).")
    parser.add_argument('--sota-comparison', action='store_true',
                        help="[V1] Compare with SOTA baselines (v1 style, baselines only).")

    args = parser.parse_args()

    # ================================================================
    # V2 COMMANDS
    # ================================================================
    if args.sota_full_multi:
        compare_with_sota_full_multi(n_folds=args.folds)

    elif args.sota_full:
        configs = args.best_config or ['all_filters', 'best_three', 'no_filters']
        compare_with_sota_full(
            dataset=args.dataset,
            n_folds=args.folds,
            aeemu_filter_configs=configs,
        )

    elif args.filter_ablation:
        # V2 FIX: respect --dataset flag
        run_filter_ablation_v2(n_folds=args.folds, dataset=args.dataset)

    # ================================================================
    # V1 COMMANDS (pass-through)
    # ================================================================
    elif args.ablation:
        compare_with_and_without_filters(n_folds=args.folds, dataset=args.dataset)

    elif args.sota_comparison:
        from AEEMU_Filtering import compare_with_sota_baselines
        df, rating_matrix, _ = prepare_data_and_models(dataset=args.dataset)
        compare_with_sota_baselines(df, rating_matrix, n_folds=args.folds)

    elif args.multi_dataset:
        # V1 multi-dataset (unchanged)
        from AEEMU_Filtering import run_filter_ablation_study as v1_ablation
        datasets = ['ml-100k', 'ml-1m', 'amazon-music', 'book-crossing']
        all_results = {}
        for ds in datasets:
            try:
                results = v1_ablation(n_folds=args.folds, dataset=ds)
                all_results[ds] = results
            except Exception as e:
                print(f"❌ {ds}: {e}")
                all_results[ds] = {}
        if any(all_results.values()):
            consolidate_multi_dataset_results(all_results)

    elif args.test_combinations:
        test_ensemble_combinations(n_folds=args.folds, use_filters=not args.no_filters)

    elif args.visualize_filters:
        visualize_filter_architecture()

    elif args.visualize_results:
        visualize_combination_results(args.visualize_results)

    elif args.full_pipeline:
        # Run v1 full pipeline
        compare_with_and_without_filters(n_folds=args.folds, dataset=args.dataset)
        test_ensemble_combinations(n_folds=args.folds, use_filters=True)
        test_ensemble_combinations(n_folds=args.folds, use_filters=False)
        visualize_filter_architecture()

    else:
        # Default: single experiment
        use_filters = not args.no_filters
        df, rating_matrix, base_models = prepare_data_and_models(dataset=args.dataset)
        results = run_experiment_with_filters(
            base_models=base_models,
            df=df,
            rating_matrix=rating_matrix,
            n_folds=args.folds,
            experiment_name=args.name,
            use_filters=use_filters,
        )

        print(f"\n{'='*80}")
        print(f"📊 RESULTS: {args.name}")
        print(f"   Ensemble RMSE: {results.get('ensemble_rmse', 'N/A'):.4f}")
        print(f"   Simple Ens RMSE: {results.get('simple_ensemble_rmse', 'N/A'):.4f}")
        print(f"{'='*80}")
