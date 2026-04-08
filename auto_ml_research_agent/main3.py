#!/usr/bin/env python3
"""
Alternative entrypoint with a different dataset acquisition strategy.

Strategy order:
1) Kaggle API search (if available/authenticated)
2) Browser search for Kaggle refs
3) Download via kagglehub using dataset ref
4) Fallback to Kaggle API dataset download
5) Fallback to browser download

Then run the normal training pipeline using the resolved local CSV path.
"""
from __future__ import annotations

import sys
import tempfile
import zipfile
import os
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import re
from datetime import datetime

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from auto_ml_research_agent.config import load_config
from auto_ml_research_agent.dataset.browser_agent import BrowserAgent


def _pick_best_csv(root: Path) -> Optional[Path]:
    csvs = sorted(root.rglob("*.csv"), key=lambda p: p.stat().st_size, reverse=True)
    return csvs[0] if csvs else None


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


def _relevance_score(problem: str, candidate: Dict[str, Any]) -> int:
    """Simple keyword overlap ranking to avoid unrelated Kaggle results."""
    p = _tokenize(problem)
    name = candidate.get("name", "")
    ref = candidate.get("kaggle_ref", "")
    desc = candidate.get("description", "")
    c = _tokenize(f"{name} {ref} {desc}")
    overlap = len(p.intersection(c))
    # Boost if entire key phrase words appear in ref/name.
    boost_terms = ["kidney", "cancer", "disease", "ckd", "renal"]
    boost = sum(1 for t in boost_terms if t in c and t in p)
    return overlap + boost


def _filter_rank_candidates(problem: str, candidates: List[Dict[str, Any]], limit: int = 5) -> List[Dict[str, Any]]:
    scored = [(c, _relevance_score(problem, c)) for c in candidates]
    scored.sort(key=lambda x: x[1], reverse=True)
    # Keep only meaningfully related rows first; fallback to top-N if all zero.
    related = [c for c, s in scored if s > 0]
    if related:
        return related[:limit]
    return [c for c, _ in scored[:limit]]


def _find_likely_target_column(df: pd.DataFrame) -> Optional[str]:
    lowered = {c.lower(): c for c in df.columns}
    hints = ["target", "label", "class", "outcome", "diagnosis", "stage", "result"]
    for h in hints:
        if h in lowered:
            return lowered[h]
    return None


def _is_leaky_dataset(df: pd.DataFrame) -> tuple[bool, str]:
    """
    Quick leakage heuristic: if many single features deterministically map to target,
    dataset is likely synthetic/rule-generated for this task.
    """
    target = _find_likely_target_column(df)
    if not target:
        return False, "no obvious target column for leakage precheck"
    X = df.drop(columns=[target])
    deterministic_cols = []
    for c in X.columns:
        try:
            if (df.groupby(c, dropna=False)[target].nunique(dropna=False) <= 1).all():
                deterministic_cols.append(c)
        except Exception:
            continue
    # One deterministic medical biomarker can happen; many is usually leakage/synthetic labels.
    if len(deterministic_cols) >= 3:
        return True, f"deterministic target mapping via {len(deterministic_cols)} features: {deterministic_cols[:6]}"
    return False, f"deterministic features count={len(deterministic_cols)}"


def _search_kaggle_api(problem: str, limit: int = 5) -> List[Dict[str, Any]]:
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        rows = list(api.dataset_list(search=problem, sort_by="votes", max_size=limit))
        out: List[Dict[str, Any]] = []
        for ds in rows[:limit]:
            ref = getattr(ds, "ref", None)
            if not ref:
                continue
            out.append(
                {
                    "name": f"kaggle_{ref.replace('/', '_')}",
                    "kaggle_ref": ref,
                    "source": "kaggle_api",
                    "url": f"https://www.kaggle.com/datasets/{ref}",
                }
            )
        return _filter_rank_candidates(problem, out, limit=limit)
    except Exception:
        return []


def _download_with_kagglehub(dataset_ref: str, workdir: Path) -> Optional[Path]:
    try:
        import kagglehub  # type: ignore
    except Exception:
        return None
    try:
        downloaded = Path(kagglehub.dataset_download(dataset_ref))
        candidate_root = downloaded if downloaded.is_dir() else downloaded.parent
        csv = _pick_best_csv(candidate_root)
        if csv:
            return csv
        if downloaded.suffix.lower() == ".zip":
            extract_dir = workdir / f"{downloaded.stem}_extracted"
            extract_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(downloaded, "r") as zf:
                zf.extractall(extract_dir)
            return _pick_best_csv(extract_dir)
    except Exception:
        return None
    return None


def _download_with_kaggle_api(dataset_ref: str, workdir: Path) -> Optional[Path]:
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(dataset=dataset_ref, path=str(workdir), unzip=True)
        return _pick_best_csv(workdir)
    except Exception:
        return None


def resolve_dataset(problem: str) -> Optional[Path]:
    config = load_config()
    browser = BrowserAgent(
        download_dir="data/raw",
        timeout=config.download_timeout,
        auth_state_path=config.playwright_auth_state_path,
        headless=config.playwright_headless,
    )

    # NOTE: Existing main.py strategy is preserved there; this is an alternate flow.
    candidates = _search_kaggle_api(problem, limit=5)
    if not candidates:
        browser_candidates = browser.search_kaggle_web(problem, max_results=20)
        candidates = _filter_rank_candidates(problem, browser_candidates, limit=5)
    if not candidates:
        return None

    with tempfile.TemporaryDirectory() as td:
        workdir = Path(td)
        for c in candidates:
            ref = c.get("kaggle_ref")
            if not ref:
                continue
            print(f"[main3] Trying dataset ref: {ref}")
            rejected_for_leakage = False

            # Strategy: kagglehub-first and kagglehub-only for consistent behavior/speed.
            csv = _download_with_kagglehub(ref, workdir)
            if csv and csv.exists():
                df = pd.read_csv(csv)
                leaky, reason = _is_leaky_dataset(df)
                if leaky:
                    print(f"[main3] Rejecting {ref}: suspected leakage ({reason})")
                    rejected_for_leakage = True
                else:
                    final_csv = Path("data/raw") / f"main3_{csv.name}"
                    final_csv.parent.mkdir(parents=True, exist_ok=True)
                    df.to_csv(final_csv, index=False)
                    print(f"[main3] Downloaded via kagglehub: {final_csv}")
                    return final_csv
            if rejected_for_leakage:
                # Once leaky, move to next candidate directly.
                continue
            print(f"[main3] Skipping {ref}: kagglehub did not yield a usable dataset in this pass.")
    return None


def main(problem: str, dataset_path: Optional[str] = None, max_iterations_override: Optional[int] = None) -> Dict[str, Any]:
    if dataset_path:
        resolved = Path(dataset_path)
    else:
        resolved = resolve_dataset(problem)
        if resolved is None:
            raise RuntimeError("main3 dataset resolution failed")

    # Reuse existing pipeline execution path with explicit dataset path.
    # IMPORTANT: isolate experiment DB per run so counts start from zero.
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_exp_dir = Path("experiments") / "runs"
    run_exp_dir.mkdir(parents=True, exist_ok=True)
    run_exp_path = run_exp_dir / f"main3_{run_id}.json"

    old_exp_path = os.environ.get("EXPERIMENT_DB_PATH")
    os.environ["EXPERIMENT_DB_PATH"] = str(run_exp_path)

    try:
        from auto_ml_research_agent.main import main as run_pipeline
        result = run_pipeline(problem, str(resolved), max_iterations_override=max_iterations_override)
    finally:
        if old_exp_path is None:
            os.environ.pop("EXPERIMENT_DB_PATH", None)
        else:
            os.environ["EXPERIMENT_DB_PATH"] = old_exp_path

    # Enrich result with per-run stats and research-style plots.
    run_history: List[Dict[str, Any]] = []
    if run_exp_path.exists():
        try:
            with open(run_exp_path, "r", encoding="utf-8") as f:
                run_history = json.load(f)
        except Exception:
            run_history = []

    total_logged = len(run_history)
    total_variants_tested = int(sum(int(h.get("n_variants", 0)) for h in run_history))
    result["run_experiment_db_path"] = str(run_exp_path)
    result["total_logged_experiments_current_run"] = total_logged
    result["total_variants_tested_current_run"] = total_variants_tested

    # Build improved run-only plots in a dedicated subfolder.
    report_dir = Path("models") / "reports" / "main3_run"
    report_dir.mkdir(parents=True, exist_ok=True)
    _generate_main3_plots(run_history, report_dir)
    result["main3_report_dir"] = str(report_dir)

    print(f"[main3] Current-run logged experiments: {total_logged}")
    print(f"[main3] Current-run total variants tested: {total_variants_tested}")
    print(f"[main3] Main3 run plots directory: {report_dir}")
    return result


def _generate_main3_plots(history: List[Dict[str, Any]], out_dir: Path) -> None:
    if not history:
        return

    iters = [int(h.get("iteration", i + 1)) for i, h in enumerate(history)]
    scores = [float(h.get("score", 0.0)) for h in history]
    metric_names = [str(h.get("metric_name", "")) for h in history]
    model_names = [h.get("config", {}).get("model_name", "unknown") for h in history]
    variants = [int(h.get("n_variants", 0)) for h in history]

    # If rmse-style metrics are stored negative for maximization, convert for readability.
    display_scores = []
    for s, m in zip(scores, metric_names):
        if m == "neg_rmse":
            display_scores.append(-s)
        else:
            display_scores.append(s)

    best_so_far = []
    cur_best = None
    for v in display_scores:
        cur_best = v if cur_best is None else max(cur_best, v)
        best_so_far.append(cur_best)

    def _ylim(vals: List[float]) -> tuple[float, float]:
        lo, hi = min(vals), max(vals)
        if hi - lo < 1e-6:
            eps = max(1e-3, abs(hi) * 0.005)
            return lo - eps, hi + eps
        pad = (hi - lo) * 0.1
        return lo - pad, hi + pad

    # Plot 1: Score trend + best-so-far.
    plt.figure(figsize=(9, 5))
    plt.plot(iters, display_scores, marker="o", linewidth=1.8, label="Score")
    plt.plot(iters, best_so_far, linestyle="--", linewidth=2.0, label="Best so far")
    plt.title("Run Score Trend (Current Run Only)")
    plt.xlabel("Iteration")
    plt.ylabel("Score")
    ymin, ymax = _ylim(display_scores + best_so_far)
    plt.ylim(ymin, ymax)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "run_score_trend.png", dpi=180)
    plt.close()

    # Plot 2: Variants tested per iteration.
    plt.figure(figsize=(9, 4.5))
    plt.bar(iters, variants, color="tab:blue")
    plt.title("Variants Tested per Iteration (Current Run)")
    plt.xlabel("Iteration")
    plt.ylabel("Variants")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_dir / "run_variants_per_iteration.png", dpi=180)
    plt.close()

    # Plot 3: Model-wise mean score with sample counts.
    model_to_scores: Dict[str, List[float]] = {}
    for m, s in zip(model_names, display_scores):
        model_to_scores.setdefault(m, []).append(s)
    labels = list(model_to_scores.keys())
    means = [sum(v) / len(v) for v in model_to_scores.values()]
    counts = [len(v) for v in model_to_scores.values()]
    x = range(len(labels))
    plt.figure(figsize=(10, 5))
    bars = plt.bar(x, means, color="tab:green")
    plt.xticks(x, labels, rotation=35, ha="right")
    plt.title("Model Performance by Mean Score (Current Run)")
    plt.xlabel("Model")
    plt.ylabel("Mean Score")
    ymin2, ymax2 = _ylim(means)
    plt.ylim(ymin2, ymax2)
    for b, c in zip(bars, counts):
        plt.text(b.get_x() + b.get_width() / 2, b.get_height(), f"n={c}", ha="center", va="bottom", fontsize=8)
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_dir / "run_model_mean_score.png", dpi=180)
    plt.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m auto_ml_research_agent.main3 \"problem\" [dataset.csv] [max_iterations]")
        raise SystemExit(1)
    problem_desc = sys.argv[1]
    dataset_file = sys.argv[2] if len(sys.argv) > 2 else None
    max_iterations_override = int(sys.argv[3]) if len(sys.argv) > 3 else None
    result = main(problem_desc, dataset_file, max_iterations_override=max_iterations_override)
    print(result)
