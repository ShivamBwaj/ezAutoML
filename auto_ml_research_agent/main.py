#!/usr/bin/env python3
"""
AUTO_ML_RESEARCH_AGENT: Main entry point

End-to-end automated machine learning system:
1. Accepts natural language problem + optional dataset
2. Searches/downloads dataset if needed
3. Profiles data and builds preprocessor
4. Generates initial pipeline variants
5. Iteratively trains, evaluates, and improves via LLM analysis
6. Stops on patience limit or max iterations
7. Saves best model and launches FastAPI

Usage:
    python main.py "predict house prices" [dataset.csv]
    python main.py "classify iris flowers" iris.csv
    python main.py "classify iris flowers"  # auto-searches dataset
"""
import sys
import json
import logging
import warnings
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List
import pandas as pd
import joblib
from pydantic import BaseModel

# ==================== MONITORING SETUP ====================
# Create logs directory
log_dir = Path('logs')
log_dir.mkdir(exist_ok=True)
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_file = log_dir / f'pipeline_{timestamp}.log'

# Tee stdout and stderr to timestamped log file for debugging hangs
class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

# Open log file in append mode
_log_file = open(log_file, 'a', buffering=1)
sys.stdout = Tee(sys.stdout, _log_file)
sys.stderr = Tee(sys.stderr, _log_file)

print(f"\n{'='*60}")
print(f"PIPELINE STARTED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Log file: {log_file}")
print(f"{'='*60}\n")
# ==========================================================

# Configure logging - also write to timestamped file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='a'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Progress monitoring file
PROGRESS_FILE = Path('progress.monitor')

# Local imports
from auto_ml_research_agent.config import load_config
from auto_ml_research_agent.llm.groq_client import GroqClient
from auto_ml_research_agent.problem.interpreter import ProblemInterpreter, ProblemSpecification
from auto_ml_research_agent.dataset.search import DatasetSearcher
from auto_ml_research_agent.dataset.evaluator import DatasetEvaluator
from auto_ml_research_agent.dataset.downloader import DatasetDownloader
from auto_ml_research_agent.dataset.browser_agent import BrowserAgent
from auto_ml_research_agent.data.profiler import DataProfiler
from auto_ml_research_agent.preprocessing.rules import PreprocessingEngine
from auto_ml_research_agent.preprocessing.llm_edge import LLMEdgeDetector
from auto_ml_research_agent.pipeline.generator import PipelineGenerator
from auto_ml_research_agent.training.trainer import Trainer
from auto_ml_research_agent.training.evaluator import TrainingEvaluator
from auto_ml_research_agent.experiments.tracker import ExperimentTracker
from auto_ml_research_agent.reasoning.llm_analyzer import LLMAnalyzer
from auto_ml_research_agent.reasoning.variant_generator import VariantGenerator
from auto_ml_research_agent.controller.loop import ControllerLoop
from auto_ml_research_agent.registry.model_registry import ModelRegistry
from auto_ml_research_agent.exceptions import AutoMLError, DatasetError


class QueryExpansion(BaseModel):
    """Schema for LLM query expansion response"""
    queries: List[str]


def expand_queries_llm(problem: str, llm_client: GroqClient, n: int = 3) -> List[str]:
    """
    Use LLM to generate alternative search queries for the given problem.

    Args:
        problem: Original natural language problem
        llm_client: Initialized GroqClient
        n: Number of additional queries to generate

    Returns:
        List of additional query strings (may be fewer than n on failure)
    """
    prompt = (
        f"You are helping to find datasets for a machine learning problem.\n"
        f"Problem: {problem}\n\n"
        f"Generate {n} diverse search queries to find relevant datasets on platforms like "
        f"HuggingFace Hub or Kaggle. Consider synonyms, related concepts, and different phrasings.\n"
        f"Focus on keywords that would appear in dataset descriptions.\n"
        f"Return JSON: {{\"queries\": [\"query1\", \"query2\", ...]}}"
    )

    try:
        result = llm_client.generate_json(prompt, QueryExpansion)
        # Deduplicate from original problem and each other, keep order
        all_original = [problem.lower().strip()]
        additional = []
        for q in result.queries:
            q_clean = q.lower().strip()
            if q_clean not in all_original:
                all_original.append(q_clean)
                additional.append(q)
        return additional[:n]
    except Exception as e:
        # Fail gracefully - just don't expand
        logger.warning(f"LLM query expansion skipped: {str(e)[:80]}")
        return []


def log_progress(stage: str, message: str = "", iteration: int = None):
    """
    Log progress to both logger and a separate monitoring file.
    This file can be tailed to see real-time progress.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    iter_str = f"[Iter {iteration}] " if iteration is not None else ""
    log_entry = f"{timestamp} [{stage}] {iter_str}{message}"

    # Write to monitoring file (always append)
    with open(PROGRESS_FILE, 'a') as f:
        f.write(log_entry + "\n")
        f.flush()

    # Also log via logger
    if stage == "ERROR":
        logger.error(message)
    elif stage == "WARN":
        logger.warning(message)
    else:
        logger.info(message)


def main(problem: str, dataset_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Main orchestration function.

    Args:
        problem: Natural language problem description
        dataset_path: Optional path to CSV dataset

    Returns:
        Summary dictionary with results
    """
    logger.info("="*60)
    logger.info("AUTO_ML_RESEARCH_AGENT STARTED")
    logger.info("="*60)
    logger.info(f"Problem: {problem}")
    logger.info(f"Dataset path: {dataset_path if dataset_path else 'Auto-search'}")

    # Load configuration
    try:
        config = load_config()
        logger.info("[OK] Configuration loaded")

        # Suppress sklearn warnings if configured
        if config.suppress_sklearn_warnings:
            warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
            warnings.filterwarnings('ignore', message='.*number of unique classes.*')
            warnings.filterwarnings('ignore', message='.*least populated class.*')
            logger.info("[INFO] Sklearn warnings suppressed (suppress_sklearn_warnings=True)")
        else:
            logger.info("[INFO] Sklearn warnings enabled (suppress_sklearn_warnings=False)")
    except Exception as e:
        logger.error(f"[FAIL]  Configuration error: {e}")
        sys.exit(1)

    # Initialize components
    logger.info("\n[1/9] Initializing components...")
    llm_client = GroqClient(
        api_key=config.groq_api_key,
        model=config.groq_model,
        temperature=config.temperature,
        max_retries=config.max_retries
    )
    interpreter = ProblemInterpreter(llm_client)
    # Create browser agent first (may be used by searcher for web fallback)
    browser_agent = BrowserAgent(download_dir="data/raw", timeout=config.download_timeout)
    searcher = DatasetSearcher(config=config, browser_agent=browser_agent)
    downloader = DatasetDownloader(cache_dir="data/raw")
    profiler = DataProfiler()
    preprocessor_builder = PreprocessingEngine()
    edge_detector = LLMEdgeDetector(llm_client)
    trainer = Trainer(
        test_size=config.test_size,
        random_state=config.random_state,
        cv_threshold=config.cv_threshold
    )
    train_eval = TrainingEvaluator()
    tracker = ExperimentTracker(db_path=config.experiment_db_path)
    analyzer = LLMAnalyzer(llm_client)
    controller = ControllerLoop(patience=config.patience)
    registry = ModelRegistry(registry_dir=config.model_registry_dir)

    logger.info("[OK] All components initialized")

    # Step 1: Get dataset
    print("\n[2/9] Loading dataset...")
    df = None
    profiler_stats = None
    problem_spec = None
    validation = None

    if dataset_path:
        # Use provided path
        try:
            df = pd.read_csv(dataset_path)
            print(f"[OK] Loaded from {dataset_path}: {df.shape}")

            # Profile, interpret, validate
            profiler_stats = profiler.profile(df)
            problem_spec = interpreter.interpret(problem, profiler_stats)
            print(f"[OK] Problem specification: Task={problem_spec.task}, Target={problem_spec.target_column}, Metric={problem_spec.metric}")

            evaluator = DatasetEvaluator(
                target_column=problem_spec.target_column,
                task=problem_spec.task,
                metric=problem_spec.metric
            )
            validation = evaluator.evaluate(df)
            if not validation['suitable']:
                print(f"[FAIL]  Dataset unsuitable: {validation['reason']}")
                sys.exit(1)
            print(f"[OK] Dataset validated (baseline: {validation.get('baseline_score', 'N/A'):.4f})")
        except Exception as e:
            print(f"[FAIL]  Dataset processing failed: {e}")
            sys.exit(1)
    else:
        # Search for dataset
        print("  Searching for dataset...")
        queries = [problem]
        if config.enable_llm_query_expansion:
            print("  Expanding queries with LLM...")
            expanded = expand_queries_llm(problem, llm_client, n=3)
            queries.extend(expanded)
            print(f"    Expanded queries: {expanded}")
        candidates = searcher.search(
            queries,
            max_results=5,  # Reasonable number of candidates
            min_downloads=config.dataset_min_downloads
        )

        if not candidates:
            print("[FAIL]  No datasets found. Provide a dataset path with: python main.py 'problem' dataset.csv")
            sys.exit(1)

        # Try candidates in order until one succeeds
        max_attempts = min(config.max_dataset_attempts, len(candidates))

        for i in range(max_attempts):
            candidate = candidates[i]
            print(f"  Trying candidate {i+1}/{max_attempts}: {candidate['name']} (source: {candidate['source']})")

            # Try to download
            trial_df = downloader.download(candidate)
            if trial_df is None:
                print(f"    [WARN]  API download failed, trying browser agent if available...")
                downloaded_path = browser_agent.search_and_download(candidate['name'])
                if downloaded_path:
                    try:
                        trial_df = pd.read_csv(downloaded_path)
                        print(f"    [OK] Downloaded via browser: {trial_df.shape}")
                    except Exception as e:
                        print(f"    [FAIL]  Failed to read browser file: {e}")
                        trial_df = None
                else:
                    print(f"    [FAIL]  Browser download also failed")
                    trial_df = None

            if trial_df is None:
                print(f"    [WARN]  Rejecting {candidate['name']}: failed to load dataset")
                continue

            # Quick validation: check if dataset has minimum rows
            if trial_df.shape[0] < 100:
                print(f"    [WARN]  Rejecting {candidate['name']}: dataset too small ({trial_df.shape[0]} rows < 100 minimum)")
                continue

            # Process candidate: profile, interpret, full validation
            try:
                trial_profiler_stats = profiler.profile(trial_df)
                trial_problem_spec = interpreter.interpret(problem, trial_profiler_stats)

                evaluator = DatasetEvaluator(
                    target_column=trial_problem_spec.target_column,
                    task=trial_problem_spec.task,
                    metric=trial_problem_spec.metric
                )
                trial_validation = evaluator.evaluate(trial_df)

                if not trial_validation['suitable']:
                    print(f"    [WARN]  Rejecting {candidate['name']}: {trial_validation['reason']}")
                    continue

                # Success!
                df = trial_df
                profiler_stats = trial_profiler_stats
                problem_spec = trial_problem_spec
                validation = trial_validation
                selected_candidate = candidate
                source = candidate.get('source', 'unknown')
                url = candidate.get('url', '')
                url_msg = f" -> {url}" if url else ""
                print(f"  [OK] Selected dataset: {candidate['name']} (source: {source}{url_msg})")
                print(f"       Baseline score: {validation.get('baseline_score', 'N/A'):.4f}")
                break
            except Exception as e:
                print(f"    [WARN]  Rejecting {candidate['name']}: processing error: {str(e)[:100]}")
                continue

        if df is None:
            print(f"[FAIL]  All {max_attempts} candidates failed. Provide a dataset path with: python main.py 'problem' dataset.csv")
            sys.exit(1)

    # Step 4: LLM edge detection (optional)
    print("\n[5/9] Checking for data issues...")
    edge_analysis = edge_detector.detect(profiler_stats, problem_spec.model_dump())
    if edge_analysis.issues:
        print(f"[WARN]  Issues detected: {', '.join(edge_analysis.issues)}")
    if edge_analysis.suggestions:
        print(f"[INFO]  Suggestions: {', '.join(edge_analysis.suggestions[:3])}")

    # Step 5: Build preprocessor
    print("\n[6/9] Building preprocessor...")
    try:
        preprocessor, preprocessing_metadata = preprocessor_builder.build_preprocessor(df, problem_spec.target_column)
        print(f"[OK] Preprocessor built")

        # Display detailed preprocessing plan
        print("\n" + "="*60)
        print("PREPROCESSING PLAN")
        print("="*60)

        summary = preprocessing_metadata['summary']
        print(f"Input features: {summary['total_input_features']} "
              f"({summary['numeric_features']} numeric, {summary['categorical_features']} categorical)")

        print("\nColumn transformations:")
        for col, info in preprocessing_metadata['input_features'].items():
            col_type = "numeric" if col in df.select_dtypes(include=['number']).columns else "categorical"
            missing = info['missing_pct']
            unique = info['unique_count']
            print(f"  {col:30s} ({col_type:10s}) missing={missing:5.1f}% unique={unique:4d} "
                  f"sample={info['sample_values'][0] if info['sample_values'] else 'N/A'}")

        print("\nTransformers to be applied:")
        for transformer in preprocessing_metadata['transformers']:
            if transformer['name'] == 'numeric':
                cols = transformer['columns']
                print(f"  [NUMERIC]  Columns: {len(cols)}")
                for step in [s for s in transformer['steps'] if s]:
                    print(f"             - {step['name']}: {step.get('strategy', step.get('type', 'N/A'))}")
            elif transformer['name'].startswith('categorical_'):
                col = transformer['column']
                card = transformer['cardinality']
                enc = transformer.get('encoding_method', 'N/A')
                steps_str = " -> ".join([s['name'] for s in transformer['steps'] if s])
                print(f"  [CAT]      {col:20s} cardinality={card:4d} encoding={enc:10s} [{steps_str}]")

        print("="*60)

    except Exception as e:
        print(f"[FAIL]  Preprocessor build failed: {e}")
        sys.exit(1)

    # Initialize pipeline and variant generators
    pipeline_gen = PipelineGenerator(task=problem_spec.task, random_state=config.random_state)
    variant_gen = VariantGenerator(task=problem_spec.task)

    # ITERATION LOOP
    print("\n" + "="*60)
    print("STARTING ITERATION LOOP")
    print("="*60)

    iteration = 0
    best_pipeline = None
    best_config = None  # Config corresponding to best pipeline
    best_score = None
    last_variant_config = None
    latest_eval_results = None
    last_analysis = None  # Store last LLM analysis to reuse

    try:
        while True:
            iteration += 1
            print(f"\n--- Iteration {iteration} ---")

            # Generate variants
            if iteration == 1:
                print("  Generating initial variants...")
                variant_configs = pipeline_gen.generate_variants(preprocessor, n_variants=5)
            else:
                # Only call LLM every N iterations to reduce cost/time
                if iteration % config.llm_analysis_interval == 0:
                    print("  Generating variants from LLM suggestions...")
                    history = tracker.get_history()
                    analysis = analyzer.analyze(history, latest_eval_results)
                    print(f"  LLM issues: {analysis.issues[:2] if analysis.issues else 'None'}")
                    print(f"  LLM suggestions: {analysis.suggestions[:2]}")
                    # Store analysis for future iterations
                    last_analysis = analysis
                else:
                    print("  Generating variants from recent trends (skipping LLM)...")
                    # Use last analysis if available, otherwise fall back to default variants
                    analysis = last_analysis if 'last_analysis' in locals() else None

                base_config = {
                    'model_name': last_variant_config['model_name'],
                    'model_params': last_variant_config['params'],
                    'preprocessor': preprocessor
                }
                # Pass analysis.suggestions if available, else empty list (variant_gen will handle)
                suggestions = analysis.suggestions if analysis else []
                variant_configs = variant_gen.generate(base_config, suggestions, preprocessor, n_variants=5)
                # Keep track of last analysis for next iteration
                if analysis:
                    last_analysis = analysis

            if not variant_configs:
                print("[FAIL]  No variants generated!")
                break

            print(f"  Training {len(variant_configs)} variants...")

            # Train all variants
            variant_results = []
            for i, variant in enumerate(variant_configs):
                try:
                    result = trainer.train(df, problem_spec.target_column, variant['pipeline'], problem_spec.task)
                    eval_result = train_eval.evaluate_variant(result)
                    eval_result['config'] = variant
                    variant_results.append(eval_result)
                    score = eval_result['score']
                    metric = eval_result['metric_name']
                    print(f"    {i+1}. {variant['name']}: {metric}={score:.4f}")
                except Exception as e:
                    print(f"    {i+1}. {variant['name']}: FAILED ({type(e).__name__}: {e})")
                    import traceback
                    traceback.print_exc()
                    continue

            if not variant_results:
                print("[FAIL]  All variants failed to train!")
                break

            # Select best variant
            best_variant = max(variant_results, key=lambda x: x['score'])
            current_score = best_variant['score']
            current_pipeline = best_variant['config']['pipeline']
            last_variant_config = best_variant['config']

            # Display positive RMSE for regression
            display_score = -current_score if best_variant['metric_name'] == 'neg_rmse' else current_score
            print(f"  [OK] Best: {best_variant['config']['name']} ({best_variant['metric_name']}={display_score:.4f})")

            # Save model for this iteration
            model_path = Path("models") / f"iter_{iteration}.pkl"
            model_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(current_pipeline, model_path)

            # Log experiment (strip non-serializable pipeline)
            log_config = {k: v for k, v in best_variant['config'].items() if k != 'pipeline'}
            extra = {
                'n_variants': len(variant_results),
                'validation_method': best_variant.get('validation_method', 'unknown')
            }

            # Include preprocessing metadata on first iteration to document the preprocessing plan
            if iteration == 1 and 'preprocessing_metadata' in locals():
                extra['preprocessing'] = preprocessing_metadata

            tracker.log(
                iteration=iteration,
                config=log_config,
                score=current_score,
                metric_name=best_variant['metric_name'],
                model_path=str(model_path),
                extra=extra
            )

            # Update best
            if best_score is None or current_score > best_score:
                best_score = current_score
                best_pipeline = current_pipeline
                best_config = best_variant['config']  # Save config for best
                print(f"  [BEST]  New best overall score!")

            # Update latest eval for next LLM analysis
            latest_eval_results = {
                'best_score': current_score,
                'best_metrics': best_variant['all_metrics'],
                'iteration': iteration,
                'n_variants_trained': len(variant_results)
            }

            # Check if we should continue
            if not controller.should_continue(current_score):
                print(f"\n[STOP]  No improvement for {config.patience} iterations. Stopping.")
                break

            if iteration >= config.max_iterations:
                print(f"\n[STOP]  Reached max iterations ({config.max_iterations}). Stopping.")
                break

    except KeyboardInterrupt:
        print("\n\n[WARN]  Interrupted by user. Using current best model.")
    except Exception as e:
        print(f"\n[FAIL]  Unexpected error: {e}")
        import traceback
        traceback.print_exc()

    # Save best model to registry
    if best_pipeline is None:
        print("\n[FAIL]  No successful iterations. Exiting.")
        sys.exit(1)

    print("\n" + "="*60)
    print("SAVING BEST MODEL")
    print("="*60)

    try:
        # Strip pipeline from config for JSON serialization
        if best_config is None:
            print("[FAIL]  No best config available (no successful iterations)")
            sys.exit(1)
        final_config = {k: v for k, v in best_config.items() if k != 'pipeline'}

        # Pass preprocessing metadata to registry
        final_model_path = registry.save_best(
            pipeline=best_pipeline,
            score=best_score,
            metric=best_variant['metric_name'],
            config=final_config,
            preprocessing_metadata=preprocessing_metadata if 'preprocessing_metadata' in locals() else None,
            path_suffix="best_model"
        )
        print(f"[OK] Best model saved to: {final_model_path}")
    except Exception as e:
        print(f"[FAIL]  Failed to save model: {e}")
        sys.exit(1)

    # Final summary - convert neg_rmse back to positive for display
    display_score = best_score
    if best_variant['metric_name'] == 'neg_rmse':
        display_score = -best_score  # Convert back to positive RMSE

    summary = {
        'best_score': display_score,  # Store positive RMSE for user clarity
        'best_model_path': final_model_path,
        'total_iterations': iteration,
        'problem_spec': problem_spec.model_dump(),
        'total_experiments': len(tracker.get_history()),
        'patience': config.patience,
        'metric_name': best_variant['metric_name'],
        'preprocessing': preprocessing_metadata if 'preprocessing_metadata' in locals() else None
    }

    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)
    print(f"Best {best_variant['metric_name']}: {display_score:.4f}")
    print(f"Model saved: {final_model_path}")
    print(f"Total iterations: {iteration}")
    print(f"Total experiments logged: {len(tracker.get_history())}")

    # Show preprocessing summary again for clarity
    if 'preprocessing_metadata' in locals() and preprocessing_metadata:
        print("\nPreprocessing applied:")
        summary = preprocessing_metadata['summary']
        print(f"  Input: {summary['total_input_features']} features -> "
              f"{summary['numeric_features']} numeric, {summary['categorical_features']} categorical")
        print(f"  Transformers: {summary['transformers_created']}")

    print("\nTo deploy API: python -m auto_ml_research_agent.deployment.api")
    print("Or: uvicorn auto_ml_research_agent.deployment.api:app --reload")

    return summary


if __name__ == "__main__":
    # Parse command line arguments
    if len(sys.argv) < 2:
        print("="*60)
        print("AUTO_ML_RESEARCH_AGENT")
        print("="*60)
        print("\nUsage:")
        print("  python main.py 'natural language problem' [dataset.csv]")
        print("\nExamples:")
        print('  python main.py "classify iris flowers" iris.csv')
        print('  python main.py "predict house prices"')
        print('  python main.py "classify breast cancer"')
        print("\nIf no dataset is provided, the system will attempt to")
        print("auto-search for a suitable dataset from HuggingFace or sklearn.")
        print("="*60)
        sys.exit(1)

    problem_desc = sys.argv[1]
    dataset_file = sys.argv[2] if len(sys.argv) > 2 else None

    try:
        result = main(problem_desc, dataset_file)
        print("\n" + "="*60)
        print("FINAL RESULT")
        print("="*60)
        print(json.dumps(result, indent=2))
    except KeyboardInterrupt:
        print("\n\n[WARN]  Interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n[FAIL]  Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
