#!/usr/bin/env python3
"""
Simple test script to verify the AUTO_ML_RESEARCH_AGENT system.
Uses sklearn iris dataset for quick validation.
"""
import sys
from pathlib import Path
import pandas as pd
from sklearn.datasets import load_iris

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def prepare_test_dataset():
    """Create iris.csv test dataset"""
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    csv_path = project_root / "iris_test.csv"
    df.to_csv(csv_path, index=False)
    print(f"[OK] Created test dataset: {csv_path} ({df.shape})")
    return str(csv_path)

def test_imports():
    """Test that all modules can be imported"""
    print("\n[TEST] Testing imports...")
    try:
        import auto_ml_research_agent.config
        import auto_ml_research_agent.llm.groq_client
        import auto_ml_research_agent.problem.interpreter
        import auto_ml_research_agent.data.profiler
        import auto_ml_research_agent.dataset.evaluator
        import auto_ml_research_agent.preprocessing.rules
        import auto_ml_research_agent.pipeline.generator
        import auto_ml_research_agent.training.trainer
        import auto_ml_research_agent.experiments.tracker
        print("[OK] All modules import successfully")
        return True
    except Exception as e:
        print(f"[FAIL] Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_config():
    """Test configuration loading (requires .env)"""
    print("\n[TEST] Testing configuration...")
    try:
        from auto_ml_research_agent.config import load_config, Config
        # Try to load - will fail without GROQ_API_KEY
        try:
            config = load_config()
            print("[OK] Configuration loaded (API key found)")
            return config
        except ValueError as e:
            if "GROQ_API_KEY" in str(e):
                print("[SKIP] GROQ_API_KEY not set - skipping full config test")
                print("       (Set it in .env to enable full pipeline)")
                return None
            raise
    except Exception as e:
        print(f"[FAIL] Config test failed: {e}")
        return None

def test_dataset_pipeline():
    """Test dataset loading and profiling"""
    print("\n[TEST] Testing dataset pipeline...")
    try:
        from auto_ml_research_agent.data.profiler import DataProfiler
        from auto_ml_research_agent.dataset.evaluator import DatasetEvaluator

        # Load test dataset
        csv_path = prepare_test_dataset()
        df = pd.read_csv(csv_path)
        print(f"  Loaded dataset: {df.shape}")

        # Profile
        profiler = DataProfiler()
        profile = profiler.profile(df)
        print(f"  Profiled: {profile['n_rows']} rows, {profile['n_cols']} columns")

        # Evaluate
        evaluator = DatasetEvaluator(
            target_column='target',
            task='classification',
            metric='accuracy'
        )
        eval_result = evaluator.evaluate(df)
        if eval_result['suitable']:
            print(f"  [OK] Dataset suitable (baseline: {eval_result['baseline_score']:.3f})")
        else:
            print(f"  [WARN] Dataset issues: {eval_result['reason']}")

        return True
    except Exception as e:
        print(f"[FAIL] Dataset pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_preprocessing():
    """Test preprocessing pipeline builder"""
    print("\n[TEST] Testing preprocessing...")
    try:
        from auto_ml_research_agent.preprocessing.rules import PreprocessingEngine

        # Use iris dataset
        iris = load_iris()
        df = pd.DataFrame(iris.data, columns=iris.feature_names)
        df['target'] = iris.target

        builder = PreprocessingEngine()
        preprocessor, metadata = builder.build_preprocessor(df, 'target')
        print(f"[OK] Preprocessor built with {len(preprocessor.transformers)} transformers")
        return True
    except Exception as e:
        print(f"[FAIL] Preprocessing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_pipeline_generation():
    """Test pipeline variant generation"""
    print("\n[TEST] Testing pipeline generation...")
    try:
        from auto_ml_research_agent.preprocessing.rules import PreprocessingEngine
        from auto_ml_research_agent.pipeline.generator import PipelineGenerator

        # Prepare data
        iris = load_iris()
        df = pd.DataFrame(iris.data, columns=iris.feature_names)
        df['target'] = iris.target

        builder = PreprocessingEngine()
        preprocessor, metadata = builder.build_preprocessor(df, 'target')

        generator = PipelineGenerator(task='classification', random_state=42)
        variants = generator.generate_variants(preprocessor, n_variants=5)

        print(f"[OK] Generated {len(variants)} variants")
        for v in variants[:3]:
            print(f"  - {v['name']}")
        if len(variants) > 3:
            print(f"  ... and {len(variants)-3} more")

        return True
    except Exception as e:
        print(f"[FAIL] Pipeline generation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_trainer():
    """Test training a single pipeline"""
    print("\n[TEST] Testing trainer...")
    try:
        from auto_ml_research_agent.preprocessing.rules import PreprocessingEngine
        from auto_ml_research_agent.pipeline.generator import PipelineGenerator
        from auto_ml_research_agent.training.trainer import Trainer
        from auto_ml_research_agent.training.evaluator import TrainingEvaluator

        # Prepare data
        iris = load_iris()
        df = pd.DataFrame(iris.data, columns=iris.feature_names)
        df['target'] = iris.target

        builder = PreprocessingEngine()
        preprocessor, metadata = builder.build_preprocessor(df, 'target')

        generator = PipelineGenerator(task='classification', random_state=42)
        variants = generator.generate_variants(preprocessor, n_variants=1)
        if not variants:
            print("[FAIL] No variants generated")
            return False

        pipeline = variants[0]['pipeline']

        trainer = Trainer(test_size=0.2, random_state=42, cv_threshold=500)
        result = trainer.train(df, 'target', pipeline, 'classification')

        evaluator = TrainingEvaluator()
        eval_result = evaluator.evaluate_variant(result)

        print(f"[OK] Trained {variants[0]['name']}")
        print(f"  Score: {eval_result['score']:.3f} ({eval_result['metric_name']})")
        print(f"  Method: {result['validation_method']}")

        return True
    except Exception as e:
        print(f"[FAIL] Trainer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_all_tests():
    """Run all tests and report results"""
    print("="*60)
    print("AUTO_ML_RESEARCH_AGENT - SYSTEM TESTS")
    print("="*60)

    results = []

    # Test 1: Imports
    results.append(("Imports", test_imports()))

    # Test 2: Config
    config_test = test_config()
    results.append(("Config", config_test is not None or "skipped"))

    # Test 3: Dataset pipeline
    results.append(("Dataset Pipeline", test_dataset_pipeline()))

    # Test 4: Preprocessing
    results.append(("Preprocessing", test_preprocessing()))

    # Test 5: Pipeline generation
    results.append(("Pipeline Generation", test_pipeline_generation()))

    # Test 6: Trainer
    results.append(("Trainer", test_trainer()))

    # Summary
    print("\n" + "="*60)
    print("TEST RESULTS")
    print("="*60)
    passed = sum(1 for _, r in results if r is True)
    total = len(results)
    for name, result in results:
        status = "PASS" if result is True else ("SKIP" if result == "skipped" else "FAIL")
        print(f"[{status}] {name}")

    print(f"\n{passed}/{total} tests passed")

    if passed == total:
        print("\nAll tests passed! System is ready.")
    else:
        print("\nSome tests failed. Review errors above.")

    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
