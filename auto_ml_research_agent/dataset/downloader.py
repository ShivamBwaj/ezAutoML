"""
Dataset Downloader: Downloads datasets from multiple sources with fallback hierarchy.
"""
from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd
import sklearn.datasets
from auto_ml_research_agent.exceptions import DatasetError


class DatasetDownloader:
    """
    Downloads datasets using multiple methods in order of priority:
    1. sklearn built-in datasets
    2. HuggingFace datasets
    3. Direct CSV URL
    4. Browser agent (handled separately in main)
    """

    def __init__(self, cache_dir: str = "data/raw"):
        """
        Initialize downloader.

        Args:
            cache_dir: Directory to save downloaded files
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def download(self, dataset_info: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """
        Download dataset using best available method.

        Args:
            dataset_info: Dataset metadata with source, name, url, etc.

        Returns:
            pandas DataFrame if successful, None otherwise
        """
        source = dataset_info.get("source")
        name = dataset_info.get("name", "")
        url = dataset_info.get("url")

        # Method 1: sklearn built-in datasets
        if source == "sklearn":
            return self._download_sklearn(dataset_info)

        # Method 2: HuggingFace datasets
        elif source == "huggingface":
            return self._download_huggingface(dataset_info)

        # Method 3: Direct CSV URL
        elif url and url.endswith(".csv"):
            return self._download_csv(url)

        # Method 4: OpenML
        elif source == "openml":
            return self._download_openml(dataset_info)

        # Method 5: Kaggle
        elif source == "kaggle":
            return self._download_kaggle(dataset_info)

        # None of the above - return None to signal fallback needed
        return None

    def _download_sklearn(self, dataset_info: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """Download from sklearn built-in datasets"""
        try:
            sklearn_name = dataset_info.get("sklearn_dataset_name")
            if not sklearn_name:
                # Extract from name (e.g., "sklearn_iris" -> "iris")
                sklearn_name = dataset_info.get("name", "").replace("sklearn_", "")

            # Try both load_* and fetch_* function patterns
            load_func = None
            for prefix in ["load", "fetch"]:
                func_name = f"{prefix}_{sklearn_name}"
                if hasattr(sklearn.datasets, func_name):
                    load_func = getattr(sklearn.datasets, func_name)
                    break

            if load_func is None:
                print(f"sklearn dataset function not found for {sklearn_name}")
                return None

            data = load_func()

            # Convert to DataFrame
            if hasattr(data, 'data'):
                X = pd.DataFrame(
                    data.data,
                    columns=data.feature_names if hasattr(data, 'feature_names') else None
                )
                y = pd.Series(data.target, name='target')
                df = pd.concat([X, y], axis=1)
                return df
            elif hasattr(data, 'frames'):
                # Special case: Olivetti faces returns dict with 'frames'
                df = pd.DataFrame(data.frames['data'])
                df['target'] = data.target
                return df
            else:
                return None

        except Exception as e:
            print(f"sklearn dataset load failed for {dataset_info.get('name')}: {e}")
            return None

    def _download_huggingface(self, dataset_info: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """Download from HuggingFace Hub"""
        try:
            from datasets import load_dataset
            ds_name = dataset_info["name"]

            # Try to load train split
            try:
                ds = load_dataset(ds_name, split="train")
            except:
                # Try without split
                ds = load_dataset(ds_name)
                if isinstance(ds, dict):
                    # Get first split
                    first_key = list(ds.keys())[0]
                    ds = ds[first_key]

            df = ds.to_pandas()
            return df

        except ImportError:
            print("HuggingFace datasets library not installed. Install: pip install datasets")
            return None
        except Exception as e:
            print(f"HuggingFace download failed for {dataset_info.get('name')}: {e}")
            return None

    def _download_csv(self, url: str) -> Optional[pd.DataFrame]:
        """Download CSV from direct URL"""
        try:
            df = pd.read_csv(url)
            return df
        except Exception as e:
            print(f"CSV download failed from {url}: {e}")
            return None

    def _download_openml(self, dataset_info: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """Download from OpenML by dataset ID"""
        try:
            import openml
        except ImportError:
            print("OpenML not installed. Install: pip install openml")
            return None

        try:
            openml_id = dataset_info.get("openml_id")
            if not openml_id:
                # Try to extract from name (e.g., "openml_123" -> 123)
                name = dataset_info.get("name", "")
                if name.startswith("openml_"):
                    try:
                        openml_id = int(name.replace("openml_", ""))
                    except:
                        return None
                else:
                    return None

            # Fetch dataset
            dataset = openml.datasets.get_dataset(openml_id)
            X, y, categorical, features = dataset.get_data(dataset_format='dataframe')

            # Combine into single DataFrame with target
            df = X.copy()
            if y is not None:
                target_name = dataset.default_target_attribute or 'target'
                df[target_name] = y

            return df

        except Exception as e:
            print(f"OpenML download failed for {dataset_info.get('name')}: {e}")
            return None

    def _download_kaggle(self, dataset_info: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """Download from Kaggle using Kaggle API"""
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
            import tempfile
            import zipfile
            from pathlib import Path as PathlibPath

            kaggle_ref = dataset_info.get("kaggle_ref")
            if not kaggle_ref:
                # Try to extract from name (e.g., "kaggle_author_dataset" -> "author/dataset")
                name = dataset_info.get("name", "")
                if name.startswith("kaggle_"):
                    kaggle_ref = name.replace("kaggle_", "", 1).replace("_", "/", 1)
                else:
                    kaggle_ref = None

            if not kaggle_ref:
                print(f"Kaggle dataset ref not found for {dataset_info.get('name')}")
                return None

            # Authenticate (uses ~/.kaggle/kaggle.json or env vars)
            api = KaggleApi()
            try:
                api.authenticate()
            except Exception as auth_e:
                print(f"Kaggle authentication failed: {auth_e}")
                return None

            # Download to temporary directory
            with tempfile.TemporaryDirectory() as tmpdir:
                print(f"Kaggle: Downloading {kaggle_ref} to {tmpdir}")
                # Download and unzip
                api.dataset_download_files(dataset=kaggle_ref, path=tmpdir, unzip=True)

                # Find CSV files (any .csv in the extracted directory)
                csv_files = list(Path(tmpdir).rglob("*.csv"))
                if not csv_files:
                    print(f"Kaggle: No CSV files found in {kaggle_ref}")
                    return None

                # Smart selection: Prefer files with 'train' in name, else 'data', else largest file
                # Also check for presence of common target columns to identify training data
                def score_csv(path):
                    name = path.name.lower()
                    score = 0
                    # Prefer training files
                    if 'train' in name:
                        score += 10
                    if 'training' in name:
                        score += 10
                    # Avoid test/submission files
                    if 'test' in name and not 'train' in name:
                        score -= 5
                    if 'submission' in name or 'sample' in name:
                        score -= 10
                    # Prefer larger files (likely training)
                    score += path.stat().st_size / 1000  # KB as proxy
                    return score

                # Sort by score descending
                sorted_csvs = sorted(csv_files, key=score_csv, reverse=True)

                # Print found CSVs for debugging
                print(f"Kaggle: Found {len(csv_files)} CSV files:")
                for csv in sorted_csvs[:5]:  # Show top 5
                    print(f"  - {csv.name} (size: {csv.stat().st_size:,} bytes, score: {score_csv(csv):.0f})")

                # Select the best candidate
                selected_csv = sorted_csvs[0]
                print(f"Kaggle: Selected CSV: {selected_csv.name}")
                df = pd.read_csv(selected_csv)
                return df

        except Exception as e:
            print(f"Kaggle download failed for {dataset_info.get('name')}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def save_to_cache(self, df: pd.DataFrame, name: str) -> str:
        """
        Save DataFrame to cache directory.

        Args:
            df: DataFrame to save
            name: Dataset name (used for filename)

        Returns:
            Path to saved file
        """
        cache_path = self.cache_dir / f"{name}.csv"
        df.to_csv(cache_path, index=False)
        return str(cache_path)
