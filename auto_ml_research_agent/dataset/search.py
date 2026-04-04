"""
Dataset Searcher: Searches for datasets across multiple sources.
"""
from typing import List, Dict, Any
import re
import pandas as pd


class DatasetSearcher:
    """
    Searches for datasets from multiple sources:
    - HuggingFace Hub
    - sklearn built-in datasets
    - OpenML (if installed)
    """

    # Mapping of classic ML problems to sklearn dataset names
    SKLEARN_DATASETS = {
        'iris': 'iris',
        'diabetes': 'diabetes',
        'digits': 'digits',
        'wine': 'wine',
        'breast': 'breast_cancer',
        'cancer': 'breast_cancer',
        'linnerud': 'linnerud',
        'faces': 'olivetti_faces',
        'california': 'california_housing',
        'housing': 'california_housing'
    }

    # Synonym mapping for query expansion: word -> [synonyms]
    # These help match datasets even when query doesn't use exact keyword
    SYNONYM_MAP = {
        # Housing/Real Estate (bidirectional)
        'house': ['housing', 'home', 'real estate', 'property', 'apartment', 'residential'],
        'housing': ['house', 'home', 'real estate', 'property'],
        'home': ['house', 'housing', 'residential'],
        'apartment': ['house', 'housing', 'residential'],
        # Crash/Accident
        'crash': ['accident', 'collision', 'safety', 'injury', 'damage'],
        'accident': ['crash', 'collision', 'safety'],
        'collision': ['accident', 'crash'],
        'safety': ['accident', 'crash', 'risk'],
        # Medical/Disease (bidirectional)
        'cancer': ['tumor', 'oncology', 'malignant', 'tumors'],
        'tumor': ['cancer', 'tumors', 'oncology', 'malignant'],
        'tumors': ['cancer', 'tumor', 'oncology'],
        # NOTE: 'disease' is NOT mapped to specific diseases to avoid false matches
        # e.g., "heart disease" should not match "diabetes" dataset
        # Users should use specific disease names
        # 'disease': ['illness', 'condition', 'disorder'],  # Commented out - too broad
        'illness': ['condition', 'disorder'],
        # Diabetes (specific)
        'diabetes': ['sugar', 'blood glucose', 'diabetic'],
        'diabetic': ['diabetes'],
        'diabetics': ['diabetes'],
        # Prediction/Forecasting
        'predict': ['forecast', 'estimate', 'regression', 'trend'],
        'forecast': ['predict', 'estimate', 'trend'],
        'estimate': ['predict', 'forecast'],
        # Classification
        'classify': ['classification', 'categorize', 'identify', 'detect'],
        'classification': ['classify', 'categorize'],
        # Price/Value
        'price': ['cost', 'value', 'expense', 'valuation', 'wages', 'income'],
        'cost': ['price', 'value'],
        'value': ['price', 'cost'],
        # Common dataset names (bidirectional)
        'wine': ['alcohol', 'beverage', 'fermented', 'wines'],
        'wines': ['wine', 'alcohol'],
        'iris': ['flower', 'plant', 'botany', 'flowers'],
        'flower': ['iris', 'flowers'],
        'flowers': ['iris', 'flower'],
        'plant': ['iris', 'botany'],
        'botany': ['iris', 'plant'],
        # Breast cancer
        'breast': ['breast_cancer', 'cancer', 'mammography'],
        'mammography': ['breast_cancer', 'cancer'],
        # Digits
        'digit': ['digits', 'mnist', 'handwritten'],
        'digits': ['digit', 'mnist', 'handwritten'],
        'handwritten': ['digits', 'mnist'],
        # California housing
        'california': ['housing', 'house', 'real estate'],
        'real estate': ['housing', 'property', 'price'],
        'property': ['housing', 'real estate'],

        # ========== FINANCE & ECONOMICS ==========
        'stock': ['market', 'equity', 'share', 'trading', 'investment', 'portfolio'],
        'stocks': ['stock', 'market', 'equities'],
        'market': ['stock', 'finance', 'trading', 'economy'],
        'finance': ['financial', 'banking', 'investment', 'stock'],
        'financial': ['finance', 'economic', 'budget'],
        'economy': ['economic', 'macro', 'market'],
        'investment': ['invest', 'portfolio', 'asset', 'stock'],
        'portfolio': ['investment', 'asset allocation', 'diversification'],
        'asset': ['asset', 'property', 'capital', 'investment'],
        'revenue': ['income', 'earnings', 'sales', 'profit'],
        'profit': ['revenue', 'earnings', 'gain', 'margin'],
        'loss': ['deficit', 'negative', 'decline'],
        'return': ['profit', 'yield', 'roi', 'gain'],
        'yield': ['return', 'interest', 'dividend'],
        'trading': ['trade', 'stock', 'market', 'buy', 'sell'],
        'risk': ['hazard', 'variance', 'volatility'],
        'volatility': ['variance', 'risk', 'standard deviation'],
        'credit': ['loan', 'debt', 'lending', 'borrow'],
        'loan': ['credit', 'debt', 'mortgage', 'finance'],
        'mortgage': ['home loan', 'housing finance', 'property'],
        'default': ['failure', 'bankruptcy', 'delinquency', 'credit risk'],
        'bankruptcy': ['insolvency', 'failure', 'default'],
        'bond': ['debt security', 'fixed income'],
        'fixed income': ['bond', 'treasury', 'yield'],
        'currency': ['money', 'forex', 'foreign exchange', 'dollar', 'euro'],
        'exchange rate': ['forex', 'currency conversion'],
        'inflation': ['price level', 'cpi', 'cost of living'],
        'interest rate': ['apr', 'loan rate', 'cost of borrowing'],
        'gdp': ['gross domestic product', 'economic output'],
        'stock price': ['share price', 'market cap', 'valuation'],
        'trading volume': ['volume', 'liquidity', 'activity'],

        # ========== NATURAL LANGUAGE PROCESSING (NLP) ==========
        'nlp': ['natural language processing', 'text mining', 'language model'],
        'natural language': ['nlp', 'text', 'language processing'],
        'text': ['document', 'string', 'corpus', 'body'],
        'language': ['linguistic', 'nlp', 'text'],
        'sentiment': ['opinion', 'emotion', 'mood', 'positive', 'negative'],
        'emotion': ['sentiment', 'feeling', 'affect'],
        'review': ['comment', 'feedback', 'evaluation', 'critique'],
        'spam': ['junk', 'unwanted', 'unsolicited'],
        'filter': ['detect', 'screen', 'remove'],
        'translation': ['translate', 'language conversion', 'mt'],
        'machine translation': ['translation', 'mt'],
        'summarization': ['summary', 'abstract', 'condense'],
        'summary': ['summarize', 'shorten', 'abstract'],
        'entity': ['named entity', 'ner', 'person', 'organization', 'location'],
        'ner': ['named entity recognition', 'entity extraction'],
        'pos': ['part of speech', 'tagging', 'syntactic'],
        'tagging': ['pos', 'label', 'annotate'],
        'parse': ['parsing', 'syntax', 'grammar'],
        'parsing': ['syntax analysis', 'grammar'],
        'chatbot': ['conversational agent', 'dialog system', 'assistant'],
        'dialogue': ['conversation', 'chat', 'dialog'],
        'conversational': ['chat', 'dialogue', 'interactive'],
        'intent': ['purpose', 'goal', 'objective'],
        'question': ['query', 'inquiry', 'ask'],
        'qa': ['question answering', 'q&a'],
        'question answering': ['qa', 'reading comprehension'],
        'language model': ['lm', 'llm', 'gpt', 'transformer'],
        'embedding': ['vector', 'representation', 'encoding'],
        'vector': ['embedding', 'numeric representation'],
        'word2vec': ['embedding', 'word representation'],
        'bert': ['transformer', 'language model', 'pretrained'],
        'transformer': ['attention', 'encoder-decoder', 'llm'],
        'pretrained': ['pre-trained', 'off-the-shelf', 'foundation model'],
        'corpus': ['dataset', 'collection', 'body of text'],
        'token': ['word', 'subword', 'piece'],
        'tokenization': ['tokenize', 'split', 'segment'],

        # ========== TIME SERIES ==========
        'time series': ['timeseries', 'temporal', 'sequential', 'chronological'],
        'timeseries': ['time series', 'temporal', 'sequence'],
        'temporal': ['time-based', 'time series', 'changing over time'],
        'sequential': ['time series', 'ordered', 'time-ordered'],
        'trend': ['pattern', 'direction', 'movement'],
        'seasonality': ['seasonal', 'periodic', 'cyclical'],
        'seasonal': ['periodic', 'cyclical', 'repeating'],
        'periodic': ['seasonal', 'regular intervals', 'cyclical'],
        'forecast': ['predict', 'project', 'extrapolate'],
        'arima': ['autoregressive', 'moving average', 'time series model'],
        'sarima': ['seasonal arima', 'time series'],
        'exponential smoothing': ['holt-winters', 'smoothing'],
        'prophet': ['facebook prophet', 'time series forecasting'],
        'lstm': ['long short-term memory', 'rnn', 'recurrent neural network'],
        'rnn': ['recurrent neural network', 'sequential'],
        'recurrent': ['rnn', 'lstm', 'gru'],
        'time step': ['timestamp', 'time index', 'datetime'],
        'datetime': ['time', 'date', 'timestamp'],
        'lag': ['delay', 'previous', 'past value'],
        'moving average': ['ma', 'rolling mean', 'smoothing'],
        'window': ['rolling', 'sliding', 'period'],
        'horizon': ['forecast horizon', 'prediction ahead'],
        'stationary': ['stationarity', 'constant mean', 'unit root'],
        'autocorrelation': ['serial correlation', 'lag correlation'],

        # ========== COMPUTER VISION ==========
        'image': ['picture', 'photo', 'visual', 'raster'],
        'picture': ['image', 'photo'],
        'photo': ['image', 'photograph'],
        'vision': ['computer vision', 'cv', 'visual perception'],
        'computer vision': ['cv', 'visual AI', 'image processing'],
        'object detection': ['detect objects', 'localization', 'bounding box'],
        'detection': ['detect', 'find', 'locate'],
        'segmentation': ['segment', 'partition', 'divide'],
        'semantic segmentation': ['pixel labeling', 'classify each pixel'],
        'instance segmentation': ['object segmentation', 'individual objects'],
        'classification': ['classify', 'categorize', 'identify'],
        'recognition': ['identify', 'recognize', 'detect'],
        'face': ['facial', 'countenance'],
        'facial': ['face', 'facial features'],
        'ocr': ['optical character recognition', 'text extraction'],
        'text extraction': ['ocr', 'read text from image'],
        'bounding box': ['bbox', 'rectangle', 'region'],
        'yolo': ['you only look once', 'real-time detection'],
        'faster r-cnn': ['faster rcnn', 'two-stage detector'],
        'mask r-cnn': ['mask rcnn', 'instance segmentation'],
        'pixel': ['dot', 'point', 'sample'],
        'feature extraction': ['extract features', 'descriptor', 'sift', 'surf'],
        'sift': ['scale-invariant feature transform', 'keypoint'],
        'surf': ['speeded-up robust features', 'keypoint'],
        'image processing': ['manipulate image', 'enhance', 'filter'],
        'filter': ['kernel', 'convolution', 'blur', 'sharpen'],
        'edge detection': ['canny', 'sobel', 'gradient'],
        'contour': ['outline', 'boundary', 'shape'],
        'histogram': ['distribution', 'frequency', 'bin'],
        'color space': ['rgb', 'hsv', 'bgr', 'grayscale'],
        'grayscale': ['black and white', 'monochrome'],
        'augmentation': ['data augmentation', 'transform', 'distort'],

        # ========== ANOMALY & OUTLIER DETECTION ==========
        'anomaly': ['outlier', 'abnormal', 'unusual', 'fraud'],
        'outlier': ['anomaly', 'rare', 'extreme'],
        'abnormal': ['anomalous', 'unusual', 'atypical'],
        'fraud': ['deception', 'cheat', 'illegal', 'criminal'],
        'novelty': ['new', 'unseen', 'unexpected'],
        'deviation': ['difference', 'variance', 'departure'],
        'isolation forest': ['isolationforest', 'outlier detection'],
        'one-class svm': ['oneclasssvm', 'oc-svm'],
        'local outlier factor': ['lof', 'density-based'],
    }

    def _expand_query(self, query: str) -> str:
        """
        Expand query with synonyms to improve matching.
        Example: 'predict house prices' -> 'predict house prices housing home real estate property'
        """
        # Tokenize: split on whitespace and punctuation, keep only word chars
        tokens = set(re.findall(r'\b\w+\b', query.lower()))
        synonyms = set()
        for token in tokens:
            if token in self.SYNONYM_MAP:
                synonyms.update(self.SYNONYM_MAP[token])
        # Return original + synonyms appended
        expanded = query + ' ' + ' '.join(sorted(synonyms))
        return expanded.strip()

    def _get_query_tokens(self, query: str) -> set:
        """Get token set from query including synonyms."""
        tokens = set(re.findall(r'\b\w+\b', query.lower()))
        for token in list(tokens):
            if token in self.SYNONYM_MAP:
                tokens.update(self.SYNONYM_MAP[token])
        return tokens

    def __init__(self, config=None, browser_agent=None):
        """
        Initialize searcher.

        Args:
            config: Optional Config object with search toggles (enable_kaggle_search, enable_huggingface_search)
            browser_agent: Optional BrowserAgent instance for web search fallback
        """
        self._config = config
        self._hf_api = None
        self._openml_available = None  # Lazy check
        self._kaggle_available = None  # Lazy check
        self._browser_agent = browser_agent  # For web search fallback

        # Determine which sources are enabled
        # If no config provided, default to True for backward compatibility
        if config is None:
            self.enable_kaggle = True
            self.enable_huggingface = True
        else:
            self.enable_kaggle = config.enable_kaggle_search
            self.enable_huggingface = config.enable_huggingface_search

    def _is_openml_available(self) -> bool:
        """Check if openml package is installed"""
        if self._openml_available is None:
            try:
                import openml
                print(f"  [OpenML] Package installed")
                self._openml_available = True
            except ImportError:
                print(f"  [OpenML] Package not installed (pip install openml)")
                self._openml_available = False
        return self._openml_available

    def _is_huggingface_available(self) -> bool:
        """Check if huggingface_hub package is installed"""
        if not hasattr(self, '_hf_available'):
            try:
                from huggingface_hub import HfApi
                print(f"  [HuggingFace] Package installed")
                self._hf_available = True
            except ImportError:
                print(f"  [HuggingFace] Package not installed (pip install huggingface-hub)")
                self._hf_available = False
        return self._hf_available

    def _is_kaggle_available(self) -> bool:
        """Check if kaggle package is installed and authenticated"""
        if self._kaggle_available is None:
            try:
                import kaggle
                # Quick check if authentication works
                from kaggle.api.kaggle_api_extended import KaggleApi
                api = KaggleApi()
                try:
                    # Try to read config - will raise if not authenticated
                    import os
                    kaggle_path = os.path.expanduser('~/.kaggle/kaggle.json')
                    if os.path.exists(kaggle_path):
                        print(f"  [Kaggle] Found credentials at: {kaggle_path}")
                        self._kaggle_available = True
                    elif os.getenv('KAGGLE_USERNAME') and os.getenv('KAGGLE_KEY'):
                        print(f"  [Kaggle] Using environment credentials")
                        self._kaggle_available = True
                    else:
                        print(f"  [Kaggle] No credentials found (expected at ~/.kaggle/kaggle.json or env vars)")
                        self._kaggle_available = False  # Installed but not authenticated
                except Exception as e:
                    print(f"  [Kaggle] Auth check failed: {e}")
                    self._kaggle_available = False
            except ImportError:
                print(f"  [Kaggle] Package not installed (pip install kaggle)")
                self._kaggle_available = False
        return self._kaggle_available

    def _get_hf_api(self):
        """Lazy import and initialize HuggingFace API"""
        if self._hf_api is None:
            try:
                from huggingface_hub import HfApi
                self._hf_api = HfApi()
            except ImportError as e:
                raise ImportError(
                    "huggingface-hub not installed. "
                    "Install with: pip install huggingface-hub"
                ) from e
        return self._hf_api

    def search(
        self,
        queries: List[str],
        max_results: int = 10,
        min_downloads: int = 10,
        min_rows: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Search for datasets across all sources.

        Args:
            queries: List of search query strings
            max_results: Maximum number of results to return
            min_downloads: Minimum download count for HuggingFace datasets
            min_rows: Minimum row count (soft filter - used later in evaluation)

        Returns:
            List of dataset candidates sorted by popularity (downloads + likes)
        """
        candidates = []
        source_counts = {'sklearn': 0, 'huggingface': 0, 'openml': 0, 'kaggle': 0}

        # Show which sources are available (and enabled)
        print(f"\n[Dataset Search] Query: '{queries[0]}' (expanded: {queries})")
        sources = ["sklearn"]
        if self.enable_huggingface and self._is_huggingface_available():
            sources.append("huggingface")
        if self._is_openml_available():
            sources.append("openml")
        if self.enable_kaggle and self._is_kaggle_available():
            sources.append("kaggle")
        print(f"  Active sources: {', '.join(sources)}")

        # Step 1: Check for sklearn built-in dataset matches (with synonym expansion)
        # NOTE: sklearn datasets get a moderate boost (1000) but not overwhelming
        # This allows high-quality Kaggle datasets with many votes to rank higher
        for query in queries:
            query_tokens = self._get_query_tokens(query)
            for keyword, ds_name in self.SKLEARN_DATASETS.items():
                if keyword in query_tokens:
                    candidates.append({
                        "name": f"sklearn_{ds_name}",
                        "description": f"Built-in sklearn dataset: {ds_name}",
                        "downloads": 1,  # Minimal boost - Kaggle datasets with any votes rank higher
                        "likes": 1,
                        "source": "sklearn",
                        "sklearn_dataset_name": ds_name,
                        "_is_sklearn_match": True  # Flag for debugging
                    })
                    source_counts['sklearn'] += 1

        # Step 2: Search Kaggle (if enabled)
        if self.enable_kaggle:
            kaggle_candidates = self._search_kaggle(queries, max_results=max_results)
            candidates.extend(kaggle_candidates)
            source_counts['kaggle'] += len(kaggle_candidates)

        # Step 3: Search HuggingFace Hub (if enabled)
        if self.enable_huggingface:
            try:
                hf_api = self._get_hf_api()
                print(f"  [HuggingFace] Searching for: {queries}")
                for query in queries:
                    expanded_query = self._expand_query(query)
                    try:
                        datasets = hf_api.list_datasets(search=expanded_query, limit=max_results)
                        for ds in datasets:
                            # Safely get description - may not exist in newer HF API versions
                            desc = ""
                            if hasattr(ds, 'description') and ds.description:
                                desc = str(ds.description)[:200]
                            elif hasattr(ds, 'card_data') and ds.card_data and hasattr(ds.card_data, 'dataset_description'):
                                desc = str(ds.card_data.dataset_description)[:200]
                            else:
                                desc = ds.id[:200] if hasattr(ds, 'id') else ""

                            candidates.append({
                                "name": ds.id,
                                "description": desc,
                                "downloads": ds.downloads or 0,
                                "likes": ds.likes or 0,
                                "source": "huggingface",
                                "url": None
                            })
                            source_counts['huggingface'] += 1
                    except Exception as e:
                        # Log but continue - HF failure shouldn't stop sklearn results
                        print(f"Warning: HuggingFace search failed for query '{query}': {e}")
                        continue
            except ImportError as e:
                print(f"  [HuggingFace] Not available: {e}")

        # Step 4: Search OpenML (optional) - DISABLED for now
        # openml_candidates = self._search_openml(queries, max_results=max_results)
        # candidates.extend(openml_candidates)
        # source_counts['openml'] += len(openml_candidates)
        pass

        # Filter out low-quality datasets (download threshold)
        # Note: sklearn datasets keep their high artificial score
        # Kaggle votes are typically low, so use lower threshold (5)
        filtered_candidates = []
        print(f"\n[Filter] Evaluating {len(candidates)} candidates (min_downloads={min_downloads})")
        kaggle_before = sum(1 for c in candidates if c.get("source") == "kaggle")
        print(f"[Filter] Kaggle candidates before filter: {kaggle_before}")
        for c in candidates:
            source = c.get("source", "")
            downloads = c.get("downloads", 0)
            likes = c.get("likes", 0)
            score = downloads + likes
            name = c.get("name", "")[:50]
            # Always allow sklearn (modest artificial score)
            if source == "sklearn":
                filtered_candidates.append(c)
                print(f"[Filter] KEEP sklearn: {name:50s} score={score:6d} (downloads={downloads}, likes={likes})")
            # Kaggle: votes are not downloads - accept any (even 0)
            elif source == "kaggle" and downloads >= 0:
                filtered_candidates.append(c)
                print(f"[Filter] KEEP kaggle ({downloads:4d} votes): {name:50s} score={score:6d}")
            # HuggingFace: use configured threshold
            elif source == "huggingface" and downloads >= min_downloads:
                filtered_candidates.append(c)
                print(f"[Filter] KEEP huggingface ({downloads:6d} downloads): {name:50s} score={score:6d}")
            # OpenML (if re-enabled): use threshold
            elif source == "openml" and downloads >= min_downloads:
                filtered_candidates.append(c)
            else:
                print(f"[Filter] REJECT {source}: {name:50s} score={score:6d}")
        print(f"[Filter] Kept {len(filtered_candidates)}/{len(candidates)} candidates")
        candidates = filtered_candidates

        # Deduplicate by name, keeping highest score
        seen = {}
        unique_candidates = []
        for c in candidates:
            name = c["name"]
            score = c["downloads"] + c["likes"]
            if name not in seen or score > seen[name]:
                seen[name] = score
                unique_candidates.append(c)

        # Sort by popularity
        unique_candidates.sort(key=lambda x: x["downloads"] + x["likes"], reverse=True)

        # Show source breakdown for final candidates
        final_counts = {'sklearn': 0, 'huggingface': 0, 'openml': 0, 'kaggle': 0}
        for c in unique_candidates[:max_results]:
            src = c.get('source', 'unknown')
            if src in final_counts:
                final_counts[src] += 1

        print(f"[Dataset Search] Found {len(unique_candidates)} total candidates")
        for source in ['sklearn', 'huggingface', 'openml', 'kaggle']:
            if final_counts[source] > 0:
                print(f"  {source}: {final_counts[source]} dataset(s)")

        # Print top candidates with scores for debugging
        print(f"\n[Dataset Search] Top {min(5, len(unique_candidates))} candidates:")
        for i, c in enumerate(unique_candidates[:5], 1):
            src = c.get('source', 'unknown')
            score = c.get('downloads', 0) + c.get('likes', 0)
            name = c.get('name', 'unknown')[:60]
            print(f"  {i}. {src:12s} score={score:6d}  {name}")
        print()  # Blank line

        return unique_candidates[:max_results]

    def get_sklearn_dataset_names(self) -> List[str]:
        """Get list of all available sklearn dataset names"""
        return list(self.SKLEARN_DATASETS.values())

    def _search_openml(self, queries: List[str], max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Search OpenML datasets by keyword matching on name/tags.

        Requires: pip install openml
        """
        print(f"  [OpenML] Searching for: {queries}")
        if not self._is_openml_available():
            print(f"  [OpenML] Not available (package not installed)")
            return []

        try:
            import openml
            candidates = []
            # Build a set of query tokens (with synonyms already expanded via queries list)
            query_tokens = set()
            for q in queries:
                query_tokens.update(self._get_query_tokens(q))

            # Fetch a sample of OpenML datasets (limit to avoid overwhelming)
            # Note: OpenML API can be slow; we fetch a limited set
            try:
                # Get dataset list (dictionary: id -> dict of metadata)
                dataset_list = openml.datasets.list_datasets(output_format='dataframe', size=1000)
            except Exception as e:
                print(f"Warning: OpenML list_datasets failed: {e}")
                return []

            # Filter by token match in name
            for _, row in dataset_list.iterrows():
                name = str(row.get('name', '')).lower()
                # Also check tags if available
                tags = str(row.get('tag', '')).lower() if 'tag' in row else ''
                # Check if any query token is in name or tags
                if any(token in name or token in tags for token in query_tokens):
                    # Estimate popularity - OpenML provides 'NumberOfInstances' and maybe downloads? not directly
                    # Use number of instances as proxy for size/popularity
                    n_instances = row.get('NumberOfInstances', 0)
                    # Convert to int safely
                    try:
                        n_instances = int(n_instances) if pd.notna(n_instances) else 0
                    except:
                        n_instances = 0

                    candidates.append({
                        "name": f"openml_{int(row['did'])}",
                        "description": str(row.get('description', '')[:200] if pd.notna(row.get('description')) else name[:200]),
                        "downloads": n_instances,  # Use instances as proxy for popularity
                        "likes": 0,
                        "source": "openml",
                        "openml_id": int(row['did']),
                        "url": f"https://www.openml.org/d/{int(row['did'])}"
                    })

            # Limit
            candidates.sort(key=lambda x: x['downloads'], reverse=True)
            return candidates[:max_results]

        except Exception as e:
            print(f"Warning: OpenML search failed: {e}")
            return []

    def _search_kaggle(self, queries: List[str], max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Search Kaggle datasets by keyword.

        Requires: pip install kaggle and authentication via ~/.kaggle/kaggle.json or env vars.
        """
        print(f"  [Kaggle] Searching for: {queries}")
        if not self._is_kaggle_available():
            print(f"  [Kaggle] Not available (package not installed or not authenticated)")
            return []

        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
            import os

            api = KaggleApi()
            try:
                api.authenticate()
                print(f"  [Kaggle] Authenticated successfully")
            except Exception as auth_e:
                print(f"  [Kaggle] Authentication failed: {auth_e}")
                return []

            candidates = []
            for query in queries:
                try:
                    print(f"  [Kaggle] Querying: '{query}' (max_results={max_results}, sort_by=votes)")
                    datasets = api.dataset_list(search=query, sort_by='votes', max_size=max_results)
                    dataset_list = list(datasets)
                    print(f"  [Kaggle] Found {len(dataset_list)} datasets for query '{query}'")
                    count_added = 0
                    for ds in dataset_list[:max_results]:
                        ref = ds.ref if hasattr(ds, 'ref') else ''
                        if not ref:
                            continue
                        name = f"kaggle_{ref.replace('/', '_')}"
                        description = ds.subtitle if hasattr(ds, 'subtitle') and ds.subtitle else ''
                        description = str(description)[:200] if description else ''
                        votes = 0
                        for attr in ['vote_count', 'totalVotes', 'votes', 'voteCount', 'upvotes']:
                            if hasattr(ds, attr):
                                val = getattr(ds, attr)
                                if val is not None and isinstance(val, (int, float)):
                                    votes = int(val)
                                    break
                        candidates.append({
                            "name": name,
                            "description": description,
                            "downloads": votes,
                            "likes": votes,
                            "source": "kaggle",
                            "url": f"https://www.kaggle.com/datasets/{ref}",
                            "kaggle_ref": ref
                        })
                        count_added += 1
                    print(f"  [Kaggle] Added {count_added} datasets from query '{query}'")
                except Exception as e:
                    print(f"  [Kaggle] Search failed for query '{query}': {type(e).__name__}: {e}")
                    continue

            # If API search returned no results, try web search fallback
            if len(candidates) == 0 and self._browser_agent is not None:
                print(f"  [Kaggle] API search returned 0 results, trying web search...")
                # Use the main query (not keywords) for web search - it's better
                main_query = queries[0] if queries else ""
                web_candidates = self._browser_agent.search_kaggle_web(main_query, max_results=max_results)
                candidates.extend(web_candidates)
                print(f"  [Kaggle] Web search added {len(web_candidates)} datasets")

            print(f"  [Kaggle] Total candidates before dedup: {len(candidates)}")
            return candidates

        except Exception as e:
            print(f"  [Kaggle] Search failed completely: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return []

            # Deduplicate by name
            seen = {}
            unique_candidates = []
            for c in candidates:
                name = c["name"]
                score = c["downloads"] + c["likes"]
                if name not in seen or score > seen[name]:
                    seen[name] = score
                    unique_candidates.append(c)

            # Sort by popularity
            unique_candidates.sort(key=lambda x: x['downloads'] + x['likes'], reverse=True)
            return unique_candidates[:max_results]

        except Exception as e:
            print(f"Warning: Kaggle search failed: {e}")
            return []
