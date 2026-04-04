"""
Browser Agent: Safe Playwright automation for dataset download (fallback only).

WARNING: This is a LAST RESORT fallback. Prefer API-based sources.
This agent only performs hardcoded, safe actions - no LLM-controlled browsing.
"""
from pathlib import Path
from typing import Optional
import time


class BrowserAgent:
    """
    Uses Playwright to download datasets from known websites.
    Only supports specific, hardcoded actions.
    """

    SUPPORTED_SITES = {
        'kaggle': {
            'base_url': 'https://www.kaggle.com',
            'search_path': '/datasets',
            'search_selector': 'input[placeholder*="Search"]',
            'result_selector': 'div[data-testid="dataset-card"] a:first-child',
            'download_selector': 'button[aria-label*="Download"]'
        }
    }

    # Selectors for search results page (different from dataset page)
    SEARCH_RESULT_SELECTORS = {
        'kaggle': {
            'results_container': 'div[data-testid="search-results-container"]',
            'dataset_cards': 'div[data-testid="dataset-card"]',
            'title_selector': 'div[data-testid="dataset-card"] a:first-child',
            'votes_selector': 'div[data-testid="dataset-card"] span[data-testid="vote-count"]',
            'subtitle_selector': 'div[data-testid="dataset-card"] p[data-testid="subtitle"]'
        }
    }

    def __init__(self, download_dir: str = "data/raw", timeout: int = 30):
        """
        Initialize browser agent.

        Args:
            download_dir: Directory to save downloads
            timeout: Default timeout for operations (seconds)
        """
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout

    def search_and_download(self, dataset_name: str, site: str = "kaggle") -> Optional[str]:
        """
        SAFE search and download from a supported site.

        Args:
            dataset_name: Name of dataset to search for
            site: Which site to use (currently only 'kaggle')

        Returns:
            Path to downloaded file if successful, None otherwise
        """
        if site not in self.SUPPORTED_SITES:
            print(f"Unsupported site: {site}. Supported: {list(self.SUPPORTED_SITES.keys())}")
            return None

        config = self.SUPPORTED_SITES[site]

        try:
            # Lazy import to avoid requirement if not used
            from playwright.sync_api import sync_playwright

            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                context = browser.new_context(accept_downloads=True)
                page = context.new_page()

                try:
                    # Navigate to site's dataset page
                    url = config['base_url'] + config['search_path']
                    page.goto(url, timeout=self.timeout * 1000)

                    # Wait for page load
                    page.wait_for_load_state("networkidle", timeout=10000)

                    # Check if we're on login page - can't proceed
                    if "signin" in page.url.lower() or "login" in page.url.lower():
                        print("Login required - cannot proceed without credentials")
                        return None

                    # Perform search
                    search_input = page.query_selector(config['search_selector'])
                    if not search_input:
                        print("Search input not found")
                        return None

                    search_input.fill(dataset_name)
                    search_input.press("Enter")

                    # Wait for results
                    page.wait_for_load_state("networkidle", timeout=10000)

                    # Click first result
                    first_result = page.query_selector(config['result_selector'])
                    if not first_result:
                        print("No search results found")
                        return None

                    first_result.click()
                    page.wait_for_load_state("networkidle", timeout=10000)

                    # Try to find and click download button
                    download_btn = page.query_selector(config['download_selector'])
                    if not download_btn:
                        print("Download button not found (may require account)")
                        return None

                    # Wait for download
                    try:
                        download = page.wait_for_event("download", timeout=30000)
                    except:
                        print("Download didn't start - may require interaction")
                        return None

                    # Save file
                    suggested_name = download.suggested_filename
                    if not suggested_name:
                        suggested_name = f"{dataset_name}_download.csv"

                    save_path = self.download_dir / suggested_name
                    download.save_as(save_path)

                    print(f"Downloaded to: {save_path}")
                    return str(save_path)

                except Exception as e:
                    print(f"Browser automation error: {e}")
                    return None
                finally:
                    browser.close()

        except ImportError:
            print("Playwright not installed. Install: pip install playwright && playwright install chromium")
            return None
        except Exception as e:
            print(f"Browser agent failed: {e}")
            return None

        return None

    def search_kaggle_web(self, query: str, max_results: int = 10) -> list:
        """
        Search Kaggle datasets via the website (not API) using Playwright.

        This is more reliable than the broken Kaggle API search.

        Args:
            query: Search query string
            max_results: Maximum number of datasets to return

        Returns:
            List of dataset candidate dictionaries in the same format as API search
        """
        try:
            from playwright.sync_api import sync_playwright, TimeoutError
        except ImportError:
            print("Playwright not installed. Install: pip install playwright && playwright install chromium")
            return []

        candidates = []
        print(f"  [Kaggle Web] Searching for: '{query}' (max_results={max_results})")

        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                context = browser.new_context(
                    user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                )
                page = context.new_page()
                page.set_default_timeout(self.timeout * 1000)

                # Navigate to Kaggle datasets search page
                search_url = f"https://www.kaggle.com/datasets?search={query}"
                page.goto(search_url)

                # Check if we're on login page
                if "signin" in page.url.lower() or "login" in page.url.lower():
                    print("  [Kaggle Web] Login required - skipping web search")
                    browser.close()
                    return []

                # Wait for results - look for dataset links
                try:
                    page.wait_for_selector('a[href*="/datasets/"]', timeout=15000)
                except TimeoutError:
                    print("  [Kaggle Web] No results found (timeout)")
                    browser.close()
                    return []

                # Extract dataset links directly
                links = page.query_selector_all('a[href*="/datasets/"]')
                print(f"  [Kaggle Web] Found {len(links)} dataset links")

                seen_refs = set()
                for link in links[:max_results]:
                    try:
                        href = link.get_attribute('href')
                        if not href or '/datasets/' not in href:
                            continue

                        # Extract dataset ref from URL
                        # Expected format: /datasets/username/dataset-name
                        parts = href.strip('/').split('/')
                        if len(parts) < 3 or parts[0] != 'datasets':
                            continue
                        dataset_ref = f"{parts[1]}/{parts[2]}"
                        if dataset_ref in seen_refs:
                            continue
                        seen_refs.add(dataset_ref)

                        name = f"kaggle_{parts[1]}_{parts[2]}"
                        description = link.text_content().strip()[:200]

                        # Try to find vote count nearby in DOM (might be in parent or sibling)
                        votes = 0
                        try:
                            # Look for vote count in the card's parent or following elements
                            parent = link.query_selector('xpath=..')
                            if parent:
                                vote_elem = parent.query_selector('span[data-testid="vote-count"], .vote-count, [aria-label*="vote"]')
                                if vote_elem:
                                    vote_text = vote_elem.text_content().strip()
                                    if 'K' in vote_text:
                                        votes = int(float(vote_text.replace('K', '')) * 1000)
                                    else:
                                        votes = int(vote_text.replace(',', ''))
                        except:
                            pass  # Votes not critical

                        candidates.append({
                            "name": name,
                            "description": description,
                            "downloads": votes,
                            "likes": votes,
                            "source": "kaggle",
                            "url": f"https://www.kaggle.com/datasets/{dataset_ref}",
                            "kaggle_ref": dataset_ref
                        })
                    except Exception as e:
                        continue  # Skip individual errors

                browser.close()
                print(f"  [Kaggle Web] Extracted {len(candidates)} datasets")
                return candidates

        except Exception as e:
            print(f"  [Kaggle Web] Search failed: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return []
