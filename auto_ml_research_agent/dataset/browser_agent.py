"""
Browser Agent: Safe Playwright automation for dataset download (fallback only).

WARNING: This is a LAST RESORT fallback. Prefer API-based sources.
This agent only performs hardcoded, safe actions - no LLM-controlled browsing.
"""
from pathlib import Path
from typing import Optional
import time
import argparse


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

    def __init__(self, download_dir: str = "data/raw", timeout: int = 30, auth_state_path: Optional[str] = None, headless: bool = True):
        """
        Initialize browser agent.

        Args:
            download_dir: Directory to save downloads
            timeout: Default timeout for operations (seconds)
        """
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.auth_state_path = Path(auth_state_path) if auth_state_path else None
        self.headless = headless

    def _new_context(self, browser, accept_downloads: bool = False):
        """Create Playwright context with optional persisted auth state."""
        if self.auth_state_path and self.auth_state_path.exists():
            print(f"[Browser] Using saved auth session: {self.auth_state_path}")
            return browser.new_context(
                accept_downloads=accept_downloads,
                storage_state=str(self.auth_state_path)
            )
        return browser.new_context(accept_downloads=accept_downloads)

    def _get_download_control(self, page, timeout_ms: int):
        """
        Poll for a Kaggle download control because SPA elements can render late
        on slower connections.
        """
        selectors = [
            self.SUPPORTED_SITES["kaggle"]["download_selector"],
            'button:has-text("Download")',
            'button:has-text("Download dataset")',
            'a:has-text("Download")',
            'a:has-text("Download dataset")',
        ]
        end_time = time.time() + (timeout_ms / 1000.0)
        while time.time() < end_time:
            for selector in selectors:
                try:
                    btn = page.query_selector(selector)
                    if btn:
                        return btn
                except Exception:
                    continue
            time.sleep(1.0)
        return None

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

        # If this looks like a Kaggle dataset ref (user/dataset), use direct page flow.
        if site == "kaggle" and "/" in dataset_name and " " not in dataset_name.strip():
            return self.download_kaggle_by_ref(dataset_name.strip())

        config = self.SUPPORTED_SITES[site]

        try:
            # Lazy import to avoid requirement if not used
            from playwright.sync_api import sync_playwright

            with sync_playwright() as p:
                browser = p.chromium.launch(headless=self.headless)
                context = self._new_context(browser, accept_downloads=True)
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
                    print(f"[Browser] Searching dataset page for query: {dataset_name}")
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

                    try:
                        with page.expect_download(timeout=30000) as dl_info:
                            download_btn.click()
                        download = dl_info.value
                    except Exception:
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

    def download_kaggle_by_ref(self, kaggle_ref: str) -> Optional[str]:
        """
        Download directly from a Kaggle dataset page using ref format 'user/dataset'.
        """
        result = self._download_kaggle_by_ref_once(kaggle_ref, headless=self.headless)
        if result is None and self.headless:
            print("[Browser] Headless flow failed; retrying full Kaggle download flow in headful mode...")
            result = self._download_kaggle_by_ref_once(kaggle_ref, headless=False)
        return result

    def _download_kaggle_by_ref_once(self, kaggle_ref: str, headless: bool) -> Optional[str]:
        """Single attempt at Kaggle direct download."""
        try:
            from playwright.sync_api import sync_playwright
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=headless)
                context = self._new_context(browser, accept_downloads=True)
                page = context.new_page()
                try:
                    timeout_ms = max(self.timeout * 1000, 90000)
                    url = f"https://www.kaggle.com/datasets/{kaggle_ref}"
                    print(f"[Browser] Opening Kaggle dataset page: {url} (headless={headless})")
                    page.goto(url, timeout=timeout_ms, wait_until="domcontentloaded")
                    print(f"[Browser] Landed URL: {page.url}")
                    print("[Browser] Waiting 5s before download trigger...")
                    time.sleep(5)

                    # Give Kaggle SPA extra time to hydrate on slower connections.
                    try:
                        page.wait_for_load_state("networkidle", timeout=timeout_ms)
                    except Exception:
                        pass

                    if "signin" in page.url.lower() or "login" in page.url.lower():
                        print("[Browser] Login required on Kaggle page; cannot download automatically.")
                        return None

                    download_btn = self._get_download_control(page, timeout_ms=timeout_ms)
                    if not download_btn:
                        print("[Browser] Download button not found on dataset page.")
                        # Helpful diagnostics for auth/paywall/UI changes
                        page_text = (page.content() or "").lower()
                        if "sign in" in page_text or "log in" in page_text:
                            print("[Browser] Page indicates authentication is required.")
                        if "notebook" in page_text and "download" not in page_text:
                            print("[Browser] Dataset page layout may have changed; no actionable download control found.")
                        return None

                    # If button looks disabled, fail fast instead of waiting for timeout.
                    try:
                        disabled_attr = download_btn.get_attribute("disabled")
                        aria_disabled = download_btn.get_attribute("aria-disabled")
                        if disabled_attr is not None or str(aria_disabled).lower() == "true":
                            print("[Browser] Download button is disabled (likely auth/permission gated).")
                            return None
                    except Exception:
                        pass

                    print(f"[Browser] Triggering download for {kaggle_ref}...")
                    download = None
                    # Strategy 1: direct click expecting download
                    try:
                        with page.expect_download(timeout=timeout_ms) as dl_info:
                            download_btn.click()
                        download = dl_info.value
                    except Exception:
                        pass

                    # Strategy 2: Kaggle often opens a modal with "Download dataset as zip"
                    if download is None:
                        try:
                            download_btn.click(force=True)
                            page.wait_for_timeout(1200)
                        except Exception:
                            pass
                        menu_btn = page.query_selector(
                            'button:has-text("Download dataset as zip"), a:has-text("Download dataset as zip"), '
                            'button:has-text("Download dataset"), a:has-text("Download dataset"), '
                            'button:has-text("Download"), a:has-text("Download")'
                        )
                        if menu_btn:
                            try:
                                with page.expect_download(timeout=timeout_ms) as dl_info:
                                    menu_btn.click(force=True)
                                download = dl_info.value
                            except Exception:
                                pass

                    if download is None:
                        raise RuntimeError("No browser download event fired after all download strategies")

                    suggested_name = download.suggested_filename or f"{kaggle_ref.replace('/', '_')}.zip"
                    save_path = self.download_dir / suggested_name
                    download.save_as(save_path)
                    print(f"[Browser] Downloaded file saved to: {save_path}")
                    return str(save_path)
                except Exception as e:
                    print(f"[Browser] Direct Kaggle download failed for {kaggle_ref}: {e}")
                    # Extra context for no-download-event failures.
                    try:
                        page_text = (page.content() or "").lower()
                        if "sign in" in page_text or "log in" in page_text:
                            print("[Browser] Kaggle appears to require login for this download in current session.")
                        elif "download" in page_text:
                            print("[Browser] Download controls exist but no browser download event fired.")
                        else:
                            print("[Browser] No clear download UI found after navigation/click attempt.")
                    except Exception:
                        pass
                    return None
                finally:
                    browser.close()
        except ImportError:
            print("Playwright not installed. Install: pip install playwright && playwright install chromium")
            return None
        except Exception as e:
            print(f"[Browser] Playwright runtime error: {e}")
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
                browser = p.chromium.launch(headless=self.headless)
                if self.auth_state_path and self.auth_state_path.exists():
                    context = browser.new_context(
                        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                        storage_state=str(self.auth_state_path)
                    )
                else:
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

    def init_kaggle_session(self) -> bool:
        """
        Open headful browser for user login and persist auth state.
        """
        if not self.auth_state_path:
            print("[Browser] auth_state_path is not configured.")
            return False
        self.auth_state_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            from playwright.sync_api import sync_playwright
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=False)
                context = browser.new_context()
                page = context.new_page()
                page.goto("https://www.kaggle.com/account/login", timeout=self.timeout * 1000)
                print("\n[Auth Init] Please complete Kaggle login in the opened browser window.")
                print("[Auth Init] After login finishes and kaggle.com is visible, press Enter here to save session.")
                input()
                context.storage_state(path=str(self.auth_state_path))
                print(f"[Auth Init] Saved authenticated state to: {self.auth_state_path}")
                browser.close()
                return True
        except Exception as e:
            print(f"[Auth Init] Failed to initialize Kaggle session: {e}")
            return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BrowserAgent utilities")
    parser.add_argument("--init-kaggle-session", action="store_true", help="Launch browser and save Kaggle auth session")
    parser.add_argument("--auth-state-path", default="playwright_auth/kaggle_state.json", help="Path to save auth state JSON")
    args = parser.parse_args()
    if args.init_kaggle_session:
        agent = BrowserAgent(auth_state_path=args.auth_state_path)
        ok = agent.init_kaggle_session()
        raise SystemExit(0 if ok else 1)
