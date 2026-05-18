import aiohttp
import asyncio
from crawl import domain_url, extract_page_data, normalize_url

class AsyncCrawler:
    def __init__(self, base_url: str, max_concurrency: int = 5, max_page:int = 3):
        self.base_url = base_url
        self.base_domain = domain_url(base_url)
        self.page_data = {}
        self.lock = asyncio.Lock()
        self.max_concurrency = max_concurrency
        self.max_page = max_page
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.session = None
        self.should_stop = False
        self.all_tasks = set()
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.session.close()

    async def add_page_visit(self, normalized_url: str) -> bool:
        """Return True if this is the first visit for normalized_url and reserve it."""
        async with self.lock:
            if normalized_url in self.page_data:
                return False
            if len(self.page_data) >= self.max_page:
                self.should_stop = True
                print("Reached maximum number of pages to crawl.")
                return False
            # reserve spot to avoid duplicate visits
            self.page_data[normalized_url] = None
            return True

    async def get_html(self, url: str) -> str:
        try:
            async with self.session.get(url, headers={"User-Agent": "BootCrawler/1.0"}) as response:
                if response.status != 200:
                    print(f"Failed to retrieve {url}")
                    return ""
                if response.headers.get("Content-Type", "").split(";")[0] != "text/html":
                    print(f"Non-HTML content at {url}")
                    return ""
                return await response.text()
        except Exception as e:
            print(f"Error fetching {url}: {e}")
            return ""

    async def crawl_page(self, current_url: str):
        # Only crawl same domain
        if domain_url(current_url) != self.base_domain or self.should_stop:
            return
        
        norm = normalize_url(current_url)
        is_new = await self.add_page_visit(norm)
        if not is_new:
            return

        async with self.semaphore:
            page_html = await self.get_html(current_url)
            print(f"Fetched: {current_url}")

            if not page_html:
                async with self.lock:
                    self.page_data[norm] = {"url": current_url, "page_urls": [], "error": "failed to retrieve"}
                return

            page = extract_page_data(page_html, current_url)
            async with self.lock:
                self.page_data[norm] = page

            # Recursively crawl child URLs (limited for demo)
            child_urls = page.get("page_urls", [])
            tasks = []
            for url in child_urls:
                if domain_url(url) != self.base_domain:
                    continue
                child_norm = normalize_url(url)
                async with self.lock:
                    if child_norm in self.page_data and self.page_data[child_norm] is not None:
                        continue
                # create task, track it in all_tasks and ensure removal when done
                task = asyncio.create_task(self.crawl_page(url))
                self.all_tasks.add(task)
                # attach a callback to remove from the set when finished
                task.add_done_callback(lambda t: self.all_tasks.discard(t))
                tasks.append(task)

            if tasks:
                try:
                    await asyncio.gather(*tasks)
                finally:
                    # make sure any remaining references are removed
                    for t in tasks:
                        self.all_tasks.discard(t)

    async def crawl(self) -> dict:
        await self.crawl_page(self.base_url)
        return self.page_data