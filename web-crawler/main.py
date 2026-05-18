import sys
import requests
import asyncio
from async_crawler import AsyncCrawler
from json_report import write_json_report

async def crawl_site_async(base_url: str, max_concurrency: int = 5, max_page: int = 3) -> dict:
    async with AsyncCrawler(base_url, max_concurrency=max_concurrency, max_page=max_page) as crawler:
        return await crawler.crawl()
    
# keep synchronous helper for compatibility
def sync_get_html(url: str) -> str:
    response = requests.get(url, headers={"User-Agent": "BootCrawler/1.0"})
    if response.status_code != 200:
        print(f"Failed to retrieve {url}")
        return ""
    if response.headers.get("Content-Type", "").split(";")[0] != "text/html":
        print(f"Non-HTML content at {url}")
        return ""
    return response.text

async def main_async():
    if len(sys.argv) < 4:
        print("no website provided")
        sys.exit(1)
    elif len(sys.argv) > 4:
        print("too many arguments provided")
        sys.exit(1)

    base_url = sys.argv[1]
    max_concurrency = int(sys.argv[2])
    max_page = int(sys.argv[3])

    print(f"Async crawl starting...{base_url} - {max_concurrency} concurrency, max {max_page} pages")
    page_data = await crawl_site_async(base_url, max_concurrency,  max_page)

   
    write_json_report(page_data)

if __name__ == "__main__":
    asyncio.run(main_async())
