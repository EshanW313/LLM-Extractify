import asyncio
from config.config import firecrawl_config

class URLProcessor:
  def __init__(self, urls):
    self.urls = urls
    self.app = firecrawl_config

  async def _crawl_website(self, url, limit):
        # Start the crawl with async_crawl_url using run_in_executor
        loop = asyncio.get_running_loop()
        crawl_status = await loop.run_in_executor(
            None,  # Use the default executor (ThreadPoolExecutor)
            lambda: self.app.async_crawl_url(
                url, params={"limit": limit, "scrapeOptions": {"formats": ["markdown"]}}
            ),
        )

        crawl_id = crawl_status["id"]

        # Poll the status until the crawl completes
        while True:
            # Use run_in_executor again for check_crawl_status
            crawl_status = await loop.run_in_executor(
                None, lambda: self.app.check_crawl_status(crawl_id)
            )

            # Check for completion
            if (
                crawl_status.get("success")
                and crawl_status.get("status") == "completed"
            ):
                data = {"url": url, "data": crawl_status["data"]}
                return data

            await asyncio.sleep(1)

  async def get_scraped_data(self):
        semaphore = asyncio.Semaphore(3)  # Limit concurrent requests to 3

        async def limited_crawl(entry):
            async with semaphore:
                return await self._crawl_website(
                    entry, 2
                )

        results = []
        for entry in self.urls:
            result = await limited_crawl(entry)
            results.append(result)
            await asyncio.sleep(30)  # Enforce a 3 URLs per minute limit
        return results