import asyncio
from config.config import firecrawl_config
import json

class URLProcessor:
    def __init__(self, urls):
        self.urls = urls
        self.app = firecrawl_config

    def get_crawled_results(self, url, limit):
        crawl_result = self.app.crawl_url(url, 
                                          params={"limit": limit, "scrapeOptions": {"formats": ["markdown"]}})
        return crawl_result

    def get_scraped_data(self):
        results = []
        
        for url in self.urls:
            result = self.get_crawled_results(url, 1)
            results.append(result)
        with open("data/scraped_data_firecrawl_project_test.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        return results