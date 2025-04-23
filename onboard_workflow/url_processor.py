from config.config import firecrawl_config
import json

class URLProcessor:
    def __init__(self, urls):
        self.urls = urls
        self.app = firecrawl_config

    def get_crawled_results(self, url, limit):
        """
        Crawls websites using firecrawl APIs and returns markdown content with metadata
        """
        crawl_result = self.app.crawl_url(url, params={
            "limit": limit, "scrapeOptions": {"formats": ["markdown"]}
        })
        return crawl_result

    def get_scraped_data(self):
        """
        Inital call to generate data - can be async and can be used to tune in number of pages to scrape"""
        results = []
        
        for url in self.urls:
            result = self.get_crawled_results(url, 1)
            results.append(result)
        
        return results