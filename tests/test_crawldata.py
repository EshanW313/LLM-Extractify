from onboard_workflow.url_processor import URLProcessor
import json
import asyncio

async def test_url_processor():
    """
    Test firecrawl API and URL Processor class
    """
    urls = ['https://coldbean.ai/', 'https://www.crunchbase.com/']

    url_processor = URLProcessor(urls)

    data = await url_processor.get_scraped_data()
    # print(data)
    with open("data/scraped_data_firecrawl.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print("Data saved to scraped_data.json")

if __name__ == "__main__":
    asyncio.run(test_url_processor())