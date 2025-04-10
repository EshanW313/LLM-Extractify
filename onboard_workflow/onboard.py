from onboard_workflow.url_processor import URLProcessor
from onboard_workflow.clean_and_chunk import DataChunker
from config.config import (
    AIAgentOnboardRequest, AIAgentOnboardingDataResponse, metaData
)

class GenerateDataSnapshot:
    def __init__(self, request: AIAgentOnboardRequest):
        if not request.urls:
            raise ValueError("At least one of urls, files, or text_input must be provided")

        self.request = request
        self.url_processor = URLProcessor(self.request.urls or [])
        self.data_chunker = DataChunker()

    def assign_tasks(self):
        url_results = self.url_processor.get_scraped_data()
        
        responses = []

        #Process URL data
        for url_result in url_results:
            for data_item in url_result.get("data", []):
                responses.append(AIAgentOnboardingDataResponse(
                    meta_data=metaData(
                        source="web",
                        url=data_item.get("metadata", {}).get("url", "")
                    ),
                    content=data_item.get("markdown", ""),
                    overview=""
                ))
        return responses

    async def get_data(self):
        print('processing your URL scraping request')
        raw_data = self.assign_tasks()

        print("Received raw data --> Now processing clean")

        clean_data = await self.data_chunker.chunk_and_clean(raw_data)
        return clean_data

