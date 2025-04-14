from onboard_workflow.url_processor import URLProcessor
from onboard_workflow.file_processor import FileProcessor
from onboard_workflow.clean_and_chunk import DataChunker
from config.config import (
    AIAgentOnboardRequest, AIAgentOnboardingDataResponse, metaData, zillizconfig
)
from typing import List
from utils.services import EmbeddingService
from collection_creator.create_zilliz_collection import ZillizClient
from fastapi import HTTPException
import asyncio

class GenerateDataSnapshot:
    def __init__(self, request: AIAgentOnboardRequest):
        if not request.urls:
            raise ValueError("At least one of urls, files, or text_input must be provided")

        self.request = request
        self.url_processor = URLProcessor(self.request.urls or [])
        self.file_processor = FileProcessor(self.request.files or [])
        self.data_chunker = DataChunker()

    async def assign_tasks(self):
        url_results = self.url_processor.get_scraped_data()
        file_task = asyncio.create_task(self.file_processor.process_files())
        print("Waiting for URL and file processing tasks to complete...")
        [file_results] = await asyncio.gather(
            file_task
        )
        
        responses = []

        #Process URL data
        for url_result in url_results:
            for data_item in url_result.get("data", []):
                responses.append(AIAgentOnboardingDataResponse(
                    meta_data=metaData(
                        session_id=self.request.session_id,
                        source="web",
                        url=data_item.get("metadata", {}).get("url", "")
                    ),
                    content=data_item.get("markdown", ""),
                    overview=""
                ))
        
        # Process file data
        for file_result, file_url in zip(file_results, self.request.files or []):
            for page in file_result.get("pages", []): 
                responses.append(AIAgentOnboardingDataResponse(
                    meta_data=metaData(
                        session_id=self.request.session_id,
                        source="KB",
                        url=file_url
                    ),
                    content=page.get("markdown", ""), 
                    overview="",
                ))

        return responses

    async def get_data(self):
        print('processing your URL scraping request')
        raw_data = self.assign_tasks()

        print("Received raw data --> Now processing clean")

        clean_data = await self.data_chunker.chunk_and_clean(raw_data)
        return clean_data

class DataUploader:
    def __init__(self):
        self.vector_db = ZillizClient()
        self.embedding_service = EmbeddingService()

    async def upload_data(self, scraped_data:List[AIAgentOnboardingDataResponse]):
        print("Received campaign data in upload function.")
        collection_records = {}
        for record in scraped_data:
            session_id = record.meta_data.session_id
            collection_records.setdefault(session_id, []).append(record)

        for session_id, records in collection_records.items():
            self.vector_db.create_collection(session_id)
            batch_size = zillizconfig.ZILLIZ_INSERTION_BATCH_SIZE

            for i in range(0, len(records), batch_size):
                print(f"Processing records {i} to {i + batch_size} for session {session_id}")
                batch = records[i:i + batch_size]
                texts = [record.content for record in batch]

                vectors = await self.embedding_service.get_embeddings(texts)
                print(f"Generated {len(vectors)} embeddings for session {session_id}")

                chunks_to_be_pushed = []
                for chunk, vector in zip(batch, vectors):
                    if len(vector) != zillizconfig.VECTOR_DIMENSION:
                        raise HTTPException(status_code=500, detail="Incorrect embedding dimension")
                    chunk.vector = vector
                    chunks_to_be_pushed.append(chunk)

                self.vector_db.insert_records(session_id, chunks_to_be_pushed)
        return {"status_code": 200, "message": "Data uploaded successfully!"}    
