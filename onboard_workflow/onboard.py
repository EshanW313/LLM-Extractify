from onboard_workflow.url_processor import URLProcessor
from onboard_workflow.file_processor import FileProcessor
from onboard_workflow.clean_and_chunk import DataChunker
from config.config import (
    AIAgentOnboardRequest, AIAgentOnboardingDataResponse, metaData, zillizconfig,
    chunk_and_clean_task_app, gemma_chunk_and_clean_task_app
)

from typing import List
from utils.services import EmbeddingService
from collection_creator.create_zilliz_collection import ZillizClient
from fastapi import HTTPException
import asyncio
import logging

logger = logging.getLogger(__name__)

class GenerateDataSnapshot:
    def __init__(self, request: AIAgentOnboardRequest, llm_choice='openai'):
        if not request.urls and not request.files:
            raise ValueError("At least one of urls, files, or text_input must be provided")

        self.request = request
        self.url_processor = URLProcessor(self.request.urls or [])
        self.file_processor = FileProcessor(self.request.files or [])
        
        if llm_choice.lower() == "gemma":
            selected_llm_config = gemma_chunk_and_clean_task_app
        elif llm_choice.lower() == "openai":
            selected_llm_config = chunk_and_clean_task_app
        else:
            logging.warning(f"Warning: Unknown LLM choice '{llm_choice}'. Defaulting to OpenAI")
            selected_llm_config = chunk_and_clean_task_app
        
        self.data_chunker = DataChunker(llm_config=selected_llm_config)
        logger.info(f"GenerateDataSnapshot initialized for session {self.request.session_id} with LLM: {llm_choice}")

    async def assign_tasks(self):
        """
        Asyncronously runs tasks for different processes - assigns urls to url processor, files to file processing
        retreives and returns output as AIAgentOnboardingDataResponse object for adding to collection curator
        """
        url_results = self.url_processor.get_scraped_data()
        file_task = asyncio.create_task(self.file_processor.process_files())
        logging.info("Waiting for URL and file processing tasks to complete...")
        [file_results] = await asyncio.gather(
            file_task
        )
        
        responses = []

        # Process URL data
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
        """
        This is the initally called method that initialised and assigns responses"""
        raw_data = await self.assign_tasks()

        logging.info("Received raw data --> Now processing clean")

        clean_data = await self.data_chunker.chunk_and_clean(raw_data)
        return clean_data

class DataUploader:
    def __init__(self):
        self.vector_db = ZillizClient()
        self.embedding_service = EmbeddingService()

    async def upload_data(self, scraped_data:List[AIAgentOnboardingDataResponse]):
        """
        Uploads data to collection curator
        """
        logging.info("Received campaign data in upload function.")
        collection_records = {}
        for record in scraped_data:
            session_id = record.meta_data.session_id
            collection_records.setdefault(session_id, []).append(record)
        logging.debug(collection_records)

        for session_id, records in collection_records.items():
            self.vector_db.create_collection(session_id)
            batch_size = zillizconfig.ZILLIZ_INSERTION_BATCH_SIZE

            for i in range(0, len(records), batch_size):
                logging.info(f"Processing records {i} to {i + batch_size} for session {session_id}")
                batch = records[i:i + batch_size]
                texts = [record.content for record in batch]

                vectors = await self.embedding_service.get_embeddings(texts)
                vectors_openai = await self.embedding_service.get_openaiembeddings(texts)
                logging.info(f"Generated {len(vectors)} embeddings for session {session_id}")

                chunks_to_be_pushed = []
                for chunk, vector, vector_openai in zip(batch, vectors, vectors_openai):
                    if len(vector) != zillizconfig.VECTOR_DIMENSION or len(vector_openai) != zillizconfig.VECTOR_DIMENSION :
                        raise HTTPException(status_code=500, detail="Incorrect embedding dimension")
                    chunk.vector = vector
                    chunk.vector_openai = vector_openai
                    chunks_to_be_pushed.append(chunk)

                self.vector_db.insert_records(session_id, chunks_to_be_pushed)
        return {"status_code": 200, "message": "Data uploaded successfully!"}    
