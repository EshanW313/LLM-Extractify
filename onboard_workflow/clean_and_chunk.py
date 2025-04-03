from typing import List, Dict
from config.config import AIAgentOnboardingDataResponse, chunk_and_clean_task_app
import asyncio
import json
import tiktoken
from pydantic import BaseModel, ValidationError

class MiniChunk(BaseModel):
    content: str
    overview: str

class DataChunker:
    def __init__(self):
        self.llm_client = chunk_and_clean_task_app
        self.openai_client = self.llm_client.initialize_model_provider()
        self.tokenizer = tiktoken.get_encoding("cl100k_base")  # GPT-4o token encoding
        self.max_tokens_per_request = 10000
        self.semaphore = asyncio.Semaphore(3)  # Controls concurrent LLM calls

    def split_into_batches(self, content: str) -> List[str]:
        """
        Splits content into token-sized chunks, ensuring sentence boundaries are respected.
        """
        sentences = content.split(". ")  # Naive sentence boundary split
        batches = []
        current_batch = []
        current_token_count = 0

        for sentence in sentences:
            token_count = len(self.tokenizer.encode(sentence))
            if current_token_count + token_count > self.max_tokens_per_request:
                if current_batch:
                    batches.append(". ".join(current_batch))
                current_batch = [sentence]
                current_token_count = token_count
            else:
                current_batch.append(sentence)
                current_token_count += token_count

        if current_batch:
            batches.append(". ".join(current_batch))  # Add last batch
        
        return batches

    async def call_llm(self, content: str) -> List[Dict[str, str]]:
        """
        Calls LLM asynchronously to refine content into structured mini-chunks.
        Handles both single dictionary and list responses.
        """
        async with self.semaphore:
            messages = self.llm_client.get_messages_from_yaml(self.llm_client.messages)
            messages = self.llm_client.format_messages(messages, content=content)

            response = await self.llm_client.send_request(self.openai_client, messages)
            try:
                # Normalize JSON response
                message_content = response.replace("```json", "").replace("```", "").strip()
                print(f"LLM Response: {message_content}")

                parsed_data = json.loads(message_content)

                # Handle case where response is a single chunk (dict)
                if isinstance(parsed_data, dict) and "content" in parsed_data and "overview" in parsed_data:
                    parsed_data = {"content": [parsed_data]}  # Convert to list format

                # Handle case where response contains a list of chunks
                if isinstance(parsed_data, dict) and isinstance(parsed_data.get("content"), list):
                    valid_chunks = []
                    for chunk in parsed_data["content"]:
                        try:
                            validated_chunk = MiniChunk(**chunk)  # Validate structure
                            valid_chunks.append(validated_chunk.dict())
                        except ValidationError as e:
                            print(f"Skipping invalid chunk: {chunk}, Error: {e}")

                    return valid_chunks

                print(f"Unexpected LLM response format: {parsed_data}")
                return []  # Return empty list if response is not valid

            except json.JSONDecodeError:
                print("Error decoding JSON from LLM response")
                return []

    async def process_entry(self, raw_entry: AIAgentOnboardingDataResponse) -> List[AIAgentOnboardingDataResponse]:
        """
        Processes a single raw entry, splitting content into token-sized chunks and calling LLM asynchronously.
        """
        content = raw_entry.content
        # meta = raw_entry.get("meta_data", {})
        meta = raw_entry.meta_data
        
        content_batches = self.split_into_batches(content)
        tasks = [self.call_llm(batch) for batch in content_batches]
        mini_chunks_list = await asyncio.gather(*tasks)

        cleaned_data = []
        for mini_chunks in mini_chunks_list:
            for chunk in mini_chunks:
                cleaned_data.append(
                    AIAgentOnboardingDataResponse(
                        meta_data=meta,
                        content=chunk["content"],  
                        overview=chunk["overview"],  
                    )
                )

        return cleaned_data

    async def chunk_and_clean(self, raw_data: List[AIAgentOnboardingDataResponse]) -> List[AIAgentOnboardingDataResponse]:
        """
        Processes raw data asynchronously with controlled concurrency.
        """
        print('processing chunks')
        # tasks = [self.process_entry(entry) for entry in raw_data if "content" in entry]
        tasks = [self.process_entry(entry) for entry in raw_data if entry.content]
        results = await asyncio.gather(*tasks)
        return [item for sublist in results for item in sublist] 