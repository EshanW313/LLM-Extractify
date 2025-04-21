from typing import List, Dict, Union
from config.config import AIAgentOnboardingDataResponse, chunk_and_clean_task_app
from config.model_provider_config import ModelProviderConfig
import asyncio
import json
import tiktoken
from pydantic import BaseModel, ValidationError
import logging


class MiniChunk(BaseModel):
	content: str
	overview: str

class DataChunker:
	def __init__(self, llm_config: ModelProviderConfig):
		self.llm_client_config = llm_config
		self.llm_provider_client = self.llm_client_config.initialize_model_provider()
		if not self.llm_provider_client:
			raise ValueError("Failed to initialize LLM provider:", self.llm_client_config.api)

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
	
	def _parse_chunks(self, parsed_data: str) -> List[Dict[str, str]]:
		"""
		Normalize and parse LLM response into a list of validated MiniChunks.
		Supports various response shapes like:
		- single dict with content & overview
		- dict with keys like 'data', 'response', 'content', or 'result'
		- raw list of dicts
		"""
		valid_chunks = []
		# Normalize single dict with content and overview
		if isinstance(parsed_data, dict) and "content" in parsed_data and "overview" in parsed_data:
			parsed_data = [parsed_data]

		# Extract list from common response keys
		elif isinstance(parsed_data, dict):
			for key in ["data", "response", "content", "result"]:
				if isinstance(parsed_data.get(key), list):
					parsed_data = parsed_data[key]
					break

		# At this point, parsed_data should be a list of dicts
		if isinstance(parsed_data, list):
			for chunk in parsed_data:
				try:
					validated_chunk = MiniChunk(**chunk)
					valid_chunks.append(validated_chunk.dict())
				except ValidationError as e:
					logging.warning(f"Skipping invalid chunk: {chunk}, Error: {e}")

		return valid_chunks
	
	async def call_llm(self, content: str) -> List[Dict[str, str]]:
		"""
		Calls LLM asynchronously to refine content into structured mini-chunks.
		Delegates parsing logic to `parse_chunks`.
		"""
		async with self.semaphore:
			msg_structure = self.llm_client_config.get_messages_from_yaml(self.llm_client_config.messages)
			msg_input = self.llm_client_config.format_messages(msg_structure, content=content)

			response = await self.llm_client_config.send_request(self.llm_provider_client, msg_input)

			try:
				# Clean and parse JSON response
				message_content = response.replace("```json", "").replace("```", "").strip()
				parsed_data = json.loads(message_content)

				return self._parse_chunks(parsed_data)

			except json.JSONDecodeError:
				logging.error("Error decoding JSON from LLM response")
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
		# tasks = [self.process_entry(entry) for entry in raw_data if "content" in entry]
		tasks = [self.process_entry(entry) for entry in raw_data if entry.content]
		results = await asyncio.gather(*tasks)
		return [item for sublist in results for item in sublist] 