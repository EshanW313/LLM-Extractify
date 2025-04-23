#!/usr/bin/env python3
"""
Test script for updating LLM parameters and running DataChunker with different GPT models.
"""

import asyncio
import logging
from copy import deepcopy

from config.config import chunk_and_clean_task_app
from onboard_workflow.onboard import GenerateDataSnapshot
from config.config import AIAgentOnboardRequest

MODEL_LIST = [
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-4.1",
    "gpt-4.1-mini"
]

async def test_model(model_name: str):
    llm_config = deepcopy(chunk_and_clean_task_app)
    llm_config.update_params(
        model=model_name,
        stop=None,
        n=1,
        max_tokens=10000,
        temperature=0,
        response_format={"type": "json_object"}
    )

    test_request = AIAgentOnboardRequest(
        session_id="session_123",
        urls=['https://www.sas.com/en/events/sas-innovate/faq.html', 'https://aiconference.com/faq/'],
        files=[]
    )

    generator = GenerateDataSnapshot(test_request, llm_choice="openai")
    generator.data_chunker.llm_config = llm_config

    result = await generator.get_data()
    print(f"Model: {model_name}\nResult: {result}\n")

async def main():
    logging.basicConfig(level=logging.INFO)
    for model in MODEL_LIST:
        await test_model(model)

if __name__ == "__main__":
    asyncio.run(main())
