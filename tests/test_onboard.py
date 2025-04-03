import asyncio
from config.config import AIAgentOnboardRequest
from onboard_workflow.onboard import GenerateDataSnapshot
import json

async def generate_data_snapshot():
    test_request = AIAgentOnboardRequest(
        session_id="session_123",
        urls=['https://coldbean.ai/', 'https://www.crunchbase.com/'],
    )

    generator = GenerateDataSnapshot(test_request)

    responses = await generator.get_data()

    response_dicts = [response.model_dump() for response in responses]

    
    
    with open("data/data_snapshot_clean.json", "w") as f:
        json.dump(response_dicts, f, indent=4)


# Run the test
asyncio.run(generate_data_snapshot())
