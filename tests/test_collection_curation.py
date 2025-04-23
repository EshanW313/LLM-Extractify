import json
from config.config import AIAgentOnboardingDataResponse
from onboard_workflow.onboard import DataUploader
import asyncio

async def test_upload_data_from_json():
    """
    dTest creating a query collection
    """
    # Load test data from JSON file
    data_path = "data/data_snapshot_clean_llm_project.json"
    with open(data_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    # Convert to model instances
    model_data = [AIAgentOnboardingDataResponse(**item) for item in raw_data]

    # Instantiate and call
    uploader = DataUploader()
    result = await uploader.upload_data(model_data)

    # Assertions
    assert result["status_code"] == 200
    assert result["message"] == "Data uploaded successfully!"

if __name__ == "__main__":
    asyncio.run(test_upload_data_from_json())