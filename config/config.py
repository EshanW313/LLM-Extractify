from typing import Optional, List, Dict, Literal, Any
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
import os
from dotenv import load_dotenv
from firecrawl import FirecrawlApp
from pydantic import BaseModel, ConfigDict
from config.model_provider_config import ModelProviderConfig
from mistralai import Mistral

load_dotenv()

class AIAgentOnboardRequest(BaseModel):
	session_id: str
	urls: List[str]
	files: List[str]
	
class metaData(BaseModel):
	session_id: str
	source: Literal["web", "KB", "email", "user_upload"]
	url: Optional[str] = None

class AIAgentOnboardingDataResponse(BaseModel):
	meta_data: metaData
	content: str
	overview: str
	vector: Optional[List[float]] = Field("", description="Vector embedding of the content")

firecrawl_config: FirecrawlApp = FirecrawlApp(
	api_key=os.getenv("FIRECRAWL_API_KEY")
)

### Setting up the OpenAI Config ###
class OpenAIConfig(ModelProviderConfig):
	model_config = ConfigDict(extra="allow")
	api: str = "openai"
	base_url: str = "https://api.openai.com/v1"
	messages: str = "MISSING"
	tokenizer: str = "tiktoken"
	params: Dict[str, Any] = {
		"model": "gpt-4o-mini",
		"stop": None,
		"n": 1,
		"temperature": 0
	}
	# Pass the openai api key from .env file
	api_key: str = os.getenv("OPENAI_API_KEY", "MISSING")

chunk_and_clean_task_app: OpenAIConfig = OpenAIConfig(
	messages = "config/prompts/message_chunk.yaml",
)

chunk_and_clean_task_app.update_params(
	model = "gpt-4o",
	stop = None,
	n = 1,
	max_tokens = 10000,
	temperature = 0,
	response_format={"type": "json_object"}
)

mistralai_config: Mistral = Mistral(
    api_key=os.getenv("MISTRAL_API_KEY")
)

class ZillizConfig(BaseSettings):
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
    ZILLIZ_AUTH_TOKEN: str = os.getenv("ZILLIZ_AUTH_TOKEN")
    ZILLIZ_CLOUD_URI: str =os.getenv("ZILLIZ_CLOUD_URI")
    VECTOR_DIMENSION: int = 3072
    ZILLIZ_INSERTION_BATCH_SIZE: int = 50

zillizconfig = ZillizConfig()

class GoogleAIConfig(ModelProviderConfig):
    model_config = ConfigDict(extra="allow")
    api: str = "google_ai"
    base_url: str = ""
    messages: str = "MISSING"
    tokenizer: str = "google_genai"
    params: Dict[str, Any] = {
        "model": "gemma-3-27b-it",
        "temperature": 0.0,
        "max_output_tokens": 8192,
    }
    api_key: str = os.getenv("GEMMA_API_KEY", "MISSING")

gemma_chunk_and_clean_task_app: GoogleAIConfig = GoogleAIConfig(
    messages="config/prompts/message_chunk_gemma.yaml",
)

gemma_chunk_and_clean_task_app.update_params(
    model="gemma-3-27b-it",
    temperature=0.0,
)