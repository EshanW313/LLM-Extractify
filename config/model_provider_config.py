from pydantic import BaseModel
from typing import Any, List, Dict
from openai import OpenAI
import yaml

class ModelProviderConfig(BaseModel):
    api: str = "openai"
    base_url: str = "https://api.openai.com/v1"
    messages: str = "MISSING"
    tokenizer: str = "tiktoken"
    params: Dict[str, Any] = {
        "model": "gpt-4o-mini",
        "stop": None,
        "n": 1,
        "temperature": 0,
        "max_tokens": 100,
        "response_format": {"type": "json_object"}
    }
    api_key: str = "MISSING"

    def update_params(self, **kwargs):
        self.params.update(kwargs)

    def initialize_model_provider(self):
        if self.api == "openai":
            return OpenAI(
                base_url=self.base_url,
                api_key=self.api_key
            )


    def get_messages_from_yaml(self, yaml_file_path: str) -> List[Dict[str, str]]:
        try:
            with open(yaml_file_path, 'r') as file:
                yaml_content = yaml.safe_load(file)
            
            messages = []
            for role, content in yaml_content.items():
                if role in ["system", "user", "assistant"]:
                    messages.append({"role": role, "content": content})
            
            return messages
        except Exception as e:
            print(f"Error reading YAML file: {e}")
            return []
        
    def format_messages(self, messages: List[Dict[str, str]], **kwargs) -> List[Dict[str, str]]:
        formatted_messages = []
        for message in messages:
            content = message['content'].format(**kwargs)
            formatted_messages.append({'role': message['role'], 'content': content})
        return formatted_messages
        
    async def send_request(self, client, model_messages: List[Dict[str, str]]):
        """
        Send a request to the model provider and return the response.

        Args:
        - model_messages: List of messages to send to the model provider.

        Returns:
        - str: Response from the model
        """
        if self.api == "openai":
            print("Sending request to OpenAI...")
            response = client.chat.completions.create(
                messages=model_messages,
                **self.params,
            )
            return response.choices[0].message.content.strip()
            