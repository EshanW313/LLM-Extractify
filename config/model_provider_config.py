from pydantic import BaseModel
from typing import Any, List, Dict
from openai import OpenAI
import yaml
import google.generativeai as gen_ai
from google import genai
from google.genai import types
import logging

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
        elif self.api == "google_ai":
            if not self.api_key or self.api_key == "MISSING":
                logging.error("Error: Google AI API key is missing.")
                return None
            client_instance = genai.Client(api_key=self.api_key)
            self._llm_client_instance = client_instance
            return client_instance
        else:
            logging.error(f"Unsupported API provider: {self.api}")
            return None

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
            logging.error(f"Error reading YAML file: {e}")
            return []
        
    def format_messages(self, messages: List[Dict[str, str]], **kwargs) -> List[Dict[str, str]]:
        if self.api == "openai":
            formatted_messages = []
            for message in messages:
                content = message['content'].format(**kwargs)
                formatted_messages.append({'role': message.get('role', 'user'), 'content': content})
            return formatted_messages

        elif self.api == "google_ai":
            prompt_parts = []
            for message in messages:
                content = message.get('content', '').format(**kwargs)
                prompt_parts.append(content)
            return "\n\n".join(prompt_parts)
        
        else:
            logging.error(f"Warning: Unsupported API type '{self.api}' in format_messages.")
            return []

        
    async def send_request(self, client, model_messages: List[Dict[str, str]]):
        """
        Send a request to the model provider and return the response.

        Args:
        - model_messages: List of messages to send to the model provider.

        Returns:
        - str: Response from the model
        """
        if self.api == "openai":
            logging.info("Sending request to OpenAI...")
            response = client.chat.completions.create(
                messages=model_messages,
                **self.params,
            )
            return response.choices[0].message.content.strip()

        elif self.api == "google_ai":
            logging.info("Sending request to Google AI (Gemma)...")
            genai_module = client
            try:
                model_name = self.params.get("model", "gemma-3-27b-it")
                generation_config = {
                    'max_output_tokens': self.params.get("max_output_tokens", 8192),
                    'temperature': self.params.get("temperature", 0.1),
                    # top_p=self.params.get("top_p", None),
                    # top_k=self.params.get("top_k", None),
                }
                safety_settings = [
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                ]
                
                response = await genai_module.aio.models.generate_content(
                    model='gemma-3-27b-it',
                    contents=model_messages,
                    # generation_config=generation_config,
                    # safety_settings=safety_settings,
                    # request_options={"timeout": 600}
                )

                if not response.candidates:
                    logging.warning("Warning: Response was blocked, possibly due to safety settings.")
                    return "Error: Response blocked by safety filters."

                return response.text.strip()
            
            except Exception as e:
                logging.error("Error sending request to Google AI:", e)
                return "Error: {e}"
        
        else:
            logging.error("send_request not implemented for API provider:", {self.api})
            return "Error: Unsupported provider"
            