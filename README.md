# LLM-Extractify

Installation steps for project:
- make sure poetry is installed - pip install poetry
- on CLI - poetry install 
- dependencies for the project include OpenAI API KEY, Gemma API KEY, Firecrawl API KEY, Mistral API KEY, Zilliz AUTH TOKEN
- replace Zilliz Cloud URI with your URL - in case a new cluster is to be created. Current cluster can be accessed with emails.
- make sure all above keys are placed in your .env, .env.example provides sample template, replace keys with your keys in order to execute the project

WINDOWS ISSUE:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process


Some test URLs:

https://foundation.wikimedia.org/wiki/Policy:Privacy_policy
https://docs.llamaindex.ai/en/stable/examples/embeddings/huggingface/

https://www.sas.com/en/events/sas-innovate/faq.html
https://aiconference.com/faq/
https://foundation.wikimedia.org/wiki/Policy:Privacy_policy

To test and run:
poetry install should create a venv. Make sure venv is activated and running on your terminal - mac command
source .venv/bin/activate

For running streamlit follow command:
poetry run streamlit run frontend/streamlit_ui.py

The test folder provides certain tests to verify onboard function(end to end processing)
To run any tests command to follow:
poetry run python tests/{insert-test-filename-here.py}
eg: poetry run python tests/test_onboard.py

For testing various models - switch to branch feat/model-evals
poetry run python tests/test_gpt_models.py
