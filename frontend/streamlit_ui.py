import streamlit as st
# import validators
from urllib.parse import urlparse
import asyncio
from config.config import AIAgentOnboardRequest
from onboard_workflow.onboard import GenerateDataSnapshot
import json

st.set_page_config(layout="wide")

# function to generate data snapshot
# this function is called when the "Scrape Webpages" button is clicked
async def generate_data_snapshot(valid_urls, selected_llm):
  test_request = AIAgentOnboardRequest(
    session_id= "session_123_" + selected_llm,
    urls=valid_urls,
    files=[],
  )

  print(f"Creating request for session {test_request.session_id} with LLM: {selected_llm}")
  print(f"URLs: {test_request.urls}")

  generator = GenerateDataSnapshot(test_request, llm_choice=selected_llm)
  print("GENERATOR CREATED")
  responses = await generator.get_data()
  response_dicts = [response.model_dump() for response in responses]

  output_file = f"data/data_snapshot_clean_{selected_llm}.json"
  with open(output_file, "w") as f:
    json.dump(response_dicts, f, indent=4)
  st.success(f"Scraping finished! Data saved to {output_file}")


# function to handle text change in textbox and update URL list
def url_changed(index):
  st.session_state.urls_list[index] = st.session_state[f"url_input_{index}"]

# function to remove URL from session state
def remove_url(index_to_remove):
  if len(st.session_state.urls_list) > 1:
    del st.session_state.urls_list[index_to_remove]
    if f"url_input_{index_to_remove}" in st.session_state:
      del st.session_state[f"url_input_{index_to_remove}"]
  # remove last one and restart URL count
  else:
    st.session_state.urls_list[0] = ""
    st.session_state["url_input_0"] = ""

# function to validate URL
# def valid_url(url):
#   check = validators.url(url, private=None)
#   print(url, "-", check)
#   if not check:
#     print("URL:", url, "is INVALID!")
#   return check

# initialize session - start with one empty string in URL list
if 'urls_list' not in st.session_state:
  st.session_state.urls_list = [""] 

# initialize widget state to match list length
for i in range(len(st.session_state.urls_list)):
  widget_key = f"url_input_{i}"
  if widget_key not in st.session_state:
    st.session_state[widget_key] = st.session_state.urls_list[i]

st.title("Extractify")
st.subheader("DS 5983 - Final Project")

st.write("Enter URLs below:")

# div for URL boxes
input_container = st.container()
with input_container:
  # iterate through URL list in session state
  indices_to_remove = []
  for i, url_value in enumerate(st.session_state.urls_list):
    col1, col2 = st.columns([4, 1])
    with col1:
      st.text_input(
        label=f"URL {i + 1}",
        key=f"url_input_{i}",     # unique key for widget state
        value=url_value,          # display URL from list (if exists)
        on_change=url_changed,    # callback to update URL list
        args=(i,),                # pass index to callback func - url_changed(i)
        placeholder="https://www.example.com"
      )
    with col2:
      if i > 0:
        st.text("")
        # button to remove specific URL box 
        if st.button("Remove", key=f"remove_{i}"):
          indices_to_remove.append(i)

  # remove URLs
  if indices_to_remove:
    # remove indices in reverse order to avoid shifting issues
    for i in sorted(indices_to_remove, reverse=True):
      remove_url(i)
    
    # rerun after processing removals
    st.rerun()

# add new URL and scrape URLs columns
col_add, col_llm_select, col_scrape = st.columns([1, 1, 3])
with col_add:
  if st.button("Add another URL"):
    st.session_state.urls_list.append("")

    # update state for new input box
    new_widget_key = f"url_input_{len(st.session_state.urls_list) - 1}"
    st.session_state[new_widget_key] = ""
    
    # rerun to show new text box
    st.rerun()

with col_llm_select:
  llm_options = ["OpenAI", "Gemma"]
  selected_llm = st.selectbox(
      "Choose LLM for Chunking:",
      options=llm_options,
      key='selected_llm'
  )

with col_scrape:
  if st.button("Scrape Webpages", type="primary"):
    # get URLs from session state URL list
    # valid_urls = [url for url in st.session_state.urls_list if valid_url(url)]
    valid_urls = st.session_state.urls_list

    if valid_urls:
      llm_to_use = st.session_state.selected_llm
      st.info(f"Starting scraping process for {len(valid_urls)} URL(s)...")
      st.write("URLs to be scraped:")
      for url in valid_urls:
        st.write(f"- {url}")
      with st.spinner(f"Scraping and processing with {llm_to_use}..."):
        try:
          asyncio.run(generate_data_snapshot(valid_urls, llm_to_use))
        except Exception as e:
          st.error(f"An error occurred during scraping: {e}")

    else:
      st.warning("Please enter at least one valid URL!")
