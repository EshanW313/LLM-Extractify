import asyncio
from collection_creator.query_milvus import ZillizVectorSearch
from utils.services import EmbeddingService

async def run_query():
    """
    Single Query Retreival for evals
    """
    embedding_service = EmbeddingService()
    collection_names = ["session_123_OpenAI", "session_123_Gemma"]
    query = 'Can I attend the AI Conference online?'
    # For each collection, do the search and then expand into columns
    for collection_name in collection_names:
        print('RUNNING FOR COLLECTION: ', collection_name)
        client = ZillizVectorSearch()
        client.load_collection(collection_name)
        
        emb = await embedding_service.get_query_embeddings(query=query)
        hits = client.search(emb, vector_field="vector", top_k=3)  

        print(hits)

asyncio.run(run_query())
