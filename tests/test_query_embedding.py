import asyncio
import pandas as pd
from collection_creator.query_milvus import ZillizVectorSearch
from utils.services import EmbeddingService
from typing import List, Dict, Any

async def run_query():
    embedding_service = EmbeddingService()
    collection_names = ["session_123_OpenAI", "session_123_Gemma"]
    df = pd.read_csv('data/faq_test_queries.csv')
    top_k = 3

    for collection_name in collection_names:
        client = ZillizVectorSearch()
        client.load_collection(collection_name)

        # 1) collect raw hits per query, unwrapping the outer list
        raw_results: List[List[Dict[str,Any]]] = []
        for query_text in df['query']:
            emb = await embedding_service.get_query_embeddings(query=query_text)
            hits_batch = client.search(emb, vector_field="vector", top_k=top_k)
            hits = hits_batch[0]            # ← unwrap here
            raw_results.append(hits)

        # 2) parse hits into dicts
        parsed_results: List[List[Dict[str,Any]]] = []
        for hits in raw_results:
            batch = []
            for hit in hits:
                batch.append({
                    "distance": hit["distance"],
                    "content":  hit["content"],
                    "overview": hit["overview"],
                })
            parsed_results.append(batch)

        # 3) flatten into DataFrame columns
        flat_rows = []
        for pr in parsed_results:
            row: Dict[str, Any] = {}
            for rank in range(top_k):
                if rank < len(pr):
                    rec = pr[rank]
                    row[f"{collection_name}_content_{rank+1}"]  = rec["content"]
                    row[f"{collection_name}_overview_{rank+1}"] = rec["overview"]
                    row[f"{collection_name}_distance_{rank+1}"] = rec["distance"]
                else:
                    row[f"{collection_name}_content_{rank+1}"]  = None
                    row[f"{collection_name}_overview_{rank+1}"] = None
                    row[f"{collection_name}_distance_{rank+1}"] = None
            flat_rows.append(row)

        flat_df = pd.DataFrame(flat_rows)
        df = pd.concat([df, flat_df], axis=1)

    print(df.head())
    df.to_csv('data/faq_with_flattened_results.csv', index=False)

asyncio.run(run_query())
