import asyncio
import pandas as pd
from collection_creator.query_milvus import ZillizVectorSearch
from utils.services import EmbeddingService
from typing import List, Dict, Any

async def run_query():
    """
    Test embedding curation service and create a collection to search - model evals
    """
    embedding_service = EmbeddingService()
    collection_names = ["openai_session_2909584092", "gemma_session_7245098249"]
    df = pd.read_csv('data/faq_test_queries.csv')
    top_k = 3

    for collection_name in collection_names:
        client = ZillizVectorSearch()
        client.load_collection(collection_name)

        # 1) Collect raw hits per query for both embeddings
        raw_results: List[List[Dict[str,Any]]] = []
        raw_results_openai: List[List[Dict[str,Any]]] = []
        for query_text in df['query']:
            emb_custom = await embedding_service.get_query_embeddings(query=query_text)
            emb_openai = await embedding_service.get_openaiembeddings(text=query_text)

            custom_batch = client.search(emb_custom, vector_field="vector", top_k=top_k)
            openai_batch = client.search(emb_openai, vector_field="vector_openai", top_k=top_k)

            # unwrap the first (and only) batch
            raw_results.append(custom_batch[0])
            raw_results_openai.append(openai_batch[0])

        # 2) Parse hits into dicts
        def parse_hits(raw_hits: List[List[Any]]) -> List[List[Dict[str,Any]]]:
            parsed = []
            for hits in raw_hits:
                batch = []
                for hit in hits:
                    batch.append({
                        "distance": hit["distance"],
                        "content":  hit["content"],
                        "overview": hit["overview"],
                    })
                parsed.append(batch)
            return parsed

        parsed_custom = parse_hits(raw_results)
        parsed_openai = parse_hits(raw_results_openai)

        # 3) Flatten into DataFrame columns (side-by-side)
        flat_rows = []
        for custom_hits, openai_hits in zip(parsed_custom, parsed_openai):
            row: Dict[str, Any] = {}
            for rank in range(top_k):
                # custom embedding result
                if rank < len(custom_hits):
                    rec = custom_hits[rank]
                    row[f"{collection_name}_content_{rank+1}"]   = rec["content"]
                    row[f"{collection_name}_overview_{rank+1}"]  = rec["overview"]
                    row[f"{collection_name}_distance_{rank+1}"]  = rec["distance"]
                else:
                    row[f"{collection_name}_content_{rank+1}"]   = None
                    row[f"{collection_name}_overview_{rank+1}"]  = None
                    row[f"{collection_name}_distance_{rank+1}"]  = None

                # OpenAI embedding result
                if rank < len(openai_hits):
                    rec_oai = openai_hits[rank]
                    row[f"{collection_name}_openai_content_{rank+1}"]   = rec_oai["content"]
                    row[f"{collection_name}_openai_overview_{rank+1}"]  = rec_oai["overview"]
                    row[f"{collection_name}_openai_distance_{rank+1}"]  = rec_oai["distance"]
                else:
                    row[f"{collection_name}_openai_content_{rank+1}"]   = None
                    row[f"{collection_name}_openai_overview_{rank+1}"]  = None
                    row[f"{collection_name}_openai_distance_{rank+1}"]  = None

            flat_rows.append(row)

        flat_df = pd.DataFrame(flat_rows)
        df = pd.concat([df, flat_df], axis=1)

    print(df.head())
    df.to_csv('data/faq_with_flattened_results.csv', index=False)

asyncio.run(run_query())
