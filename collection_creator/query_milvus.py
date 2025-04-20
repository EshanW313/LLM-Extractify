from typing import List, Optional, Dict, Any
from pymilvus import connections, Collection
from pymilvus.exceptions import ConnectError
from config.config import zillizconfig

class ZillizVectorSearch:
    """
    Client for connecting to Zilliz Cloud (Milvus) and performing cosine similarity vector searches.
    """
    def __init__(self, alias: str = "zilliz-cloud", secure: bool = True):
        self.alias = alias
        self.collection: Optional[Collection] = None
        self._connect(secure)

    def _connect(self, secure: bool = True) -> None:
        """
        Establish and register a Milvus connection alias for Zilliz Cloud.
        """
        try:
            print(f"Connecting to Zilliz Cloud (alias={self.alias})...")
            connections.connect(
                alias=self.alias,
                uri=zillizconfig.ZILLIZ_CLOUD_URI,
                token=zillizconfig.ZILLIZ_AUTH_TOKEN,
                secure=secure,
            )
            print("Successfully connected to Zilliz Cloud.")
        except ConnectError as e:
            raise ConnectionError(f"Could not connect to Zilliz Cloud: {e}")

    def load_collection(self, collection_name: str) -> None:
        """
        Load a Milvus collection into the client for subsequent search operations.

        Args:
            collection_name: The name of the collection to load.
        """
        self.collection = Collection(name=collection_name, using=self.alias)

    def search(
        self,
        embeddings: List[List[float]],
        vector_field: str = "vector",
        top_k: int = 3,
        metric: str = "COSINE",
        nprobe: int = 10,
        output_fields: Optional[List[str]] = ['content', 'overview'],
        expr: Optional[str] = None,
    ) -> List[List[Dict[str, Any]]]:
        """
        Perform a vector similarity search against the loaded collection.

        Args:
            embeddings: A list of query vectors.
            vector_field: The name of the vector field in the schema.
            top_k: Maximum number of results to return per query.
            metric: Similarity metric (e.g., 'COSINE', 'L2').
            nprobe: Number of probes controlling recall/latency tradeoff.
            output_fields: Additional metadata fields to return alongside id and distance.
            expr: Optional filter expression (e.g., "status == \"active\"").

        Returns:
            A nested list where each inner list corresponds to one query vector,
            containing dicts with 'id', 'distance', and any requested output_fields.
        """
        if self.collection is None:
            raise ValueError("Collection not loaded; call load_collection() first.")

        search_params = {"metric_type": metric, "params": {"nprobe": nprobe}}

        results = self.collection.search(
            data=embeddings,
            anns_field=vector_field,
            param=search_params,
            limit=top_k,
            expr=expr,
            output_fields=output_fields or [],
            consistency_level="Session",
        )

        parsed_results: List[List[Dict[str, Any]]] = []
        for hits in results:
            batch_results: List[Dict[str, Any]] = []
            for hit in hits:
                record: Dict[str, Any] = {"id": hit.id, "distance": hit.distance}
                if output_fields:
                    for field in output_fields:
                        record[field] = hit.entity.get(field)
                batch_results.append(record)
            parsed_results.append(batch_results)

        return parsed_results