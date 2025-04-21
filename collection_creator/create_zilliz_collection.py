from fastapi import HTTPException
from pymilvus import MilvusClient, DataType
from config.config import zillizconfig
from pymilvus.exceptions import MilvusException, ConnectError
import logging

class ZillizClient:
    def __init__(self):
        self.client=None
        self.connect()

    def connect(self):
        try:
            print("Connecting to Zilliz Cloud...")
            self.client = MilvusClient(
                uri=zillizconfig.ZILLIZ_CLOUD_URI,
                token=zillizconfig.ZILLIZ_AUTH_TOKEN,
            )
            print("Successfully connected to Zilliz Cloud")
        except ConnectError as e:
            print(f"Failed to connect to Zilliz Cloud: {str(e)}")
            raise ConnectionError(f"Could not connect to Zilliz Cloud: {str(e)}")
        except Exception as e:
            print(f"Unexpected error when connecting to Zilliz Cloud: {str(e)}")
            raise
        
    def create_collection(self, collection_name: str):
        try:
            if collection_name in self.client.list_collections():
                print(f"Collection {collection_name} exists.")
                return 
            print(f"Collection doesnt exist, Creating new collection: {collection_name}")
            schema = self.client.create_schema(
                enable_dynamic_field=False,
                description=f"Schema for {collection_name}'s collection.",
            )
            schema.add_field(
                field_name="id",
                datatype=DataType.INT64,
                description="Unique identifier",
                is_primary=True,
                auto_id=True,
            )
            schema.add_field(
                field_name="content",
                datatype=DataType.VARCHAR,
                description="QA pair",
                max_length=65535
            )
            schema.add_field(
                field_name="vector",
                datatype=DataType.FLOAT_VECTOR,
                description="Content Embedding",
                dim=zillizconfig.VECTOR_DIMENSION
            )
            schema.add_field(
                field_name="overview",
                datatype=DataType.VARCHAR,
                description="Question/Overview",
                max_length=65535
            )
            schema.add_field(
                field_name="source",
                datatype=DataType.VARCHAR,
                description="Source from which data is extracted from",
                max_length=256
            )
            schema.add_field(
                field_name="url",
                datatype=DataType.VARCHAR,
                description="URL from which data is extracted from",
                max_length=500
            )
            index_params = self.client.prepare_index_params()
            index_params.add_index(
                field_name="vector", metric_type="COSINE", index_type="AUTOINDEX"
            )
            self.client.create_collection(
                collection_name=collection_name, schema=schema, index_params=index_params, using="default"
            )
        
            print(f"Collection {collection_name} created successfully.")
            return 
        except MilvusException as e:
            print(f"Milvus error when creating collection {collection_name}: {str(e)}")
            raise HTTPException(status_code=500, detail="Milvus error when creating collection")
        except Exception as e:
            print(f"Unexpected error when creating collection {collection_name}: {str(e)}")
            raise HTTPException(status_code=500, detail="Unexpected error when creating collection")

    def insert_records(self, collection_name: str, records: list):
        try:
            entities = []
            for record in records:
                entities.append({
                    "content" :record.content,
                    "overview": record.overview,
                    "source": record.meta_data.source,
                    "url": record.meta_data.url or "",
                    "vector": record.vector
                })
            print(f"Inserting {len(records)} records into collection {collection_name}")
            self.client.insert(collection_name=collection_name, data=entities)
            print(f"Successfully inserted {len(records)} records into {collection_name}")
        except MilvusException as e:
            print(f"Milvus error when inserting records into {collection_name}: {str(e)}")
            raise HTTPException(status_code=500, detail="Milvus error when inserting records")
        except Exception as e:
            print(f"Unexpected error when inserting records into {collection_name}: {str(e)}")
            raise HTTPException(status_code=500, detail="Unexpected error when inserting records")