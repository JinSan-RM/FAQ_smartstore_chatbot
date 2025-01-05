from openai import OpenAI
import os
from typing import List
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility

OPENAI_KEY = os.environ['OPENAI_API_KEY']

client = OpenAI()

class DataHandle:
    
    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_KEY)
        
        connections.connect(host='milvus', port='19530') 
        
        if utility.has_collection("faq"):
            collection = Collection("faq")
            collection.drop()
            print("Existing collection dropped.")
            
        self.dim = 1536
        
        self.fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="question", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="answer", dtype=DataType.VARCHAR, max_length=4000),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dim)
        ]
        
        self.schema = CollectionSchema(fields=self.fields, description="FAQ embeddings")
        if not utility.has_collection("faq"):
            self.collection = Collection(name="faq", schema=self.schema)

            index_params = {
                "metric_type": "COSINE",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 128}
            }
            self.collection.create_index(field_name="embedding", index_params=index_params)
            print("Milvus collection created and indexed.")
        else:
            self.collection = Collection(name="faq")
            print("Milvus collection already exists.")
            
        self.collection.load()
        
        self.print_collection_info()
    
    def print_collection_info(self):

        print("\nCollection Information:")

        print(f"Collection name: {self.collection.name}")
        

        for field in self.collection.schema.fields:
            print(f"-field {field.name} (type: {field.dtype})")
        
        indexes = self.collection.indexes
        print("Indexes:")
        if indexes:
            for index in indexes:
                print(f"- Index type: {index.params.get('index_type')}, Metric: {index.params.get('metric_type')}")
        else:
            print("No indexes found.")
    
    def text_embedding(self, question: str) -> List[str]:
        print(f"Generating embedding for question: {question}")
        response = client.embeddings.create(
            input=question,
            model="text-embedding-3-small"
        )
        print(f"Embedding generated: {response.data[0].embedding}")
        return response.data[0].embedding
    
    def insert_FAQ(self, faq_data):
        questions = []
        answers = []
        embeddings = []
        cnt = 0
        for question, answer in faq_data.items():
            print(f"Processing FAQ {cnt}: Question='{question}', Answer='{answer}'")
            questions.append(question)
            answers.append(answer)
            embeddings.append(self.text_embedding(question=question))
            cnt += 1
            
        
        field_data = [
            questions,
            answers,
            embeddings
        ]
        print("Inserting data into Milvus...")
    
        self.collection.insert(field_data)
        self.collection.flush()
        print(f"Data inserted. Total entities in collection: {self.collection.num_entities}")
        
    def search_similar_question(self, user_question: str, top_k: int = 1):
        user_embedding = self.text_embedding(user_question)
        
        search_result = self.collection.search(
            data=[user_embedding],
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"nprobe": 10}},
            limit=top_k
        )
        
        if search_result:
            return search_result[0].entity["answer"]
        return "Sorry, I couldn't find an answer."