from openai import OpenAI
import os
from typing import List
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility

OPENAI_KEY = os.environ['OPENAI_API_KEY']

client = OpenAI()

class DataHandle:
    
    def __init__(self, OPENAI_KEY, data):
        self.client = OpenAI(api_key=OPENAI_KEY)
        self.data = data
        
        connections.connect(host='localhost', port='19530')
        
        self.dim = 1536
        
        self.fields = [
            FieldSchema(name="question", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="answer", dtype=DataType.VARCHAR, max_length=2000),
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
        else:
            self.collection = Collection(name="faq")
            
        self.collection.load()
        
    def text_embedding(self, question: str) -> List[str]:
        response = client.embeddings.create(
            input=question,
            model="text-embedding-3-small"
        )
        return response.data[0].embedding
    
    def insert_FAQ(self, faq_data):
        questions = []
        answers = []
        embeddings = []
        
        for question, answer in faq_data.items():
            questions.append(question)
            answers.append(answer)
            embeddings.append(self.text_embedding(question=question))
            
        
        field_data = [
            questions,
            answers,
            embeddings
        ]
    
        self.collection.insert(field_data)
        self.collection.flush()
        
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