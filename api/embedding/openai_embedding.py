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
        
        # if utility.has_collection("faq"):
        #     collection = Collection("faq")
        #     collection.drop()
        #     print("Existing collection dropped.")
            
        self.dim = 1536
        
        self.fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="question", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="answer", dtype=DataType.VARCHAR, max_length=16384),
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
        self.print_field_max_length()
        
    def print_field_max_length(self):
        # 필드의 max_length 출력
        for field in self.collection.schema.fields:
            if field.name == "answer":
                print(f"Field 'answer' max_length: {field.max_length}")
    
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
        MAX_LENGTH = 16384  # Milvus의 최대 문자열 길이 제한

        cnt = 0
        for question, answer in faq_data.items():
            try:
                print(f"Processing FAQ {cnt}: Question='{question[:50]}...', Answer length={len(answer)}")

                # Answer 길이 검사
                if len(answer) > MAX_LENGTH:
                    print(f"Error: Answer for FAQ {cnt} exceeds MAX_LENGTH: {len(answer)} characters. Skipping this entry.")
                    cnt += 1
                    continue

                # 문자열 전처리
                import re
                answer = answer[:MAX_LENGTH]
                answer = re.sub(r'[\r\n\t]', ' ', answer).strip()

                # 질문, 답변, 임베딩 준비
                embedding = self.text_embedding(question=question)

                # 데이터 삽입
                field_data = [
                    [question],
                    [answer],
                    [embedding]
                ]
                print(f"Inserting FAQ {cnt} into Milvus...")
                self.collection.insert(field_data)  # Milvus에 개별 삽입
                print(f"FAQ {cnt} inserted successfully.")
            except Exception as e:
                # 특정 데이터 삽입 중 오류 발생 시 로그 출력
                print(f"Error inserting FAQ {cnt}: {e}. Skipping this entry.")
            finally:
                cnt += 1

        try:
            print("Flushing data into Milvus...")
            self.collection.flush()  # 전체 데이터 플러시
            print(f"Data flushed successfully. Total entities in collection: {self.collection.num_entities}")
        except Exception as e:
            print(f"Error during data flush: {e}")
            
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