from openai import OpenAI
import os
import re
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
        #     print("이미 있지만 다시 지웠다가 생성.")
            
        self.dim = 1536
        
        self.fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="question", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="answer", dtype=DataType.VARCHAR, max_length=60535),
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
            print("컬렉션 생성")
        else:
            self.collection = Collection(name="faq")
            print("컬렉션 이미 존재")
            
        self.collection.load()
        
        self.print_collection_info()
        self.print_field_max_length()
        
    def print_field_max_length(self):
        for field in self.collection.schema.fields:
            if field.name == "answer":
                print(f"answer 필드 맥스 렝스: {field.max_length}")
    
    def print_collection_info(self):


        print(f"DB 컬렉션 이름 : {self.collection.name}")
        

        for field in self.collection.schema.fields:
            print(f"-field {field.name} (type: {field.dtype})")
        
        indexes = self.collection.indexes
        print("번호 :")
        if indexes:
            for index in indexes:
                print(f"- Index type: {index.params.get('index_type')}, Metric: {index.params.get('metric_type')}")
        else:
            print("번호 못 찾음.")
    
    def text_embedding(self, question: str):
        print(f"임베딩 생성할 질문: {question}")
        response = client.embeddings.create(
            input=question,
            model="text-embedding-3-small"
        )
        # print(f"임베딩 생성: {response.data[0].embedding}")
        return response.data[0].embedding

    
    def preprocess_long_text(self, text, max_length=12000, overlap=200):

        if len(text) <= max_length:
            return [text]
        
        # 문장 단위로 분리
        import re
        sentences = re.split('(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = ''
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) < max_length:
                current_chunk += ' ' + sentence if current_chunk else sentence
            else:

                if current_chunk:
                    chunks.append(current_chunk.strip())
                

                if len(sentence) > max_length:

                    while len(sentence) > max_length:
                        chunks.append(sentence[:max_length-overlap])
                        sentence = sentence[max_length-overlap:]
                    current_chunk = sentence
                else:
                    current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks

    
    def insert_FAQ(self, faq_data):
        MAX_LENGTH = 60000
        processed_data = []
        
        for question, answer in faq_data.items():
            try:
                # 답변이 긴 경우 분할 처리
                if len(answer) > MAX_LENGTH:
                    chunks = self.preprocess_long_text(answer, MAX_LENGTH)
                    for i, chunk in enumerate(chunks):

                        chunk_question = f"{question} (파트 {i+1}/{len(chunks)})"
                        processed_data.append((chunk_question, chunk))
                else:
                    processed_data.append((question, answer))
                    
            except Exception as e:
                print(f"FAQ 에러 : {e}")
                continue
        
        BATCH_SIZE = 1000
        
        for i in range(0, len(processed_data), BATCH_SIZE):
            batch = processed_data[i:i + BATCH_SIZE]
            batch_questions = []
            batch_answers = []
            batch_embeddings = []
            
            for question, answer in batch:
                try:
                    # 띄어쓰기 줄바꿈 삭제 maxlength 초과?
                    answer = re.sub(r'[\r\n\t]', ' ', answer).strip()
                    
                    if len(answer) > MAX_LENGTH:
                        answer = answer[:MAX_LENGTH]
                    
                    batch_questions.append(question)
                    batch_answers.append(answer)
                    batch_embeddings.append(self.text_embedding(question=question))
                    
                except Exception as e:
                    print(f"Error processing batch item: {e}")
                    continue
            
            try:
                field_data = [
                    batch_questions,
                    batch_answers,
                    batch_embeddings
                ]
                self.collection.insert(field_data)
                self.collection.flush()
                print(f"Inserted batch {i//BATCH_SIZE + 1}")
                
            except Exception as e:
                print(f"배치 삽입 중 에러: {e}")
            
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
        return "답을 찾을 수 없어요~."
    
    def check_stored_data(self):
        res = self.collection.query(
            expr="id >= 0",
            output_fields=["question", "answer"],
            limit=5  
        )
        print("샘플 데이터 확인:", res)