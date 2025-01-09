from openai import OpenAI
import os
import re
from typing import List
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility

# Docker compose 파일의 environments에 적힌 openai api key 불러오기
OPENAI_KEY = os.environ['OPENAI_API_KEY']

client = OpenAI()

# ============================
#    vectorDB 컨트롤 함수
# ============================

class MilvusHandle:
    
    def __init__(self):
        # 데이터 핸들 모듈 호출 시 openai 클라이언트 생성
        self.client = OpenAI(api_key=OPENAI_KEY)
        
        # milvus DB connection 점검
        connections.connect(host='milvus', port='19530') 
        
        # ※ 최초 생성시 faq 데이터 혹시 모르니까 지웠다가 재 생성
        # ※ 필요에 따라서 주석 처리. 최초 한번만 진행하고 주석
        
        # if utility.has_collection("faq"):
        #     collection = Collection("faq")
        #     collection.drop()
        #     print("이미 있지만 다시 지웠다가 생성.")
        
        # 설정 차원 text-embedding-3-small = 512 or 1536
        self.dim = 1536
        
        # collection의 field 정의
        self.fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="question", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="answer", dtype=DataType.VARCHAR, max_length=60535),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dim)
        ]
        
        # 존재 유무 확인 후 컬렉션 있는지 체크 후 생성.
        self.schema = CollectionSchema(fields=self.fields, description="FAQ embeddings")
        if not utility.has_collection("faq"):
            self.collection = Collection(name="faq", schema=self.schema)
            # 텍스트 임베딩 데이터를 벡터간의 유사도로  조회 / cosine 유사도 기준으로 설정 
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
    
    # 대답이 긴 answer field의 max length 확인 차원        
    def print_field_max_length(self):
        for field in self.collection.schema.fields:
            if field.name == "answer":
                print(f"answer 필드 맥스 렝스: {field.max_length}")
    
    # milvus 컬렉션 생성시 필드랑 타입 조회
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
            
            
    # 텍스트 openai 'text-embedding-3-small' 모델 사용해서 embedding
    def text_embedding(self, question: str):
        print(f"임베딩 생성 질문: {question}")
        response = client.embeddings.create(
            input=question,
            model="text-embedding-3-small"
        )
        # print(f"임베딩 생성: {response.data[0].embedding}")
        return response.data[0].embedding


    # answer field보다 answer값이 길 경우 텍스트 분리해서 저장 할수 있게 나누기기
    def preprocess_long_text(self, text, max_length=60000, overlap=200):

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

    # milvus vector DB에 데이터 입력
    # 데이터를 입력하기 위해 전처리
    def insert_FAQ(self, faq_data):
        print("start insert", flush=True)
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
        
        # 데이터 분할 insert
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
                
            # field 구조대로 삽입 하기위해 묶음
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
                




# ======================== test code ======================================= #
    # RAG 방식의 임베딩 질문 기반의  결과 조회 해보기.            
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
    # milvus vector DB에
    def check_stored_data(self):
        res = self.collection.query(
            expr="id >= 0",
            output_fields=["question", "answer"],
            limit=5  
        )
        print("샘플 데이터 확인:", res)