from fastapi import FastAPI
from .embedding.openai_embedding import DataHandle

app = FastAPI()

@app.post('/openai_embedding')
def vector_embedding_FAQ(path : str):
    
    return path

@app.post('/openai_faq_test')
def test_faq(question: str):
    """사용자가 질문을 입력하면 Milvus에서 유사한 질문을 찾아 답변을 반환"""
    data_handle = DataHandle()
    answer = data_handle.search_similar_question(question)
    return {"question": question, "answer": answer}
@app.post('/openai_faq')
def insert_faq(data):
    return data