from fastapi import FastAPI
from embedding.openai_embedding import DataHandle
import pandas as pd

app = FastAPI()

@app.post('/openai_embedding')
def vector_embedding_FAQ(path : str):
    
    return path

@app.post('/openai_faq_test')
def test_faq(question: str):
    data_handle = DataHandle()
    answer = data_handle.search_similar_question(question)
    return {"question": question, "answer": answer}

# 데이터 삽입 테스트
@app.post('/openai_faq')
def insert_faq():
    faq_data = '/app/api/utils/preprocess_final_data.pkl'
    data_handle= DataHandle()
    
    # df = pd.read_pickle(faq_data)
    # data_handle.insert_FAQ(df)
    return "sucess"