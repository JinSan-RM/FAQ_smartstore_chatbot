from embedding.openai_embedding import DataHandle
from milvus.FAQ_RAG import DBHandling
from openai import OpenAI

from fastapi import FastAPI
from fastapi.responses import StreamingResponse, JSONResponse
import pandas as pd
import json

app = FastAPI()

@app.post('/openai_embedding')
def vector_embedding_FAQ(path : str):
    
    return path

@app.post('/openai_faq_test')
def test_faq(question: str):
    data_handle = DataHandle()
    answer = data_handle.search_similar_question(question)
    return {"question": question, "answer": answer}

#데이터 삽입 테스트
@app.post('/openai_faq')
def insert_faq():
    faq_data = '/app/api/utils/preprocess_final_data.pkl'
    data_handle= DataHandle()
    
    df = pd.read_pickle(faq_data)
    data_handle.insert_FAQ(df)
    return "sucess"

@app.post('/openai_faq_search')
async def search_faq(query: str):
    try:
        db_handle = DBHandling()

        retrieved_context = db_handle.search_FAQ(query=query)

        if not retrieved_context or not retrieved_context.get('answer'):
            return StreamingResponse(
                db_handle.generate_error_response("저는 스마트 스토어 FAQ를 위한 챗봇입니다. 스마트 스토어에 대한 질문을 부탁드립니다."),
                media_type='text/event-stream'
            )

        print(f"retrieved_context : {retrieved_context}")
        async def generate():
            try:
                async for content in db_handle.generate_response(query, retrieved_context):
                    yield f"data: {json.dumps({'content': content})}\n\n"
                yield f"data: [DONE]\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

        return StreamingResponse(generate(), media_type='text/event-stream')

    except Exception as e:
        print(f"Error in search_faq: {str(e)}")
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )
