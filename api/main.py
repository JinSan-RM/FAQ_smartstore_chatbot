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

import sys

@app.post('/openai_faq_search')
async def search_faq(query: str):
    try:
        print("search_faq 시작")
        db_handle = DBHandling()
        retrieved_context = db_handle.search_FAQ(query=query)
        print(f"retrieved_context: {retrieved_context}")

        def generate():
            print("generate 함수 시작")
            for content in db_handle.generate_response(query=query, retrieved_context=retrieved_context):
                print(content, flush=True)
                yield f"data: {json.dumps({'content': content})}\n\n"
            yield "data: [DONE]\n\n"
            print("generate 함수 종료")

        print("StreamingResponse 반환 직전")
        return StreamingResponse(generate(), media_type='text/event-stream')
    
    except Exception as e:
        print(f"search_faq 오류: {str(e)}")
        return StreamingResponse(
            content=iter([f"data: {json.dumps({'error': str(e)})}\n\n"]),
            media_type='text/event-stream'
        )
# import os, re, json
# import asyncio
# import sys
# OPENAI_KEY = os.environ['OPENAI_API_KEY']

# client = OpenAI(api_key=OPENAI_KEY)
# @app.post("/stream")
# async def stream_response(query: str):
#     db_handle = DBHandling()
#     retrieved_context = db_handle.search_FAQ(query=query)
#     # print(f"retrieved_context: {retrieved_context}")
#     prompt = f"""다음은 네이버 스마트스토어 FAQ 내용입니다:
#         질문: {retrieved_context['question']}
#         답변: {retrieved_context['answer']}
        
#         사용자 질문: {query}
        
#         위 FAQ 내용을 참고하여 사용자의 질문에 답변해주세요. 
#         답변은 자연스러운 대화체로 작성하되, FAQ의 정확한 정보만 포함해야 합니다."""
#     def generate():
#         stream = client.chat.completions.create(
#             model="gpt-4",
#             messages=[{"role": "user", "content": prompt}],
#             stream=True
#         )
        
#         for chunk in stream:
#             if chunk.choices[0].delta.content is not None:
#                 print(chunk.choices[0].delta.content, flush=True)
#                 yield f"data: {json.dumps({'content': chunk.choices[0].delta.content})}\n\n"
        
#         yield "data: [DONE]\n\n"

#     return StreamingResponse(generate(), media_type="text/event-stream")
    # return StreamingResponse(db_handle.generate_response(query, retrieved_context), media_type="text/event-stream")
