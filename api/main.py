from embedding.openai_embedding import MilvusHandle
from service.faq_response_handle import DataHandle, ResponseHandle
from context.context import ChatContext
#==============================================

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import pandas as pd
from collections import defaultdict, deque

# FastAPI app 생성
app = FastAPI()

# 전역변수로 사용자별 히스토리 저장, 느림
# 앱 새로 시작하면 초기화.
# 10개 까지만 저장.
history_store = defaultdict(lambda: deque(maxlen=10))
    
# pydantic형식을 사용해서 입력값 체크
class FAQRequest(BaseModel):
    query: str
    user_id: str

@app.post('/openai_faq_search')
async def search_faq(request: FAQRequest):
    print(f"질문 : {request.query} || 유저 : {request.user_id}")
    try:
        # 유저의 이전 질문과 상황을 저장하고 히스토리 맥락을 유지
        history = ChatContext(history_store)
        print(f"히스토리 : {history_store}")
        user_history = history.get_user_history(request.user_id)
        history.add_message(request.user_id, "user", request.query)
        
        # RAG 실행.
        db_handle = DataHandle()
        retrieved_context = db_handle.search_FAQ(query=request.query)
        print(f"retrieved_context: {retrieved_context}")

        def generate():
            # 데이터 생성 모듈 호출출 
            response = ResponseHandle()
            # 스마트 스토어에 대한 응답인지 아닌지 확인.
            # 부정 응답시 분기
            if (retrieved_context["answer"] == "" and
                "스마트 스토어에 대한 질문을 부탁드립니다." in retrieved_context["question"]):
                yield from response.handle_error_response( request, retrieved_context, history )
                
                return
            # 정상 응답시 분기
            else:
                yield from response.handle_normal_response( request, retrieved_context, user_history, history )
            
                return

        return StreamingResponse(generate(), media_type='text/event-stream')

    except Exception as e:
        print(f"search_faq 오류: {str(e)}")
        return e

# yield from 을 하면 생성함수로부터 모든 값들이 자동으로 yield를 하게됨. 몰랐던 부분.
    
    
# RAG 테스트 호출
@app.post('/openai_faq_test')
def test_RAG_faq(question: str):
    data_handle = MilvusHandle()
    answer = data_handle.search_similar_question(question)
    return {"question": question, "answer": answer}

#데이터 삽입 테스트
@app.post('/openai_faq')
def insert_faq():
    try:
        # docker 데이터가 있는 경로.
        faq_data = '/app/api/utils/preprocess_final_data.pkl'
        data_handle = MilvusHandle()
        
        df = pd.read_pickle(faq_data)
        print(f"df : {df}")
        data_handle.insert_FAQ(df)
        return "success"
    except Exception as e:
        print(f"Error occurred: {e}")
        return f"Error: {e}"
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
# ==================================== stream 구현 함수 테스트 ============================================== #    
# import os, re, json
# import asyncio
# import sys
# OPENAI_KEY = os.environ['OPENAI_API_KEY']

# client = OpenAI(api_key=OPENAI_KEY)
# @app.post("/stream")
# async def stream_response(query: str):
#     db_handle = DataHandle()
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
