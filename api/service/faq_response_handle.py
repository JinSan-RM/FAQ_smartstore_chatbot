from openai import OpenAI
from pymilvus import connections, Collection
import os, re

# Docker compose 파일의 environments에 적힌 openai api key 불러오기
OPENAI_KEY = os.environ['OPENAI_API_KEY']

client = OpenAI()

# ==================================
#         데이터 핸들링 모듈
# ==================================

class DataHandle():
    
    def __init__(self):
            self.client = OpenAI(api_key=OPENAI_KEY)
            
            connections.connect(host='milvus', port='19530') 
            
            self.collection = Collection(name="faq")
            self.collection.load()

    # embedding 함수
    def text_embedding(self, question: str):
        print(f"질문 임베딩 생성: {question}")
        response = client.embeddings.create(
            input=question,
            model="text-embedding-3-small"
        )
        # print(f"생성된 임베딩: {response.data[0].embedding}")
        return response.data[0].embedding

    # collection에 필드들에 값이 제대로 들어가 있나 확인 함수
    def check_stored_data(self):
        res = self.collection.query(
            expr="id >= 0",  
            output_fields=["question", "answer"],
            limit=5  
        )
        print("Stored data samples:", res)
    
    # RAG 방식의 문서 검색 방식 collection에서 embedding 값을 기준으로 search 및 반환
    def search_FAQ(self, query, limit=5, threshold=0.5):
        # cosine vector 비교 방식으로 조회
        results = self.collection.search(
            data=[self.text_embedding(query)],
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"nprobe": 10}},
            limit=limit,
            expr=None,
            output_fields=["question", "answer"]
        )

        if not results or not results[0]:
            # 아무 결과 없을 때 처리
            return {
                "question": "해당 질문에 대한 답변을 찾을 수 없습니다.",
                "answer": ""
            }

        top_5_scores = [hit.score for hit in results[0]]
        print(f"top_5_scores : {top_5_scores}")
        # 벡터 질문과 vector DB의 질문들과 일치하는지 체크 후 가장 유사한거 뽑고 없으면 무관 질문 반환
        if all(s < threshold for s in top_5_scores):
            # 스마트스토어 무관 질문
            return {
                "question": "저는 스마트 스토어 FAQ를 위한 챗봇입니다. 스마트 스토어에 대한 질문을 부탁드립니다.",
                "answer": ""
            }

        # FAQ 파트 합치기
        # 검색결과 그룹화
        grouped_results = {}
        for hit in results[0]:
            question = hit.entity.get('question')
            answer = hit.entity.get('answer')

            if question:
                base_question = question.split(' (파트')[0]
                grouped_results.setdefault(base_question, []).append((question, answer))
            else:
                print("질문에 답이 없음")
        # 그룹별로 결과 정렬 및 병합합
        final_results = []
        for question, parts in grouped_results.items():
            # 파트 번호를 기준으로 정렬
            sorted_parts = sorted(
                parts,
                key=lambda x: int(re.search(r'\(파트 (\d+)\)', x[0]).group(1))
                            if re.search(r'\(파트 (\d+)\)', x[0]) else 0
            )
            merged_answer = ' '.join(part[1] for part in sorted_parts)
            final_results.append({
                'question': question,
                'answer': merged_answer
            })

        if final_results:
            # 우선 가장 첫 번째(가장 높은 스코어) 결과를 반환
            return final_results[0]
        else:
            return {
                "question": "해당 질문에 대한 답변을 찾을 수 없습니다.",
                "answer": ""
            }
            
            
            
            
# =======================================
#             응답 생성 모듈
# =======================================
class ResponseHandle():
    # 정상 응답 생성 및 출력 데이터 ( stream )
    def handle_normal_response(self, request, retrieved_context, user_history, history):
        text_buffer = []
        
        # 정상 FAQ 응답 생성
        for content in self.generate_response(
                query=request.query,  # 질문
                retrieved_context=retrieved_context, # RAG 검색 데이터
                user_history=user_history # 유저 히스토리
            ):
            # 최종 응답을 위한 데이터 모으기
            text_buffer.append(content)
            print(content, flush=True)
            yield f"data: {content}\n\n"
            
        # SSE 스트리밍 종료 및 최종 응답답 구성
        final_text = ''.join(text_buffer)

        
        result = f"유저 : {request.query}\n챗봇 : {final_text}"
        history.add_message(request.user_id, "assistant", final_text) # 히스토리 업데이트
        print(result, flush=True)
        
        yield f"{result}\n\n"

    
    # 부정 응답 생성 및 출력 데이터 ( stream )
    def handle_error_response(self, request, retrieved_context, history):
        text_buffer = []
        content = retrieved_context["question"]  # "저는 스마트 스토어 FAQ를..."
        
        # 무관 질문에 대한 오류 응답 생성
        for content_add_answer in self.generate_error_response(
                question=request.query, # 질문
                content=content,  # 무관한 데이터임을 알려주는 답
            ):
            print(content_add_answer, flush=True)
            text_buffer.append(content_add_answer)
            yield f"data: {content_add_answer}\n\n"
        
        final_text = ''.join(text_buffer)
        result = f"유저 : {request.query}\n챗봇 : {final_text}" 
        history.add_message(request.user_id, "assistant", final_text) # 히스토리 업데이트트
        print(result, flush=True)
        
        yield f"{result}\n\n"

        
        
    def generate_response(self, query: str, retrieved_context: dict, user_history):
        # 이전 코드에 기반한 로직 재작성
        if user_history and user_history[-2]['content'] != query:
            last_user_question = user_history[-2]["content"] if len(user_history) >= 2 else ""
            last_assistant_answer = user_history[-1]["content"] if len(user_history) >= 1 else ""
            cleaned_answer = re.sub(r'(?m)^\s*-\s.*(?:\n)?', '', last_assistant_answer)
            prompt = f"""
            다음은 네이버 스마트스토어 FAQ 내용입니다:
            질문: {retrieved_context['question']}
            답변: {retrieved_context['answer']}

            이전 질문: {last_user_question}
            이전 답변: {cleaned_answer}

            현재 사용자 질문: {query}

            위 FAQ 내용을 참고하여 사용자의 질문에 답변해주세요.
            FAQ의 정확한 정보만 포함해야 합니다.
            이전 질문으로 대화의 맥락을 이어주시면 됩니다.
            이 정보를 활용하여:

            1) 사용자 질문에 FAQ 내용을 충실히 반영한 답변을 작성해 주세요.
            2) 답변이 끝난 후, 스마트스토어 관련해서 추가로 궁금해할 만한 1~2개 질문을 꼭 포함해 주세요.
            """
        else:
            prompt = f"""
            다음은 네이버 스마트스토어 FAQ 내용입니다:
            질문: {retrieved_context['question']}
            답변: {retrieved_context['answer']}

            사용자 질문: {query}

            위 FAQ 내용을 참고하여 사용자의 질문에 답변해주세요.
            FAQ의 정확한 정보만 포함해야 합니다.
            이 정보를 활용하여:

            1) 사용자 질문에 FAQ 내용을 충실히 반영한 답변을 작성해 주세요.
            2) 답변이 끝난 후, 스마트스토어 관련해서 추가로 궁금해할 만한 1~2개 질문을 꼭 포함해 주세요.
            """

        try:
            print(f"prompt : {prompt}", flush=True)

            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "당신은 네이버 스마트스토어 가입/서류 안내 전문가입니다. "
                            "답변을 토대로 대답해주세요."
                            "FAQ에 나온 내용으로만 답변을 구성하시고, "
                            "추가로 궁금해할 만한 1~2개 질문을 제안해주세요."
                        )
                    },
                    {
                        "role": "user",
                        "content": prompt
                    },
                    {
                        "role": "assistant",
                        "content": (
                            "아래 형식을 지켜 주세요.\n\n"
                            """
                            '[답변이 들어가야할 자리.]'\n"
                            ' - [추가로 궁금해할 만한 질문(1~2개)]\n'
                            ' - [추가로 궁금해할 만한 질문(1~2개)]'
                            """
                            "답변 시 FAQ 정보를 반드시 반영하되, FAQ에 언급되지 않은 내용은 섣불리 답하지 말아주세요."
                            " [답변이 들어가야할 자리.]에는 다른 첨언 이나 기호 표시 없이 답변만 생성하세요."
                            "추가로 궁금해할 만한 질문은 '-' 로만 시작해주세요."
                            "추가로 궁금해할 만한 질문에는 '-'를 제외하고는 다른 어떤 첨언이나 형식도 붙이지 마세요."
                            "스마트 스토어에 관련된 내용이 없다고 하지말고 질문에 대한 답변으로 정보를 전달해줘."
                            
                        )
                    }
                ],
                stream=True
            )

            for chunk in response:
                content = chunk.choices[0].delta.content
                if content:
                    yield content

        except Exception as e:
            print(f"Error in generate_response: {str(e)}")
            yield f"Error: {str(e)}"
            
    def generate_error_response(self, question: str, content: str):
        system_prompt = """
        당신은 네이버 스마트스토어 FAQ 챗봇입니다.
        사용자가 스마트스토어와 관련 없는 질문을 했을 때 다음을 수행하세요:
        1. 먼저 **"저는 스마트 스토어 FAQ를 위한 챗봇입니다. 스마트스토어에 대한 질문을 부탁드립니다."**라고 안내합니다.
        2. 그 후, 사용자의 질문에서 추론 가능한 스마트스토어 관련 후속 질문 하나만 제안하세요.
        출력은 하나의 간단한 문장, 즉 하나의 후속 질문으로 제한합니다.
        예시 출력: "- 음식도 스토어 등록이 가능한지 궁금하신가요?"
        """
        user_prompt = f"""
        사용자의 질문: {question}

        FAQ:
        {content}

        질문을 토대로 스마트스토어 후속 안내를 작성해 주세요.
        """

        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": (
                        "아래 형식을 지켜 주세요.\n\n"
                        "'저는 스마트스토어 FAQ를 위한 챗봇입니다. 스마트스토어에 대한 질문을 부탁드립니다.'\n"
                        "' - 질문을 토대로 스마트스토어 후속 안내(1~2개)'\n"
                        "질문을 토대로 스마트스토어 후속 안내 '-' 로만 시작해주세요."
                        "질문을 토대로 스마트스토어 후속 안내에는 '-'를 제외하고는 다른 어떤 첨언이나 형식도 붙이지 마세요."
                    )}
                ],
                stream=True
            )

            for chunk in response:
                content = chunk.choices[0].delta.content
                if content:
                    yield content

        except Exception as e:
            return f"{e}"


    
                



    
