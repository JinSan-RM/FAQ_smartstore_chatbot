from openai import OpenAI
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
import os, re, json
import asyncio

OPENAI_KEY = os.environ['OPENAI_API_KEY']

client = OpenAI()

class DBHandling():
    
    def __init__(self):
            self.client = OpenAI(api_key=OPENAI_KEY)
            
            connections.connect(host='milvus', port='19530') 
            
            self.collection = Collection(name="faq")
            self.collection.load()

    def text_embedding(self, question: str):
        print(f"질문 임베딩 생성: {question}")
        response = client.embeddings.create(
            input=question,
            model="text-embedding-3-small"
        )
        # print(f"생성된 임베딩: {response.data[0].embedding}")
        return response.data[0].embedding


    def generate_response(self, query: str, retrieved_context: dict, user_history):
        
        # retrieved_context['question']가 "저는 스마트 스토어 FAQ를..." 등의 문구일 수도 있으니,
        # 실제 FAQ인지 확인이 필요. 여기서는 그대로 사용한다고 가정.

        if user_history and user_history[-2]['content'] != query:
            # 사용자의 이전 질문과 답변을 기반으로 더 적합한 답변을 생성하기 위해
            last_user_question = user_history[-2]["content"] if len(user_history) >= 2 else ""
            last_assistant_answer = user_history[-1]["content"] if len(user_history) >= 1 else ""
            
            # 이전 대화 내용을 바탕으로 적절한 맥락을 추가할 수 있음
            prompt = f"""
            다음은 네이버 스마트스토어 FAQ 내용입니다:
            질문: {retrieved_context['question']}
            답변: {retrieved_context['answer']}

            이전 질문: {last_user_question}
            이전 답변: {last_assistant_answer}

            현재 사용자 질문: {query}

            위 FAQ 내용을 참고하여 사용자의 질문에 답변해주세요.
            FAQ의 정확한 정보만 포함해야 합니다.
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

            # OpenAI API 호출
            response = client.chat.completions.create(
                model="gpt-4o",  # 모델명
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "당신은 네이버 스마트스토어 가입/서류 안내 전문가입니다. "
                            "사용자 질문이 스마트스토어와 무관하다면, "
                            "'저는 스마트 스토어 FAQ를 위한 챗봇입니다. 스마트 스토어에 대한 질문을 부탁드립니다.' "
                            "라고만 답변하세요. "
                            "위 FAQ에 나온 내용으로만 답변을 구성하시고, "
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
                            '답변이 들어가야할 자리.'\n"
                            ' - 추가로 궁금해할 만한 질문(1~2개)\n'
                            ' - 추가로 궁금해할 만한 질문(1~2개)'"""
                            "답변 시 FAQ 정보를 반드시 반영하되, "
                            "FAQ에 언급되지 않은 내용은 섣불리 답하지 말아주세요."
                            "추가로 궁금해할 만한 질문은 '-' 로만 시작해주세요."
                            "추가로 궁금해할 만한 질문에는 '-'를 제외하고는 다른 어떤 첨언이나 형식도 붙이지 마세요."
                            
                        )
                    }
                ],
                stream=True
            )
            
            # SSE로 스트리밍 (yield)
            for chunk in response:
                content = chunk.choices[0].delta.content
                if content:
                    yield content

        except Exception as e:
            print(f"Error in generate_response: {str(e)}")
            yield f"Error: {str(e)}"
            
    def generate_error_response(self, question: str, content: str, user_history):
    
        # 저는 스마트 스토어 FAQ를 위한 챗봇입니다. 스마트 스토어에 대한 질문을 부탁드립니다. RAG 단계에서 데이터를 생성.
        # 대답은 굳이 생성 안해도 되는 비용 절감의 가능성이 있음.

        system_prompt = """
        당신은 네이버 스마트스토어 FAQ 챗봇입니다.
        사용자가 스마트스토어와 관련 없는 질문을 했을 때 다음을 수행하세요:
        1. 먼저 **"저는 스마트 스토어 FAQ를 위한 챗봇입니다. 스마트 스토어에 대한 질문을 부탁드립니다."**라고 안내합니다.
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
            # 4) LLM 호출 (아래는 예시로 non-stream 모드)
            response = client.chat.completions.create(
                model="gpt-4o",  # 모델명
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant",  "content": (
                            "아래 형식을 지켜 주세요.\n\n"
                            """
                            '저는 스마트 스토어 FAQ를 위한 챗봇입니다. 스마트 스토어에 대한 질문을 부탁드립니다.'\n"
                            ' - 질문을 토대로 스마트스토어 후속 안내(1~2개)\n'"""
                            " 질문을 토대로 스마트스토어 후속 안내 '-' 로만 시작해주세요."
                            " 질문을 토대로 스마트스토어 후속 안내에는 '-'를 제외하고는 다른 어떤 첨언이나 형식도 붙이지 마세요."
                            
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
            # LLM 호출 중 에러 발생 시, 원래 error_message를 보여주고 종료
            return  f"data: [DONE]\n\n"
                
    
    def check_stored_data(self):
        res = self.collection.query(
            expr="id >= 0",  
            output_fields=["question", "answer"],
            limit=5  
        )
        print("Stored data samples:", res)
        
    def search_FAQ(self, query, limit=5, threshold=0.5):
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

        # 1) 임계값 판별
        if all(s < threshold for s in top_5_scores):
            # => 스마트스토어 무관 질문
            return {
                "question": "저는 스마트 스토어 FAQ를 위한 챗봇입니다. 스마트 스토어에 대한 질문을 부탁드립니다.",
                "answer": ""
            }

        # 2) FAQ 파트 합치기
        grouped_results = {}
        for hit in results[0]:
            question = hit.entity.get('question')
            answer = hit.entity.get('answer')

            if question:
                base_question = question.split(' (파트')[0]
                grouped_results.setdefault(base_question, []).append((question, answer))
            else:
                print("질문에 답이 없음")

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
                
    # def search_FAQ(self, query, limit=5):
    #     import re  # re 모듈이 필요합니다.

    #     results = self.collection.search(
    #         data=[self.text_embedding(query)],
    #         anns_field="embedding",
    #         param={"metric_type": "COSINE", "params": {"nprobe": 10}}, 
    #         limit=limit,
    #         expr=None,
    #         output_fields=["question", "answer"]  # 추가된 부분
    #     )

    #     grouped_results = {}
    #     for hit in results[0]:
    #         question = hit.entity.get('question')
    #         answer = hit.entity.get('answer')
            
    #         if question:
    #             base_question = question.split(' (파트')[0]
                
    #             if base_question not in grouped_results:
    #                 grouped_results[base_question] = []
    #             grouped_results[base_question].append((question, answer))
    #         else:
    #             print("질문에 답이 없음")
        
    #     final_results = []
    #     for question, parts in grouped_results.items():
    #         sorted_parts = sorted(
    #             parts,
    #             key=lambda x: int(re.search(r'\(파트 (\d+)\)', x[0]).group(1)) if re.search(r'\(파트 (\d+)\)', x[0]) else 0
    #         )
            
    #         merged_answer = ' '.join(part[1] for part in sorted_parts)
    #         final_results.append({
    #             'question': question,
    #             'answer': merged_answer
    #         })
        
    #     return final_results



    
