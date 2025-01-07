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
    
    def create_embedding(self, text):
            response = self.client.embeddings.create(
                input=text,
                model="text-embedding-ada-002"
            )
            return response.data[0].embedding

    def get_relevant_context(self, query, top_k=3):
        query_embedding = self.create_embedding(query)
        search_params = {
            "metric_type": "COSINE",
            "params": {"nprobe": 16}
        }
        
        results = self.collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["question", "answer"]
        )
        
        contexts = []
        for hits in results:
            print(f"Found {len(hits)} hits")
            for hit in hits:
                print(f"Score: {hit.score}, Question: {hit.entity.get('question')}")
                contexts.append({
                    "question": hit.entity.get('question'),
                    "answer": hit.entity.get('answer'),
                    "score": hit.score
                })
        
        return contexts
    def generate_response(self, query: str, retrieved_context: dict):
        print(f"generate_response 시작")
        prompt = f"""다음은 네이버 스마트스토어 FAQ 내용입니다:
        질문: {retrieved_context['question']}
        답변: {retrieved_context['answer']}
        
        사용자 질문: {query}
        
        위 FAQ 내용을 참고하여 사용자의 질문에 답변해주세요. 
        답변은 자연스러운 대화체로 작성하되, FAQ의 정확한 정보만 포함해야 합니다."""

        try:
            print(f"여기까지는 실행")
            # OpenAI API 호출
            response = client.chat.completions.create(
                model="gpt-4o",  # 모델명
                messages=[
                    {"role": "system", "content": "당신은 네이버 스마트스토어 고객상담 전문가입니다."},
                    {"role": "user", "content": prompt}
                ],
                stream=True 
            )
            print("response는 끝.")
            
            for chunk in response:
                content = chunk.choices[0].delta.content
                if content:
                    yield content

        except Exception as e:
            print(f"Error in generate_response: {str(e)}")
            yield f"Error: {str(e)}"
            
    async def generate_error_response(error_message: str):
        yield f"data: {json.dumps({'error': error_message})}\n\n"
        yield "data: [DONE]\n\n"
        
    # def generate_response(self, query, chat_history=None):

    #     relevant_contexts = self.get_relevant_context(query)
        

    #     system_prompt = """스마트스토어 FAQ 챗봇으로서 답변해주세요. 
    #     제공된 FAQ 컨텍스트를 기반으로 답변하되, 스마트스토어와 관련 없는 질문에는 
    #     "저는 스마트 스토어 FAQ를 위한 챗봇입니다. 스마트 스토어에 대한 질문을 부탁드립니다."라고 답변하세요."""
        
    #     context_str = "\n\n".join([
    #         f"Q: {ctx['question']}\nA: {ctx['answer']}" 
    #         for ctx in relevant_contexts
    #     ])
        
    #     messages = [
    #         {"role": "system", "content": system_prompt},
    #         {"role": "system", "content": f"FAQ 컨텍스트:\n{context_str}"}
    #     ]
        
    #     if chat_history:
    #         messages.extend(chat_history)
            
    #     messages.append({"role": "user", "content": query})
        
    #     response = self.client.chat.completions.create(
    #         model="gpt-4",
    #         messages=messages,
    #         temperature=0.7,
    #         stream=True 
    #     )
        
    #     return response

    def generate_follow_up_questions(self, query, answer):
        prompt = f"""사용자의 질문과 답변을 바탕으로, 
        사용자가 추가로 궁금해할 만한 스마트스토어 관련 질문을 1-2개 생성해주세요.
        
        사용자 질문: {query}
        답변: {answer}
        """
        
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "스마트스토어 관련 후속 질문을 생성하는 어시스턴트입니다."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        
        return response.choices[0].message.content
    
    def check_stored_data(self):
        res = self.collection.query(
            expr="id >= 0",  
            output_fields=["question", "answer"],
            limit=5  
        )
        print("Stored data samples:", res)
        
    def search_FAQ(self, query, limit=5):
        results = self.collection.search(
            data=[self.text_embedding(query)],
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"nprobe": 10}}, 
            limit=limit,
            expr=None,
            output_fields=["question", "answer"] 
        )

        grouped_results = {}
        for hit in results[0]:
            question = hit.entity.get('question')
            answer = hit.entity.get('answer')
            
            if question:
                base_question = question.split(' (파트')[0]
                
                if base_question not in grouped_results:
                    grouped_results[base_question] = []
                grouped_results[base_question].append((question, answer))
            else:
                print("질문에 답이 없음")
        
        final_results = []
        for question, parts in grouped_results.items():
            sorted_parts = sorted(
                parts,
                key=lambda x: int(re.search(r'\(파트 (\d+)\)', x[0]).group(1)) if re.search(r'\(파트 (\d+)\)', x[0]) else 0
            )
            
            merged_answer = ' '.join(part[1] for part in sorted_parts)
            final_results.append({
                'question': question,
                'answer': merged_answer
            })
        
        if final_results:
            return final_results[0]
        else:
            return {"question": "해당 질문에 대한 답변을 찾을 수 없습니다.", "answer": ""}
            
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



    
