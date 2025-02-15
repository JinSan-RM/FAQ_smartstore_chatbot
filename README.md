docker desktop을 설치

git bash 열기 (encoding)

git clone https://github.com/JinSan-RM/FAQ_smartstore_chatbot

명령어 입력
docker compose build --no-cache --progress=plain

docker compose up
(최초의 milvus 실행시간이 소요될 수 있습니다.)

openai_api_key docker-compose.yml의 service > chatbot > environment > OPENAI_API_KEY= 여기에 입력

postman 호출 or curl 통신 방식으로 진행. 

curl -X POST http://localhost:8001/openai_faq
- vectorDB에 schema 및 field 생성 후 data embedding -> insert
##### ※전체 데이터 입력부분이라 시간이 조금 걸립니다. 프린트 찍어놔서 임베딩되어 입력되는거 실시간 확인 가능.

- curl 호출 방식.
echo '{"query":"스마트스토어에서 취급 불가한 상품이 있나요?","user_id":"test"}' > data.json
curl -X POST http://localhost:8001/openai_faq_search   -H "Content-Type: application/json"   --data @data.json
    ```jsx

    최초 질문

    유저 : 스마트스토어에서 취급 불가한 상품이 있나요?
    챗봇 : 챗봇 : 스마트스토어에서 취급 불가한 상품은 여러 가지가 있습니다. 성인용품(콘돔, 성기구류, 러브젤 등), 성인용 DVD/블루레이/중고도서/음반/공연티켓, 대여 보증금을 받는 상품, 개인정보 수집이 포함된 상품, 의약품, 주류 및 담배, 마약류, 헌혈증서, 음란물, 시력보정용 안경 및 콘택트렌즈(미용렌즈 포함), 야생 동식물, 군복 및 군용장구, 총포·도검·화약류 등 관련 법령에 의해 인터넷 거래가 불가한 상품들이 포함됩니다. 휴대폰 개통을 필요로 하는 상품, 후원금/기부금 형태의 상품, 자동차 매매와 같은 특정 카테고리의 상품도 취급이 불가합니다. 스마트스토어에서의 판매가 제한되는 경우에는 네이버쇼핑의 가이드라인에 따라야 합니다.

    - 취급하는 상품이 전안법 대상인지 확인할 수 있나요?
    - 스마트스토어 이용 시 주의해야 할 거래 정책은 무엇인가요?
    echo '{"query":"취급하는 상품이 전안법 대상인지 확인할 수 있나요?","user_id":"test"}' > data.json
    curl -X POST http://localhost:8001/openai_faq_search   -H "Content-Type: application/json"   --data @data.json

    후속 질문

    유저 : 취급하는 상품이 전안법 대상인지 확인할 수 있나요?
    챗봇 : 판매하시는 상품의 KC인증대상여부는 국가기술표준원에서 발행한 가이드라인자료를 통해 판매자님께서 직접 확인해주시기 바라며, 가이드라인에 제시되어 있지 않은 추가적인 질의의 경우 국가기술표준원으로 문의해 주시기 바랍니다. 가이드라인 자료는 스마트스토어 판매자센터 공지 또는 국가기술표준원 사이트에서 확인하실 수 있습니다.

    - 스마트스토어 가입 시 요구되는 서류는 무엇인가요?
    - 네이버 스마트스토어에서 판매 가능한 제품군은 어떻게 확인할 수 있나요?
    ```

echo '{"query":"합정역 맛집에 대해서 알려줘","user_id":"test"}' > data.json
curl -X POST http://localhost:8001/openai_faq_search   -H "Content-Type: application/json"   --data @data.json
    ```jsx

    잘못된 입력 시 
    
    유저 : 합정역 맛집에 대해서 알려줘                 
    챗봇 : 저는 스마트 스토어 FAQ를 위한 챗봇입니다. 스마트 스토어에 대한 질문을 부탁드립니다.  
    - 음식도 스토어 등록이 가능한지 궁금하신가요?   
    ```


처리 순서
- RAG 방식 구현
- 스트리밍 방식 구현
- 부적절한 답변 반응 구현
- 사용자가 궁금해 할만한 추가 요소 질문
- 유저 히스토리 기록
- 히스토리 기반 맥락있는 답변

### 요구사항

- **자유로운 방식(언어, 프레임워크, …)으로, 대화가 가능한 챗봇 API를 구현**
    - FastAPI, Django, …
    - **스트리밍 방식 필수**
- **RAG(Retrieval-Augmented Generation)를 활용**
    - 제공된 FAQ 데이터(`final_result.pkl`)를 근거로 답변을 제공
    - 유저의 이전 질문과 상황 등을 토대로 더 적절한 답변을 제공
- **대화 기록을 저장함으로써 대화의 맥락을 기반으로 답변할 수 있어야 함**
- 사용자의 질문에 대해 답을 해준 뒤, 질의응답 맥락에서 사용자가 궁금해할만한 다른 내용을 물어봐야 함.
    
    ```jsx
    유저: 미성년자도 판매 회원 등록이 가능한가요?
    챗봇: 네이버 스마트스토어는 만 14세 미만의 개인(개인 사업자 포함) 또는 법인사업자는 입점이 불가함을 양해 부탁 드립니다.
    챗봇:   - 등록에 필요한 서류 안내해드릴까요?
    챗봇:   - 등록 절차는 얼마나 오래 걸리는지 안내가 필요하신가요?
    	
    ```
    
- **스마트스토어와 관련 없는 질문에는 답변하지 않아야 함**
    - 부적절한 질문에는 "저는 스마트 스토어 FAQ를 위한 챗봇입니다. 스마트 스토어에 대한 질문을 부탁드립니다."와 같이 안내 메시지를 출력합니다.
    
    ```yaml
    유저 : 오늘 저녁에 여의도 가려는데 맛집 추천좀 해줄래?
    챗봇 : 저는 스마트 스토어 FAQ를 위한 챗봇입니다. 스마트 스토어에 대한 질문을 부탁드립니다.
    챗봇:   - 음식도 스토어 등록이 가능한지 궁금하신가요?
    ```
    
- **2가지 이상의 질의응답 시나리오를 만들어 데모 내용을 정리**
    - 데모 내용은 노션 문서에 포함.
- **사용 가능한 파이썬 패키지를 확인**
    - LLM, embedding 모델은 `OpenAI`, `Huggingface`를 이용 (**요청시 OpenAI API key 제공**)
    가급적 OpenAI 를 추천
    - `Pinecone` 과 같은 SaaS 보다는 [Milvus](https://github.com/milvus-io/milvus) 혹은 [Chroma](https://github.com/chroma-core/chroma) 등의 로컬 기반 오픈소스를 사용
    - **LangChain, Llama Index 와 같은 LLM 오케스트레이션 프레임워크는 사용하지 금지!**

### 제출물

- **노션 문서**
    1. 코드 결과물
        - 코드를 정리하여 github 링크를 제공부탁드립니다.
        - 실험 재현을 위해 poetry 또는 requirements 파일을 포함해주세요!
    2. 문제에 대한 접근법 및 코드 결과물 설명 문서
    3. 2가지 이상의 질의응답 데모 (txt, png 등 자유 형식)
    4. 데모를 실제로 실행할 수 있도록 코드 실행 방법

### 평가기준

평가는 아래 요소들을 고려하여 종합적으로 이루어집니다.

1. 문제 해결 접근 방법
2. 챗봇의 답변 품질 (문맥 이해도, 적절한 답변 제공 여부)
3. 챗봇의 성능 최적화 (답변 시간, 비용, …)
4. 코드 가독성 및 구조 (모듈화, 코드 스타일 등)
5. 최종 제출 문서 품질
6. 깃허브 커밋 로그