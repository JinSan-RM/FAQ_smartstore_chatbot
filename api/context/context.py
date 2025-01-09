
# 과제 형태로 제출을 위해 DB나 외부 저장소를 사용해 불러오는 형태 보다는 
# 전역변수로 설정하여 앱 초기화시 히스토리 초기화하는 방식 채택.
# ===========================
#      히스토리 관리 모듈
# ===========================

class ChatContext:
    
    def __init__(self, history_store):
        self.history_store = history_store
        
    def add_message(self, user_id, role, content):
        self.history_store[user_id].append({"role": role, "content": content})
    
    def get_user_history(self, user_id):
        return list(self.history_store[user_id])
    