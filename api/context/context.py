from collections import defaultdict, deque

class ChatContext:
    
    def __init__(self, history_store):
        self.history_store = history_store
    def add_message(self, user_id, role, content):
        self.history_store[user_id].append({"role": role, "content": content})
    
    def get_user_history(self, user_id):
        return list(self.history_store[user_id])
    