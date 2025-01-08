from collections import defaultdict

class ChatContext:
    def __init__(self):
        self.history = defaultdict(list)
    
    def add_message(self, user_id, role, content):
        self.history[user_id].append({"role": role, "content": content})
    
    def get_history(self, user_id):
        return self.history[user_id]
    
    def clear_history(self, user_id):
        self.history[user_id] = []
        