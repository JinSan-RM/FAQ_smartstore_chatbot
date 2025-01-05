from fastapi import FastAPI

app = FastAPI()

@app.post('/openai_embedding')
def vector_embedding_FAQ(path : str):
    
    return path