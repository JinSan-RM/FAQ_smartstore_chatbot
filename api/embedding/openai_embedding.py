from openai import OpenAI

client = OpenAI()


def embed_data(data : dict):
    
    preprocessing_embeddings = {}
    
    for q, a in data.items():
        json_to_embed = f""
    response = client.embeddings.create(
        input="",
        model="text-embedding-3-small"
    )