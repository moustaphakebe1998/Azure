import os
import requests
import json
from PyPDF2 import PdfReader
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from transformers import GPT2TokenizerFast

# Configuration Azure Cognitive Search
search_service_endpoint = "https://ia-searchkeb...search.windows.net"  # URL de votre service Azure Cognitive Search
search_index_name = "documents-index"  # Nom de l'index Azure Cognitive Search
admin_key = "fok2jLVK4MdHuem78RTDd5jmWMaR...Axg7svkAzSeAho9Dy"  # Clé API administrateur pour Azure Cognitive Search

# Configurations pour l'API d'Embedding
embedding_endpoint = "https://ai-moustaphakebea....openai.azure.com/openai/deployments/text-embedding-3-large/embeddings?api-version=2023-05-15"  # URL de l'API d'Embedding
embedding_key = "6121701270b94d179ee4a3....6b9d"  # Clé API pour l'API d'Embedding

# Headers pour les requêtes API
headers_embedding = {
    "Content-Type": "application/json",
    "api-key": embedding_key
}

search_client = SearchClient(
    endpoint=search_service_endpoint,
    index_name=search_index_name,
    credential=AzureKeyCredential(admin_key)
)

# Initialisez le tokenizer
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

def split_text_into_chunks(text, max_tokens=1024):
    tokens = tokenizer.encode(text)
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i:i + max_tokens]
        chunk_text = tokenizer.decode(chunk_tokens, clean_up_tokenization_spaces=True)
        print(f"Chunk length: {len(chunk_tokens)} tokens")  # Debug line
        chunks.append(chunk_text)
    return chunks


# Fonction pour lire le contenu d'un PDF
def read_pdfs_from_directory(directory):
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            file_path = os.path.join(directory, filename)
            pdf_reader = PdfReader(file_path)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            documents.append({
                "id": filename,
                "content": text
            })
    return documents

# Fonction pour obtenir les embeddings d'un texte
def get_embeddings(text):
    response = requests.post(
        embedding_endpoint,
        headers=headers_embedding,
        json={"input": text}
    )
    return response.json()["data"][0]["embedding"]

# Fonction pour indexer les documents dans Azure Cognitive Search
def index_documents(documents):
    for doc in documents:
        chunks = split_text_into_chunks(doc["content"])
        for i, chunk in enumerate(chunks):
            embedding = get_embeddings(chunk)
            try:
                search_client.upload_documents(documents=[{
                    "id": f"{doc['id']}_{i}",
                    "content": chunk,
                    "embedding": embedding
                }])
            except Exception as e:
                print(f"Erreur lors de l'indexation du document {doc['id']}_{i}: {e}")

# Exemple d'utilisation
directory = "/Users/moustaphakebe1998gmail.com/Azure/data/filerag"
documents = read_pdfs_from_directory(directory)
index_documents(documents)
