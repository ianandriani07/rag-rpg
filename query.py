from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import subprocess
import textwrap

# Se possuir gpu usa ela
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Carrega modelos
embedder = SentenceTransformer("BAAI/bge-m3")  # Embedding para busca semântica
reranker_tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-reranker-base")
reranker_model = AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-base").to(device)

qdrant = QdrantClient("localhost", port=6333)

def responder(pergunta):
    # Embedding da pergunta
    question_vector = embedder.encode(pergunta).tolist()

    # Busca os 20 mais relevantes
    resultados = qdrant.search(
        collection_name="rpg_lore_bge_m3",
        query_vector=question_vector,
        limit=20,
        with_payload=True
    )

    # Pares pergunta-contexto
    pares = [(pergunta, r.payload['text']) for r in resultados]

    # Tokeniza para reranking
    inputs = reranker_tokenizer.batch_encode_plus(
        pares,  # lista de tuplas (q, p)
        padding=True,
        truncation=True,
        return_tensors="pt"
    ).to(device)

    # Faz reranking com gpu se tiver
    with torch.no_grad():
        scores = reranker_model(**inputs).logits.squeeze(-1)

    # Seleciona os top 5
    top_indices = scores.topk(5).indices.tolist()
    contexto = "\n\n".join([pares[i][1] for i in top_indices])

    # Prompt final
    prompt = f"""Você é um cronista experiente e imparcial, com acesso apenas ao conteúdo abaixo.
Responda à pergunta de forma precisa e fundamentada, usando apenas as informações fornecidas.
Evite florear ou inventar elementos que não estejam no texto.

Contexto:
{contexto}

Pergunta: {pergunta}
Resposta:"""

    # Usa o modelo local via Ollama
    resposta = subprocess.run(
        ["ollama", "run", "gemma3:4b-it-qat"],
        input=prompt.encode(),
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL
    )

    print("\n🧠 Resposta da IA:")
    print(textwrap.fill(resposta.stdout.decode(), width=100)+"\n")

if __name__ == "__main__":

    while True:
        pergunta = input("Digite sua pergunta sobre o mundo: ")

        if pergunta == "sair":
            break
        responder(pergunta)
