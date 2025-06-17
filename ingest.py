from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models
import os
import textwrap

# Carrega o modelo de embeddings BGE-M3 (dimensão 1024)
model = SentenceTransformer("BAAI/bge-m3")
qdrant = QdrantClient("localhost", port=6333)

collection_name = "rpg_lore_bge_m3"

# Cria a coleção se ela ainda não existir
if not qdrant.collection_exists(collection_name):
    qdrant.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=1024,
            distance=models.Distance.COSINE
        )
    )

# Quebra textos longos em pedaços menores (aproximadamente 600 caracteres)
def split_text(text, max_len=600):
    # Garante que não quebra no meio das palavras
    return textwrap.wrap(text, max_len, break_long_words=False, break_on_hyphens=False)

# Inicia contador
point_id = 1

# Lê todos os arquivos .txt dentro de docs/
for file_name in os.listdir("docs"):
    if not file_name.endswith(".txt"):
        continue

    file_path = os.path.join("docs", file_name)
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()
        chunks = split_text(content)

        for chunk in chunks:
            # Prefixo obrigatório do bge-m3
            prompt_chunk = "Representacao para recuperacao: " + chunk
            vector = model.encode(prompt_chunk).tolist()

            qdrant.upsert(
                collection_name=collection_name,
                points=[
                    models.PointStruct(
                        id=point_id,
                        vector=vector,
                        payload={"text": chunk, "source": file_name}
                    )
                ]
            )
            point_id += 1

print("✅ Ingestão finalizada com sucesso.")
