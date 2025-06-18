from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models
import torch
import os
import textwrap

# Carrega o modelo com suporte ao código remoto
model = SentenceTransformer(
    "BAAI/bge-m3",
    trust_remote_code=True
)


# Move para GPU se disponível
if torch.cuda.is_available():
    model = model.to("cuda")
    print("✅ Rodando na GPU")
else:
    print("⚠️  Rodando na CPU")

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
    return textwrap.wrap(text, max_len, break_long_words=False, break_on_hyphens=False)

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
