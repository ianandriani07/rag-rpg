# 🧠 RAG-RPG: IA que responde perguntas sobre o meu mundo de fantasia

Este projeto usa **RAG (Retrieval-Augmented Generation)** para responder perguntas sobre o universo fictício Exandria RPG, utilizando arquivos `.txt` com a lore completa. Tudo roda **localmente**, sem depender de APIs externas.

---

## 🌍 Sobre o universo

Os textos carregados descrevem o mundo de Exandria RPG, suas cidades, reinos, conflitos históricos, organizações mágicas e figuras lendárias como **Caroline Windspur**, a **Assembleia do Cérbero**, os eventos do **Ano das Portas Abertas**, entre outros.

---

## ⚙️ Como funciona

O projeto é um pipeline local que usa:

- **[BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3)** – para gerar embeddings semânticos dos textos
- **[Qdrant](https://qdrant.tech/)** – banco vetorial para armazenar e buscar vetores
- **[BAAI/bge-reranker-base](https://huggingface.co/BAAI/bge-reranker-base)** – reranker para ordenar trechos por relevância
- **`gemma3:4b-it-qat`** via [Ollama](https://ollama.com) – modelo de linguagem que gera a resposta final com base no contexto
- Tudo feito com **Python**, **Docker**, e aceleração via **GPU**, se disponível

---

## 🚀 Como rodar o projeto

### 1. Pré-requisitos

- Python 3.12
- [Docker + Docker Compose](https://docs.docker.com/get-docker/)
- [Ollama](https://ollama.com) instalado
- Git

### 2. Clone o projeto

```bash
git clone https://github.com/ianandriani07/rag-rpg.git
cd rag-rpg
```

### 3. Crie um ambiente virtual

```bash
python -m venv .venv
.\.venv\Scripts ctivate  # no Windows
```

### 4. Instale as dependências

```bash
pip install -r requirements.txt
```

### 5. Suba o Qdrant com Docker

```bash
docker-compose up -d
```

### 6. Baixe o modelo do Ollama

```bash
ollama pull gemma3:4b-it-qat
```

### 7. Adicione seus arquivos de lore em `docs/`

Coloque seus arquivos `.txt` com a lore do seu mundo dentro da pasta `docs/`.

### 8. Rode a ingestão (indexação dos textos)

```bash
python ingest.py
```

### 9. Pergunte sobre o mundo

```bash
python query.py
```

Digite perguntas como:

- Quem foi Gots e o que aconteceu com Toya?
- O que foi o Ano das Portas Abertas?
- Qual é a origem da Assembleia do Cérbero?

Para sair, digite `sair`.

---

## ✨ Exemplo de uso

```bash
Digite sua pergunta sobre o mundo: O que foi o Ano das Portas Abertas?

🧠 Resposta da IA:
O Ano das Portas Abertas é um evento celebrado como um exemplo da bondade e união entre os humanoides,
que se originou durante os conflitos...
```

---

## 🧩 Arquitetura (resumo técnico)

```
[docs/*.txt] → [embedding com bge-m3] → [armazenamento no Qdrant]
                       ↓
               [busca semântica top-k=20]
                       ↓
             [reranking com bge-reranker]
                       ↓
         [top 5 trechos] → [prompt para gemma3]
                       ↓
                 🧠 Resposta gerada!
```

