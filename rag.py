# %% [markdown]
# # RAG Pipeline — Medikal Diyabet Dokümanları
# %% [markdown]
# ## Kurulum
# %% [markdown]
# ## Adım 1 — Kütüphaneler
# %%

from pathlib import Path

import pymupdf4llm
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

print("Kütüphaneler başarıyla yüklendi.")
# %% [markdown]
# ## Adım 2 — PDF → Markdown Dönüşümü
# 
# Bu adım her PDF'i `data_processed/` klasörüne `.md` dosyası olarak kaydeder.  
# Sonraki çalıştırmalarda mevcut dosyalar atlanır (yeniden işlenmez).
# %%
def get_pdf_metadata(filename: str) -> dict:
    """Dosya adından otomatik metadata çıkarır."""
    name = filename.lower()

    # Hasta tipi
    p_type = "pediatric" if any(x in name for x in ["child", "pediatric", "cocuk"]) else "adult"

    # Diyabet tipi
    if "type 1" in name or "type1" in name.replace(" ", ""):
        d_type = "type1"
    elif "type 2" in name or "type2" in name.replace(" ", ""):
        d_type = "type2"
    else:
        d_type = "general"

    # Kategori & aciliyet
    if any(x in name for x in ["keto", "hyperglycemia", "emergency", "acute"]):
        category, urgency = "emergency_acute", "high"
    elif any(x in name for x in ["diagnosis", "classification", "tani"]):
        category, urgency = "diagnosis_criteria", "normal"
    else:
        category, urgency = "treatment_management", "normal"

    # Dil
    language = "tr" if any(x in name for x in ["komplikasyon", "kilavuz", "tedavi", "tani"]) else "en"

    return {
        "patient_type": p_type,
        "diabetes_type": d_type,
        "category": category,
        "urgency_level": urgency,
        "language": language,
    }
# %%
def convert_pdfs_to_markdown(
    source_dir: str,
    output_dir: str = "./data_processed",
    pages: list = None,
) -> list[Path]:
    """
    source_dir içindeki tüm PDF'leri markdown'a çevirir ve output_dir'e kaydeder.
    pages: 0-indexed sayfa numaraları listesi (None = tüm sayfalar)
    """
    src = Path(source_dir)
    out = Path(output_dir)
    out.mkdir(exist_ok=True)

    pdf_files = sorted(src.glob("*.pdf"))
    created = []

    print(f"\n📁 Kaynak: {src.resolve()}")
    print(f"💾 Hedef : {out.resolve()}")
    print(f"📄 Toplam: {len(pdf_files)} PDF\n")

    for pdf_file in pdf_files:
        output_file = out / (pdf_file.stem + ".md")

        if output_file.exists():
            print(f"  ✓ Zaten mevcut, atlanıyor: {output_file.name}")
            created.append(output_file)
            continue

        print(f"  ⚙  Dönüştürülüyor: {pdf_file.name} ...", end="", flush=True)
        try:
            if pages is not None:
                md_text = pymupdf4llm.to_markdown(str(pdf_file), pages=pages)
            else:
                md_text = pymupdf4llm.to_markdown(str(pdf_file))

            meta = get_pdf_metadata(pdf_file.name)

            # YAML frontmatter başlığı — yükleme sırasında parse edilir
            frontmatter = (
                f"---\n"
                f"source: {pdf_file.name}\n"
                f"patient_type: {meta['patient_type']}\n"
                f"diabetes_type: {meta['diabetes_type']}\n"
                f"category: {meta['category']}\n"
                f"urgency_level: {meta['urgency_level']}\n"
                f"language: {meta['language']}\n"
                f"---\n\n"
            )

            output_file.write_text(frontmatter + md_text, encoding="utf-8")
            created.append(output_file)
            print(f" {len(md_text):,} karakter → {output_file.name}")

        except Exception as e:
            print(f" HATA: {e}")

    print(f"\nToplam {len(created)} dosya hazır.")
    return created
# %%
# --- İngilizce ClinicalKey PDF'leri (data/) --- #
convert_pdfs_to_markdown(
    source_dir="./data",
    output_dir="./data_processed",
    pages=None,
)
# %%
# --- Türkçe TEMD Kılavuzu (data2_images/) --- #

turkish_pages = list(range(16, 191))  

convert_pdfs_to_markdown(
    source_dir="./data2_images",
    output_dir="./data_processed",
    pages=turkish_pages,
)
# %%
# Oluşturulan dosyaları kontrol et
processed = sorted(Path("./data_processed").glob("*.md"))
print(f"data_processed/ içinde {len(processed)} markdown dosyası:")
for f in processed:
    size_kb = f.stat().st_size / 1024
    print(f"  {f.name:<70} {size_kb:6.1f} KB")
# %% [markdown]
# ## Adım 3 — Markdown Dosyalarını Yükle
# %%
def parse_frontmatter(text: str) -> tuple[dict, str]:
    """YAML frontmatter'ı parse eder, (metadata_dict, body_text) döndürür."""
    metadata = {}
    if text.startswith("---"):
        end_idx = text.find("---", 3)
        if end_idx != -1:
            fm_block = text[3:end_idx].strip()
            for line in fm_block.splitlines():
                if ":" in line:
                    key, _, value = line.partition(":")
                    metadata[key.strip()] = value.strip()
            text = text[end_idx + 3:].strip()
    return metadata, text


def load_markdown_documents(processed_dir: str = "./data_processed") -> list[Document]:
    """Tüm .md dosyalarını okur, frontmatter'dan metadata çeker."""
    docs = []
    for md_file in sorted(Path(processed_dir).glob("*.md")):
        raw = md_file.read_text(encoding="utf-8")
        meta, body = parse_frontmatter(raw)
        meta["file"] = md_file.name  # dosya adını da ekle
        docs.append(Document(page_content=body, metadata=meta))

    print(f"Toplam {len(docs)} doküman yüklendi.")
    return docs


documents = load_markdown_documents("./data_processed")

# Örnek kontrol
sample = documents[0]
print(f"\nÖrnek doküman:")
print(f"  Metadata : {sample.metadata}")
print(f"  İçerik   : {sample.page_content[:300]} ...")
# %% [markdown]
# ## Adım 4 — Chunking
# %%
def create_chunks(
    docs: list[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> list[Document]:
    """
    Markdown dokümanlarını chunk'lara böler.
    Separators: önce başlık sınırları, sonra paragraf, satır, kelime.
    Her chunk ebeveyn dokümanın metadata'sını taşır.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n## ", "\n### ", "\n#### ", "\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"{len(docs)} doküman → {len(chunks)} chunk (chunk_size={chunk_size}, overlap={chunk_overlap})")
    return chunks


chunks = create_chunks(documents)

# Chunk dağılımı
from collections import Counter
source_counts = Counter(c.metadata.get("source", "?") for c in chunks)
print("\nDosya başına chunk sayısı:")
for src, count in sorted(source_counts.items()):
    print(f"  {src:<70} {count:>4} chunk")
# %% [markdown]
# ## Adım 5 — Embedding & Vektör Store (Chroma)
# %%
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
import uuid
from typing import List, Any
from sklearn.metrics.pairwise import cosine_similarity
# %%
class Embedding:
    """Handles document embedding generation using SentenceTransformer."""

    DEFAULT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self.model = self._load_model()

    def _load_model(self) -> SentenceTransformer:
        try:
            return SentenceTransformer(model_name_or_path=self.model_name)
        except Exception as e:
            print(f"Error loading model {self.model_name}: {e}")
            raise

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True,
            batch_size=64,
        )
        print(f"Generated embeddings with shape: {embeddings.shape}")
        return embeddings

 
embedding_manager = Embedding()
embedding_manager
# %%
class VectorStore:
    """Storing vectors to ChromaDB."""

    def __init__(
        self,
        collection_name: str = "pdf_vector_multilingual",
        persist_directory: str = "./data/vector_store",
    ):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self._initialize_store()

    def _initialize_store(self):
        """Initialize ChromaDB client and collection."""
        try:
            Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "PDF document embeddings for RAG"},
            )
            print(f"Vector store initialized. Collection: {self.collection_name}")
            print(f"Existing documents in collection: {self.collection.count()}")
        except Exception as e:
            print(f"Error initializing vector store: {e}")
            raise


    def add_documents(self, documents: List[Any], embeddings: np.ndarray, skip_if_populated: bool = True):
        """Add documents and their embeddings to the vector store."""
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents must match number of embeddings")

        if skip_if_populated and self.collection.count() > 0:
            print(f"Koleksiyon zaten dolu ({self.collection.count()} doküman). Atlanıyor.")
            print("Yeniden yüklemek için: vectorstore.reset_collection() çalıştır")
            return

        print(f"Adding {len(documents)} documents to vector store...")

        ids = [f"doc_{uuid.uuid4().hex[:8]}_{i}" for i in range(len(documents))]

        metadatas = []
        for i, doc in enumerate(documents):
            meta = dict(doc.metadata)
            meta["doc_index"] = i
            meta["content_length"] = len(doc.page_content)
            metadatas.append(meta)

        try:
            self.collection.add(
                ids=ids,
                embeddings=embeddings.tolist(),
                metadatas=metadatas,
                documents=[doc.page_content for doc in documents],
            )
            print(f"Successfully added {len(documents)} documents to vector store")
            print(f"Total documents in collection: {self.collection.count()}")
        except Exception as e:
            print(f"Error adding documents to vector store: {e}")
            raise

    def query(
        self,
        query_embedding: np.ndarray,
        n_results: int = 5,
        where: dict = None,
    ) -> dict:
        """Retrieve the top-n most similar chunks."""
        kwargs = {
            "query_embeddings": [query_embedding.tolist()],
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where
        return self.collection.query(**kwargs)


vectorstore = VectorStore()
vectorstore
# %%
chunks
# %%
texts_to_embed = [c.page_content for c in chunks]
embedded_texts = embedding_manager.generate_embeddings(texts_to_embed)

stored_vectors= vectorstore.add_documents(chunks ,embedded_texts)