from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
import uuid
from typing import List, Any
import pymupdf4llm
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

print("Kütüphaneler başarıyla yüklendi.")


# ==========================================
# METADATA
# ==========================================
def get_pdf_metadata(filename: str) -> dict:
    name = filename.lower()
    p_type = "pediatric" if any(x in name for x in ["child", "pediatric", "cocuk"]) else "adult"
    if "type 1" in name or "type1" in name.replace(" ", ""):
        d_type = "type1"
    elif "type 2" in name or "type2" in name.replace(" ", ""):
        d_type = "type2"
    else:
        d_type = "general"
    if any(x in name for x in ["keto", "hyperglycemia", "emergency", "acute"]):
        category, urgency = "emergency_acute", "high"
    elif any(x in name for x in ["diagnosis", "classification", "tani"]):
        category, urgency = "diagnosis_criteria", "normal"
    else:
        category, urgency = "treatment_management", "normal"
    language = "tr" if any(x in name for x in ["komplikasyon", "kilavuz", "tedavi", "tani"]) else "en"
    return {
        "patient_type": p_type,
        "diabetes_type": d_type,
        "category": category,
        "urgency_level": urgency,
        "language": language,
    }


# ==========================================
# PDF → MARKDOWN
# ==========================================
def convert_pdfs_to_markdown(source_dir: str, output_dir: str = "./data_processed", pages: list = None) -> list[Path]:
    src = Path(source_dir)
    out = Path(output_dir)
    out.mkdir(exist_ok=True)
    pdf_files = sorted(src.glob("*.pdf"))
    created = []
    print(f"\n📁 Kaynak: {src.resolve()} — {len(pdf_files)} PDF")
    for pdf_file in pdf_files:
        output_file = out / (pdf_file.stem + ".md")
        if output_file.exists():
            print(f"  ✓ Zaten mevcut, atlanıyor: {output_file.name}")
            created.append(output_file)
            continue
        print(f"  ⚙  Dönüştürülüyor: {pdf_file.name} ...", end="", flush=True)
        try:
            md_text = pymupdf4llm.to_markdown(str(pdf_file), pages=pages) if pages else pymupdf4llm.to_markdown(str(pdf_file))
            meta = get_pdf_metadata(pdf_file.name)
            frontmatter = (
                f"---\nsource: {pdf_file.name}\npatient_type: {meta['patient_type']}\n"
                f"diabetes_type: {meta['diabetes_type']}\ncategory: {meta['category']}\n"
                f"urgency_level: {meta['urgency_level']}\nlanguage: {meta['language']}\n---\n\n"
            )
            output_file.write_text(frontmatter + md_text, encoding="utf-8")
            created.append(output_file)
            print(f" {len(md_text):,} karakter → {output_file.name}")
        except Exception as e:
            print(f" HATA: {e}")
    print(f"Toplam {len(created)} dosya hazır.")
    return created


# ==========================================
# MARKDOWN YÜKLE & CHUNK
# ==========================================
def parse_frontmatter(text: str) -> tuple[dict, str]:
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
    docs = []
    for md_file in sorted(Path(processed_dir).glob("*.md")):
        raw = md_file.read_text(encoding="utf-8")
        meta, body = parse_frontmatter(raw)
        meta["file"] = md_file.name
        docs.append(Document(page_content=body, metadata=meta))
    print(f"Toplam {len(docs)} doküman yüklendi.")
    return docs


def create_chunks(docs: list[Document], chunk_size: int = 1000, chunk_overlap: int = 150) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n## ", "\n### ", "\n#### ", "\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"{len(docs)} doküman → {len(chunks)} chunk")
    return chunks


# ==========================================
# EMBEDDING
# ==========================================
class Embedding:
    DEFAULT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name_or_path=self.model_name)

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True,
            batch_size=64,
        )
        print(f"Generated embeddings with shape: {embeddings.shape}")
        return embeddings


# ==========================================
# VECTOR STORE
# ==========================================
class VectorStore:
    def __init__(self, collection_name: str = "pdf_vector_multilingual", persist_directory: str = "./vector_database/vector_store"):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "PDF document embeddings for RAG"},
        )
        print(f"Vektör deposu hazır. Mevcut doküman sayısı: {self.collection.count()}")

    def add_documents(self, documents: List[Any], embeddings: np.ndarray):
        ids = [f"doc_{uuid.uuid4().hex[:8]}_{i}" for i in range(len(documents))]
        metadatas = []
        for i, doc in enumerate(documents):
            meta = dict(doc.metadata)
            meta["doc_index"] = i
            meta["content_length"] = len(doc.page_content)
            metadatas.append(meta)
        self.collection.add(
            ids=ids,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
            documents=[doc.page_content for doc in documents],
        )
        print(f"✓ {len(documents)} doküman vektör deposuna eklendi. Toplam: {self.collection.count()}")

    def query(self, query_embedding: np.ndarray, n_results: int = 5, where: dict = None) -> dict:
        kwargs = {
            "query_embeddings": [query_embedding.tolist()],
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where
        return self.collection.query(**kwargs)


# ==========================================
# BAŞLATMA — sadece bir kez çalışır
# ==========================================
embedding_manager = Embedding()
vectorstore = VectorStore()

if vectorstore.collection.count() == 0:
    print("\n⚙️  Vektör deposu boş, ilk kez işleniyor...")
    convert_pdfs_to_markdown(source_dir="./data", output_dir="./data_processed", pages=None)
    convert_pdfs_to_markdown(source_dir="./data2_images", output_dir="./data_processed", pages=list(range(16, 191)))
    documents = load_markdown_documents("./data_processed")
    chunks = create_chunks(documents)
    texts_to_embed = [c.page_content for c in chunks]
    embedded_texts = embedding_manager.generate_embeddings(texts_to_embed)
    vectorstore.add_documents(chunks, embedded_texts)
    print("✅ Vektör deposu hazırlandı ve kaydedildi.")
else:
    print(f"✅ Vektör deposu zaten hazır ({vectorstore.collection.count()} doküman). Atlıyor.")
