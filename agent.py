import os
from typing import List
from typing_extensions import TypedDict
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import END, StateGraph
from langchain_xai import ChatXAI  # Grok için

from rag import embedding_manager, vectorstore

# Grok API Anahtarınızı ortam değişkeni olarak ayarlayın


# Grok modelini başlatalım
llm = ChatXAI(model="grok-4.20-reasoning", temperature=0)


# ==========================================
# 1. GRAPH STATE (Ajanın Hafızası)
# ==========================================
class GraphState(TypedDict):
    """
    Ajanın işlem boyunca taşıyacağı değişkenler.
    """
    question: str
    documents: List[Document]
    generation: str


# ==========================================
# 2. NODES (Düğümler - Ajanın Aksiyonları)
# ==========================================

def retrieve(state: GraphState):
    """
    Kullanıcının sorusunu ChromaDB'de arar.
    (Daha önce oluşturduğunuz vectorstore objesini kullanıyoruz)
    """
    print("--- 🔍 RETRIEVE (DOKÜMAN GETİRİLİYOR) ---")
    question = state["question"]

    # Burada sizin yazdığınız query fonksiyonunu kullanıyoruz
    # Soru embedding'ini oluşturup arama yapıyoruz
    query_embedding = embedding_manager.generate_embeddings([question])[0]

    # En yakın 4 chunk'ı getir
    results = vectorstore.query(query_embedding=query_embedding, n_results=4)

    # ChromaDB'den dönen sonuçları Langchain Document formatına çeviriyoruz
    documents = []
    if results and "documents" in results and results["documents"]:
        for i, doc_content in enumerate(results["documents"][0]):
            meta = results["metadatas"][0][i]
            documents.append(Document(page_content=doc_content, metadata=meta))

    return {"documents": documents, "question": question}


def grade_documents(state: GraphState):
    """
    Getirilen dokümanların soruyla gerçekten alakalı olup olmadığını Grok'a sorar.
    Alakasız olanları eler.
    """
    print("--- ⚖️ DOKÜMANLAR DEĞERLENDİRİLİYOR ---")
    question = state["question"]
    documents = state["documents"]

    # Doküman değerlendirme promptu
    prompt = PromptTemplate(
        template="""Sen medikal bir değerlendiricisin. 
        Aşağıdaki getirilen dokümanın kullanıcının sorusuyla anlamsal olarak alakalı olup olmadığını değerlendir.
        Eğer doküman soruyu cevaplamak için faydalı bilgiler içeriyorsa 'evet', içermiyorsa 'hayır' cevabını ver.
        Başka hiçbir şey yazma.

        Soru: {question}
        Doküman: {document}
        """,
        input_variables=["question", "document"],
    )

    chain = prompt | llm | StrOutputParser()

    filtered_docs = []
    for d in documents:
        score = chain.invoke({"question": question, "document": d.page_content})
        grade = score.strip().lower()
        if "evet" in grade:
            print(f"  ✓ Alakalı doküman bulundu: {d.metadata.get('file', 'Bilinmeyen Dosya')}")
            filtered_docs.append(d)
        else:
            print("  x Alakasız doküman elendi.")

    return {"documents": filtered_docs, "question": question}


def generate(state: GraphState):
    """
    Grok'u kullanarak alakalı dokümanlardaki bilgileri hastanın anlayacağı basit bir dille özetler.
    """
    print("--- 📝 HASTA İÇİN CEVAP ÜRETİLİYOR ---")
    question = state["question"]
    documents = state["documents"]

    # Eğer alakalı doküman kalmadıysa
    if not documents:
        return {
            "generation": "Üzgünüm, mevcut diyabet kılavuzlarımızda bu sorunun cevabına dair net bir bilgi bulamadım. Lütfen bu konuyu doğrudan doktorunuza danışın."}

    # Dokümanları tek bir metinde birleştir
    context = "\n\n".join([doc.page_content for doc in documents])

    # Hastaya ve aileye yönelik özel prompt
    prompt = PromptTemplate(
        template="""Sen şefkatli, anlaşılır ve güvenilir bir medikal yapay zeka asistanısın.
        Görevin, aşağıda verilen diyabet kılavuzu bilgilerini kullanarak hastaların ve ailelerin sorularını cevaplamaktır.

        LÜTFEN ŞU KURALLARA KESİNLİKLE UY:
        1. Karmaşık tıbbi terimleri (örneğin 'DKA', 'hipoglisemi', 'postprandiyal') kullanman gerekirse, bunların ne anlama geldiğini hastanın anlayacağı çok basit ve günlük bir dille açıkla.
        2. Şefkatli, sakinleştirici ve destekleyici bir üslup kullan, ancak SADECE aşağıdaki bağlamda (context) yer alan gerçeklere dayan. Asla bağlam dışına çıkma ve uydurma.
        3. Metinleri kopyala-yapıştır yapmak yerine doğal bir konuşma diliyle özetle.
        4. Cevabının sonuna her zaman mutlaka şu uyarıyı ekle: "Not: Ben bir yapay zeka asistanıyım. Bu bilgiler kılavuzlara dayansa da, herhangi bir tedavi değişikliği yapmadan önce lütfen mutlaka kendi doktorunuza danışın."

        Bağlam:
        {context}

        Soru: {question}

        Cevap:""",
        input_variables=["context", "question"],
    )

    chain = prompt | llm | StrOutputParser()
    generation = chain.invoke({"context": context, "question": question})

    return {"documents": documents, "question": question, "generation": generation}

# ==========================================
# 3. GRAPH AKIŞI (Ajanın Mantığı)
# ==========================================
workflow = StateGraph(GraphState)

# Düğümleri ekle
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate)

# Akışı bağla
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_edge("grade_documents", "generate")
workflow.add_edge("generate", END)

# Ajanı derle
app = workflow.compile()


# ==========================================
# 4. ÇALIŞTIRMA VE SOHBET DÖNGÜSÜ
# ==========================================

def run_agent_loop():
    print("\n" + "=" * 50)
    print("🩺 Diyabet Asistanı Başlatıldı!")
    print("Aklınıza takılanları sorabilirsiniz. (Çıkmak için 'q', 'çıkış' veya 'exit' yazın)")
    print("=" * 50)

    while True:
        # Kullanıcıdan soruyu al
        user_input = input("\nSiz: ")

        # Çıkış komutlarını kontrol et
        if user_input.lower() in ['q', 'çıkış', 'exit', 'kapat']:
            print("\n🤖 Asistan: Geçmiş olsun, sağlıklı günler dilerim! Kapatılıyor...")
            break

        # Boş girdileri atla
        if not user_input.strip():
            print("Lütfen bir soru girin.")
            continue

        print("\n⏳ Asistan kayıtları inceliyor ve düşünüyor...\n")

        try:
            # Ajanı başlatmak için başlangıç state'ini veriyoruz
            initial_state = {"question": user_input, "documents": [], "generation": ""}

            # app.invoke() tüm grafı baştan sona çalıştırır ve son state'i döndürür
            result = app.invoke(initial_state)

            # Üretilen cevabı ekrana yazdır
            print("\n🤖 Asistan:")
            print(result.get("generation", "Üzgünüm, bir cevap üretilemedi."))
            print("\n" + "-" * 50)

        except Exception as e:
            print(f"\n❌ Bir hata oluştu: {e}")
            print("Lütfen sistem bağlantılarınızı (API anahtarı, ChromaDB vb.) kontrol edin.")


# Döngüyü başlat
if __name__ == "__main__":
    run_agent_loop()