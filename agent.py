import os
from typing import List
from typing_extensions import TypedDict
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import END, StateGraph
from langchain_groq import ChatGroq

from rag import embedding_manager, vectorstore

# Groq API Anahtarı


# Groq modelini başlatalım
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)


# ==========================================
# 1. GRAPH STATE (Ajanın Hafızası)
# ==========================================
class GraphState(TypedDict):
    question: str
    documents: List[Document]
    generation: str
    chat_history: List[dict]  # {"role": "user"/"assistant", "content": "..."}


# ==========================================
# 2. NODES (Düğümler - Ajanın Aksiyonları)
# ==========================================

def _parse_results(results) -> List[Document]:
    docs = []
    if results and "documents" in results and results["documents"]:
        for i, content in enumerate(results["documents"][0]):
            docs.append(Document(page_content=content, metadata=results["metadatas"][0][i]))
    return docs


def retrieve(state: GraphState):
    print("--- 🔍 RETRIEVE (DOKÜMAN GETİRİLİYOR) ---")
    question = state["question"]
    chat_history = state.get("chat_history", [])

    if chat_history:
        recent = " ".join([m["content"] for m in chat_history[-4:]])
        enriched_query = f"{recent} {question}"
    else:
        enriched_query = question

    query_embedding = embedding_manager.generate_embeddings([enriched_query])[0]

    # Her iki dil grubundan ayrı ayrı çek, böylece Türkçe sorgular İngilizce kaynakları ezmez
    tr_results = vectorstore.query(query_embedding=query_embedding, n_results=3, where={"language": "tr"})
    en_results = vectorstore.query(query_embedding=query_embedding, n_results=3, where={"language": "en"})

    tr_docs = _parse_results(tr_results)
    en_docs = _parse_results(en_results)

    print(f"  📘 Türkçe kaynak: {len(tr_docs)} chunk")
    print(f"  📗 İngilizce kaynak: {len(en_docs)} chunk")

    # Türkçe + İngilizce birleştir (tekrar eden içerikler grade aşamasında elenir)
    documents = tr_docs + en_docs

    return {"documents": documents, "question": question, "chat_history": chat_history}


def grade_documents(state: GraphState):
    print("--- ⚖️ DOKÜMANLAR DEĞERLENDİRİLİYOR ---")
    question = state["question"]
    documents = state["documents"]
    chat_history = state.get("chat_history", [])

    # Önceki semptomları da bağlam olarak al
    history_summary = ""
    if chat_history:
        history_summary = "\n".join([
            f"{'Hasta' if m['role'] == 'user' else 'Asistan'}: {m['content']}"
            for m in chat_history[-4:]
        ])

    prompt = PromptTemplate(
        template="""Sen medikal bir değerlendiricisin.
        Bir hasta ile asistan arasındaki konuşma geçmişi ve hastanın mevcut sorusu aşağıda verilmiştir.
        Getirilen dokümanın bu konuşma bağlamında diyabet veya ilgili sağlık konularına dair GENEL OLARAK yararlı
        bilgi içerip içermediğini değerlendir.
        Doküman diyabet, belirtiler, tanı, tedavi veya komplikasyonlar hakkında herhangi bir bilgi içeriyorsa 'evet' de.
        Tamamen alakasız bir konudaysa (örneğin kalp hastalığı, ortopedi vb.) 'hayır' de.
        Sadece 'evet' veya 'hayır' yaz, başka hiçbir şey yazma.

        Konuşma geçmişi:
        {history}

        Mevcut soru: {question}
        Doküman: {document}
        """,
        input_variables=["question", "document", "history"],
    )

    chain = prompt | llm | StrOutputParser()

    filtered_docs = []
    for d in documents:
        score = chain.invoke({
            "question": question,
            "document": d.page_content,
            "history": history_summary
        })
        grade = score.strip().lower()
        if "evet" in grade:
            print(f"  ✓ Alakalı doküman bulundu: {d.metadata.get('file', 'Bilinmeyen Dosya')}")
            filtered_docs.append(d)
        else:
            print("  x Alakasız doküman elendi.")

    return {"documents": filtered_docs, "question": question, "chat_history": chat_history}


def generate(state: GraphState):
    print("--- 📝 HASTA İÇİN CEVAP ÜRETİLİYOR ---")
    question = state["question"]
    documents = state["documents"]
    chat_history = state.get("chat_history", [])

    # Konuşma geçmişini formatlı hale getir
    history_text = ""
    if chat_history:
        history_text = "\n".join([
            f"{'Hasta' if m['role'] == 'user' else 'Asistan'}: {m['content']}"
            for m in chat_history[-6:]
        ])

    if not documents:
        return {
            "generation": "Üzgünüm, mevcut diyabet kılavuzlarımızda bu sorunun cevabına dair net bir bilgi bulamadım. Lütfen bu konuyu doğrudan doktorunuza danışın.",
            "chat_history": chat_history
        }

    context = "\n\n".join([doc.page_content for doc in documents])

    prompt = PromptTemplate(
        template="""Sen şefkatli, anlaşılır ve güvenilir bir medikal yapay zeka asistanısın.
        Görevin, aşağıda verilen diyabet kılavuzu bilgilerini kullanarak hastaların sorularını cevaplamaktır.

        LÜTFEN ŞU KURALLARA KESİNLİKLE UY:
        1. Önceki konuşma geçmişini göz önünde bulundur; hasta daha önce belirttiği semptomları tekrar sormasa bile bunları bağlam olarak kullan.
        2. Karmaşık tıbbi terimleri kullanman gerekirse, bunları basit ve günlük bir dille açıkla.
        3. Şefkatli ve destekleyici bir üslup kullan; SADECE aşağıdaki bağlamda yer alan gerçeklere dayan.
        4. Doğal bir konuşma diliyle özetle, kopyala-yapıştır yapma.
        5. Cevabının sonuna her zaman şu uyarıyı ekle: "Not: Ben bir yapay zeka asistanıyım. Bu bilgiler kılavuzlara dayansa da, herhangi bir tedavi değişikliği yapmadan önce lütfen mutlaka kendi doktorunuza danışın."

        Önceki Konuşma:
        {history}

        Diyabet Kılavuzu Bağlamı:
        {context}

        Hastanın Mevcut Sorusu: {question}

        Cevap:""",
        input_variables=["context", "question", "history"],
    )

    chain = prompt | llm | StrOutputParser()
    generation = chain.invoke({"context": context, "question": question, "history": history_text})

    return {"documents": documents, "question": question, "generation": generation, "chat_history": chat_history}


# ==========================================
# 3. GRAPH AKIŞI
# ==========================================
workflow = StateGraph(GraphState)

workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_edge("grade_documents", "generate")
workflow.add_edge("generate", END)

app = workflow.compile()


# ==========================================
# 4. ÇALIŞTIRMA VE SOHBET DÖNGÜSÜ
# ==========================================

def run_agent_loop():
    print("\n" + "=" * 50)
    print("🩺 Diyabet Asistanı Başlatıldı!")
    print("Aklınıza takılanları sorabilirsiniz. (Çıkmak için 'q', 'çıkış' veya 'exit' yazın)")
    print("=" * 50)

    chat_history = []

    while True:
        user_input = input("\nSiz: ")

        if user_input.lower() in ['q', 'çıkış', 'exit', 'kapat']:
            print("\n🤖 Asistan: Geçmiş olsun, sağlıklı günler dilerim! Kapatılıyor...")
            break

        if not user_input.strip():
            print("Lütfen bir soru girin.")
            continue

        print("\n⏳ Asistan kayıtları inceliyor ve düşünüyor...\n")

        try:
            initial_state = {
                "question": user_input,
                "documents": [],
                "generation": "",
                "chat_history": chat_history
            }

            result = app.invoke(initial_state)
            answer = result.get("generation", "Üzgünüm, bir cevap üretilemedi.")

            print("\n🤖 Asistan:")
            print(answer)
            print("\n" + "-" * 50)

            # Konuşma geçmişini güncelle
            chat_history.append({"role": "user", "content": user_input})
            chat_history.append({"role": "assistant", "content": answer})

            # Son 10 mesajı tut (5 tur)
            if len(chat_history) > 10:
                chat_history = chat_history[-10:]

        except Exception as e:
            print(f"\n❌ Bir hata oluştu: {e}")
            print("Lütfen sistem bağlantılarınızı (API anahtarı, ChromaDB vb.) kontrol edin.")


if __name__ == "__main__":
    run_agent_loop()
