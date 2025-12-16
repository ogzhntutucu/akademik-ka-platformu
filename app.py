import streamlit as st
import arxiv
import pandas as pd

st.set_page_config(page_title="Akademik Trend Asistanı", layout="wide")

st.title("🎓 Akademik Trend ve Makale Keşif Platformu")
st.markdown("ArXiv veritabanı üzerinden gerçek zamanlı veri madenciliği.")

with st.sidebar:
    st.header("Arama Parametreleri")
    keyword = st.text_input("Anahtar Kelime", value="Artificial Intelligence")
    max_results = st.slider("Makale Sayısı", 10, 100, 20)
    search_btn = st.button("Verileri Getir")

def get_arxiv_data(query, max_res):
    client = arxiv.Client()
    
    search = arxiv.Search(
        query = query,
        max_results = max_res,
        sort_by = arxiv.SortCriterion.SubmittedDate
    )
    
    results = []
    for r in client.results(search):
        results.append({
            "Tarih": r.published.date(),
            "Başlık": r.title,
            "Yazarlar": ", ".join([a.name for a in r.authors]),
            "Özet": r.summary,
            "Link": r.entry_id,
            "Kategori": r.primary_category
        })
    
    return pd.DataFrame(results)

if search_btn:
    with st.spinner(f"'{keyword}' için son {max_results} makale çekiliyor..."):
        try:
            df = get_arxiv_data(keyword, max_results)
            
            st.success(f"✅ Toplam {len(df)} makale başarıyla çekildi!")
            
            st.subheader("Ham Veri Seti")
            st.dataframe(df, use_container_width=True)
        
            st.session_state['df'] = df
            
        except Exception as e:
            st.error(f"Bir hata oluştu: {e}")