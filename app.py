import streamlit as st
import arxiv
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import networkx as nx
import re
import requests
from bs4 import BeautifulSoup
from collections import Counter
from textblob import TextBlob

st.set_page_config(page_title="Akademik Keşif Platformu", layout="wide", page_icon="🎓")

st.markdown("""
    <style>
    .block-container {padding-top: 1rem; padding-bottom: 2rem;}
    div[data-testid="stMetricValue"] {font-size: 24px;}
    
    .tech-info {
        background-color: var(--secondary-background-color); /* Tema uyumlu arka plan */
        color: var(--text-color); /* Tema uyumlu yazı rengi */
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #4b7bec;
        font-size: 14px;
        margin-top: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🎓 Akademik Keşif Platformu")
st.markdown("""
**Amaç:** Akademik literatürü API ile taramak, görselleştirmek, NLP ile duygu analizi yapmak ve Web Scraping ile atıf verisi üretmektir.
""")

if 'arxiv_data' not in st.session_state:
    st.session_state['arxiv_data'] = pd.DataFrame()
if 'search_performed' not in st.session_state:
    st.session_state['search_performed'] = False
if 'bibtex_result' not in st.session_state:
    st.session_state['bibtex_result'] = None

with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/b/bc/ArXiv_logo_2022.svg/320px-ArXiv_logo_2022.svg.png", width=150)
    st.header("🔍 Analiz Parametreleri")
    
    st.info("ℹ️ Sistem, trend analizi için en alakalı sonuçları (Relevance) otomatik getirir.")

    with st.form(key='search_form'):
        keyword = st.text_input("Araştırma Konusu", value="Generative AI")
        max_results = st.slider("Maksimum Makale Sayısı", 10, 100, 50)
        submit_search = st.form_submit_button("🚀 Analizi Başlat")
    
    st.caption("Bu proje Dr. Öğretim Üyesi Halil İbrahim Okur rehberliğinde **Mühendislikte Bilgisayar Uygulamaları I** dersi kapsamında Oğuzhan Tutucu tarafından geliştirildi.")

def extract_balanced_bibtex(text):
    """BibTeX parantez dengeleyici."""
    start_index = text.find('@')
    if start_index == -1: return None
    balance = 0
    started = False
    for i in range(start_index, len(text)):
        char = text[i]
        if char == '{':
            balance += 1
            started = True
        elif char == '}':
            balance -= 1
        if started and balance == 0:
            return text[start_index : i+1]
    return None

def scrape_bibtex(paper_id):
    """BibTeX Scraper."""
    clean_id = re.sub(r'v\d+$', '', paper_id)
    url = f"https://export.arxiv.org/bibtex/{clean_id}"
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64)'}
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            bibtex_div = soup.find('div', id='bibtex')
            if bibtex_div: return bibtex_div.text.strip(), "Başarılı (Div Kaynağı)"
            text_content = soup.get_text()
            extracted_bib = extract_balanced_bibtex(text_content)
            if extracted_bib: return extracted_bib, "Başarılı (Metin Analizi)"
            return None, "Format bulunamadı."
        else: return None, f"Hata: {response.status_code}"
    except Exception as e: return None, f"Hata: {str(e)}"

def get_arxiv_data(query, max_res):
    client = arxiv.Client()
    search = arxiv.Search(query=query, max_results=max_res, sort_by=arxiv.SortCriterion.Relevance)
    
    data = []
    for r in client.results(search):
        paper_id = r.entry_id.split('/')[-1]
        data.append({
            "Tarih": r.published.date(),
            "Yıl": r.published.year,
            "Başlık": r.title,
            "Özet": r.summary.replace("\n", " "),
            "Yazarlar": [a.name for a in r.authors],
            "Ana Kategori": r.primary_category,
            "Link": r.entry_id,
            "ID": paper_id
        })
    return pd.DataFrame(data)

def plot_sentiment_analysis(df):
    """NLP Duygu Analizi."""
    df['Polarity'] = df['Özet'].apply(lambda x: TextBlob(x).sentiment.polarity)
    
    def get_sentiment_label(score):
        if score > 0.05: return "Pozitif (Umut Verici)"
        elif score < -0.05: return "Negatif (Kritik/Sorun Odaklı)"
        else: return "Nötr (Teknik/Tanımsal)"
    
    df['Duygu'] = df['Polarity'].apply(get_sentiment_label)
    
    sentiment_counts = df['Duygu'].value_counts().reset_index()
    sentiment_counts.columns = ['Duygu', 'Makale Sayısı']
    
    fig = px.pie(sentiment_counts, values='Makale Sayısı', names='Duygu', 
                 title="Literatürün Duygu Durumu (Abstract Sentiment)",
                 color='Duygu',
                 color_discrete_map={
                     "Pozitif (Umut Verici)": "#2ecc71",
                     "Nötr (Teknik/Tanımsal)": "#95a5a6",
                     "Negatif (Kritik/Sorun Odaklı)": "#e74c3c"
                 },
                 hole=0.4)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("#### 🔍 Örnek İnceleme")
    col1, col2 = st.columns(2)
    with col1:
        top_pos = df.nlargest(1, 'Polarity').iloc[0]
        st.info(f"**En Pozitif Makale:**\n{top_pos['Başlık']}")
    with col2:
        top_neg = df.nsmallest(1, 'Polarity').iloc[0]
        st.error(f"**En Kritik Makale:**\n{top_neg['Başlık']}")

def plot_trend_line(df):
    year_counts = df['Yıl'].value_counts().reset_index()
    year_counts.columns = ['Yıl', 'Makale Sayısı']
    year_counts = year_counts.sort_values('Yıl')
    fig = px.area(year_counts, x='Yıl', y='Makale Sayısı', markers=True, title="Yıllara Göre Yayın Trendi")
    st.plotly_chart(fig, use_container_width=True)

def analyze_keywords(df):
    text = " ".join(df['Özet'].tolist())
    text = re.sub(r'[^a-zA-Z\s]', '', text).lower()
    custom_stopwords = set(STOPWORDS)
    custom_stopwords.update(["paper", "proposed", "method", "result", "model", "based", "approach", "using", "show", "performance"])
    
    words = text.split()
    filtered_words = [w for w in words if w not in custom_stopwords and len(w) > 2]
    word_counts = Counter(filtered_words)
    
    wordcloud = WordCloud(width=800, height=350, background_color='white', stopwords=custom_stopwords, colormap='viridis').generate(text)
    return wordcloud, word_counts

def plot_optimized_network(df):
    """Plotly ile İnteraktif Ağ Analizi"""
    G = nx.Graph()
    for authors in df['Yazarlar']:
        if len(authors) > 1:
            for i in range(len(authors)):
                for j in range(i + 1, len(authors)):
                    if G.has_edge(authors[i], authors[j]): 
                        G[authors[i]][authors[j]]['weight'] += 1
                    else: 
                        G.add_edge(authors[i], authors[j], weight=1)
    
    if len(G.nodes) > 0:
        if len(G.nodes) > 30:
            degrees = dict(G.degree)
            top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:30]
            G = G.subgraph(top_nodes)
        
        pos = nx.spring_layout(G, k=0.5, seed=42)
        
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines')

        node_x = []
        node_y = []
        node_text = [] 
        node_adjacencies = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node)
            node_adjacencies.append(len(G.adj[node]))

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers', 
            hoverinfo='text',
            text=node_text,
            marker=dict(
                showscale=True,
                colorscale='YlGnBu', 
                reversescale=True,
                color=node_adjacencies, 
                size=20, 
                colorbar=dict(
                    thickness=15,
                    title='Bağlantı Sayısı',
                    xanchor='left'
                ),
                line_width=2))

        fig = go.Figure(data=[edge_trace, node_trace],
                     layout=go.Layout(
                        title={
                            'text': '<br>Akademik İş Birliği Ağı (İnteraktif)',
                            'font': {'size': 16}
                        },
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20,l=5,r=5,t=40),
                        annotations=[ dict(
                            text="İsimleri görmek için noktaların üzerine geliniz.",
                            showarrow=False,
                            xref="paper", yref="paper",
                            x=0.005, y=-0.002 ) ],
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Bu veri setinde yeterli yazar iş birliği bulunamadı.")

if submit_search:
    with st.spinner('Veriler API ile çekiliyor...'):
        df_new = get_arxiv_data(keyword, max_results)
        st.session_state['arxiv_data'] = df_new
        st.session_state['search_performed'] = True
        st.session_state['bibtex_result'] = None

if st.session_state.get('search_performed') and not st.session_state['arxiv_data'].empty:
    df = st.session_state['arxiv_data']
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Toplam Makale", len(df))
    unique_authors = set([a for sublist in df['Yazarlar'] for a in sublist])
    col2.metric("Farklı Yazar", len(unique_authors))
    col3.metric("En Aktif Yıl", int(df['Yıl'].mode()[0]) if not df['Yıl'].mode().empty else 0)
    col4.metric("Kategori Sayısı", df['Ana Kategori'].nunique())

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📈 Trend Analizi", 
        "☁️ Konu Modelleme", 
        "🧠 Duygu Analizi", 
        "🕸️ Yazar Ağı", 
        "📄 Detaylı Veri Seti",
        "🕷️ BibTeX Scraping"
    ])
    
    with tab1:
        st.subheader("Zaman İçindeki Yayın Eğilimi")
        plot_trend_line(df)
        st.markdown("""
        <div class="tech-info">
        <b>🛠️ Teknik Altyapı ve Metodoloji</b><br>
        <ul>
        <li><b>Nedir?</b> Seçilen araştırma konusunun yıllara göre yayınlanma sıklığını gösteren bir zaman serisi analizidir.</li>
        <li><b>Neden?</b> Bir teknolojinin veya akademik konunun yükselişte mi (Trending) yoksa düşüşte mi olduğunu tespit etmek için kullanılır.</li>
        <li><b>Nasıl?</b> Çekilen veriler Pandas kütüphanesi ile <code>groupby('Yıl')</code> işlemi uygulanarak gruplandırılır ve <code>Plotly Area Chart</code> ile görselleştirilir.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with tab2:
        st.subheader("Özet Analizi ve Kelime Frekansları")
        wc, counts = analyze_keywords(df)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)
        plt.close(fig)
        
        st.markdown("#### 🔢 En Sık Geçen Kelimeler (Top 50)")
        common_words_df = pd.DataFrame(counts.most_common(50), columns=['Kelime', 'Frekans'])
        st.dataframe(common_words_df, use_container_width=True, height=300)

        st.markdown("""
        <div class="tech-info">
        <b>🛠️ Teknik Altyapı ve Metodoloji</b><br>
        <ul>
        <li><b>Nedir?</b> Makale özetlerinden (Abstract) en sık kullanılan terimlerin çıkarılması işlemidir (Topic Modeling).</li>
        <li><b>Neden?</b> Literatürdeki alt çalışma alanlarını ve popüler terminolojiyi belirlemek için kullanılır.</li>
        <li><b>Nasıl?</b> Metinler önce Regex ile temizlenir, Stopwords (etkisiz kelimeler) çıkarılır ve <code>Counter</code> ile frekans analizi yapılır. Sonuçlar <code>WordCloud</code> kütüphanesi ile görselleştirilir.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    with tab3:
        st.subheader("NLP ile Özet Duygu Analizi (Sentiment Analysis)")
        st.markdown("Makale özetleri, **Doğal Dil İşleme (NLP)** kullanılarak analiz edilmiştir.")
        plot_sentiment_analysis(df)

        st.markdown("""
        <div class="tech-info">
        <b>🛠️ Teknik Altyapı ve Metodoloji</b><br>
        <ul>
        <li><b>Nedir?</b> Akademik metinlerin dilinin pozitif (başarılı/umut verici) mi yoksa negatif (sorun odaklı/kritik) mi olduğunu analiz eden bir NLP sürecidir.</li>
        <li><b>Neden?</b> Literatürün genel atmosferini ve araştırmacıların konuya yaklaşımını anlamak için kullanılır.</li>
        <li><b>Nasıl?</b> Python <code>TextBlob</code> kütüphanesi kullanılarak her özet için bir Polarity skoru (-1 ile +1 arası) hesaplanır ve kategorize edilir.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    with tab4:
        st.subheader("Akademik İş Birliği Ağı")
        plot_optimized_network(df)

        st.markdown("""
        <div class="tech-info">
        <b>🛠️ Teknik Altyapı ve Metodoloji</b><br>
        <ul>
        <li><b>Nedir?</b> Yazarlar arasındaki ortak çalışma (Co-authorship) ilişkilerini gösteren bir grafik teorisi uygulamasıdır.</li>
        <li><b>Neden?</b> Alanın en üretken gruplarını, merkezi yazarları (Hubs) ve iş birliği kümelerini keşfetmek için kullanılır.</li>
        <li><b>Nasıl?</b> <code>NetworkX</code> kütüphanesi ile düğümler (Yazarlar) ve kenarlar (Ortak Makaleler) oluşturulur. Fruchterman-Reingold algoritması ile yerleşim yapılır ve <code>Plotly</code> ile interaktif hale getirilir.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
    with tab5:
        st.subheader("Ham Veri Tablosu")
        display_df = df[['Tarih', 'Ana Kategori', 'Başlık', 'Yazarlar', 'Özet', 'Link']]
        st.dataframe(display_df, use_container_width=True)
        
        csv = display_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Veri Setini CSV Olarak İndir",
            data=csv,
            file_name=f'{keyword}_arxiv_data.csv',
            mime='text/csv',
        )

        st.markdown("""
        <div class="tech-info">
        <b>🛠️ Teknik Altyapı ve Metodoloji</b><br>
        <ul>
        <li><b>Nedir?</b> Analiz edilen tüm verinin yapılandırılmış (Structured) ham halidir.</li>
        <li><b>Neden?</b> Şeffaflık sağlamak ve verilerin başka araçlarda (Excel, SPSS) kullanılabilmesine olanak tanımak için.</li>
        <li><b>Nasıl?</b> Veriler <code>Pandas DataFrame</code> objesinde tutulur ve <code>to_csv</code> fonksiyonu ile UTF-8 formatında dışa aktarılır.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    with tab6:
        st.subheader("📚 Otomatik BibTeX Oluşturucu")
        st.success("Web Scraping modülü, ArXiv'in export sunucularına bağlanarak atıf verisini doğrular.")
        
        with st.form(key='scrape_form'):
            selected_paper = st.selectbox("Kaynakçası oluşturulacak makaleyi seçiniz:", df['Başlık'])
            scrape_btn = st.form_submit_button("BibTeX Kodunu Kazı (Scrape)")

        if scrape_btn:
            paper_id = df[df['Başlık'] == selected_paper]['ID'].values[0]
            with st.spinner("ArXiv sunucularına bağlanılıyor..."):
                bibtex_code, status = scrape_bibtex(paper_id)
                st.session_state['bibtex_result'] = (bibtex_code, status, selected_paper)
        
        if st.session_state['bibtex_result']:
            code, stat, title = st.session_state['bibtex_result']
            st.markdown(f"**Seçilen Makale:** {title}")
            
            if code:
                st.success(f"✅ Scraping Başarılı! ({stat})")
                st.code(code, language='latex')
            else:
                st.error(f"❌ {stat}")
        
        st.markdown("""
        <div class="tech-info">
        <b>🛠️ Teknik Altyapı ve Metodoloji</b><br>
        <ul>
        <li><b>Nedir?</b> ArXiv API'sinin sağlamadığı BibTeX (LaTeX Atıf Formatı) verisinin web sitesinden canlı olarak çekilmesidir.</li>
        <li><b>Neden?</b> Araştırmacıların makaleyi kaynakçalarına ekleyebilmesi için gereklidir.</li>
        <li><b>Nasıl?</b> Python <code>Requests</code> ve <code>BeautifulSoup</code> kütüphaneleri kullanılarak HTML ayrıştırma (Web Scraping) yapılır. Regex ve metin analizi ile BibTeX bloğu tespit edilip temizlenir.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)