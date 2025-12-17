
# 📚 Akademik Keşif ve Analiz Platformu  


> Bu proje, **İskenderun Teknik Üniversitesi – Mühendislikte Bilgisayar Uygulamaları I** dersi kapsamında,  
> **Dr. Öğr. Üyesi H. İbrahim OKUR** gözetiminde, **2025–2026 Güz Döneminde** geliştirilmiştir.

---

*Akademik literatürü yapay zeka ile tarayan, analiz eden ve görselleştiren hibrit karar destek sistemi.*
 
📜 **Lisans:** `MIT License`  

---

## 🚀 Proje Özeti

Akademik dünyada her gün binlerce makale yayınlanmaktadır.  
Bu platform; ArXiv API, Web Scraping, NLP ve Ağ Analizi tekniklerini hibrit bir mimaride birleştirerek
araştırmacılara güçlü bir karar destek sistemi sunar.

---

## ✨ Öne Çıkan Özellikler

### 📈 Hibrit Veri Madenciliği (API + Scraping)
- ArXiv API ile temel meta veriler
- Web Scraping ile otomatik BibTeX üretimi

### 🧠 NLP Tabanlı Duygu Analizi
- TextBlob ile pozitif / negatif literatür analizi
- İnteraktif grafikler

### 🕸️ İnteraktif Yazar Ağı
- NetworkX + Plotly
- Zoom / Hover destekli ağ yapısı

### ☁️ Konu Modelleme (WordCloud)
- Teknik terim ve trend analizi

### 🌑 Akıllı Arayüz
- Streamlit Session State
- Dark / Light Mode uyumu
- CSV Export

---

## 🧱 Teknoloji Yığını

- Python 3.12+
- Streamlit
- Pandas, NumPy
- Plotly, Matplotlib
- TextBlob, NLTK
- BeautifulSoup, Requests, Regex
- NetworkX

---

## ⚙️ Kurulum

```bash
git clone https://github.com/thwisse/akademik-ka-platformu.git
cd akademik-ka-platformu
```

### Sanal Ortam

**Windows**
```bash
python -m venv venv
venv\Scripts\activate
```

**Mac / Linux**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Bağımlılıklar
```bash
pip install -r requirements.txt
```

### Çalıştırma
```bash
streamlit run app.py
```

---

## 👨‍💻 Geliştirici

**Oğuzhan Tutucu**  
Bilgisayar Mühendisliği – İSTE  
Öğrenci No: 212523033  