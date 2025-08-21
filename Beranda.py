
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Deteksi Penyakit Tomat", layout="wide")

# Header dengan logo dan judul
logo = Image.open("images/logo.png")
st.image(logo, width=100)
st.markdown("<h1 style='color:#b22222;'>🍅 Aplikasi Deteksi Penyakit Tomat</h1>", unsafe_allow_html=True)

st.write("""
Selamat datang di aplikasi deteksi penyakit tomat berbasis Deep Learning.
Gunakan menu di sidebar untuk membaca informasi penyakit dan memprediksi penyakit tomat.
""")

st.image("Beranda_tomat.jpeg", use_container_width=True)

st.write(""" Sektor pertanian memegang peranan penting dalam mendorong pertumbuhan ekonomi nasional. 
Secara umum, sektor ini terdiri atas beberapa subsektor, seperti hortikultura, perkebunan, dan tanaman pangan. 
Di antara ketiganya, subsektor hortikultura, yang mencakup komoditas buah dan sayuran, menjadi komponen krusial dalam mendukung peningkatan Produk Domestik Bruto (PDB). 
Salah satu komoditas hortikultura yang memiliki prospek tinggi adalah tomat, yaitu sayuran multifungsi dengan beragam kegunaan.
""")

st.write("""
Sebagai Catatan: Ini adalah alat diagnosis dengan bantuan Kecerdasan Buatan dan sebaiknya digunakan hanya sebagai panduan.
Untuk diagnosis konklusif, konsultasikan dengan ahli patologi tanaman profesional.
""")

# Footer
st.markdown("---")
st.markdown("""

<div style='text-align: center; font-size:14px;'>
<b>© - 2025 | Muhammad Sahrul Farhan | 51421076</b><br>
🔗 <a href="https://www.linkedin.com/in/muhammad-sahrul-farhan/" target="blank_">LinkedIn</a> | 
<a href="https://www.instagram.com/eitcheien/" target="blank_">Instagram</a> | 
<a href="https://www.facebook.com/skywalkr12" target="blank_">Facebook</a>
</div>
""", unsafe_allow_html=True)




