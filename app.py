
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Deteksi Penyakit Tomat", layout="wide")

# Header dengan logo dan judul
logo = Image.open("images/logo.png")
st.image(logo, width=100)
st.markdown("<h1 style='color:#b22222;'>🍅 Aplikasi Deteksi Penyakit Tomat</h1>", unsafe_allow_html=True)

st.write("""
Selamat datang di aplikasi deteksi penyakit tomat berbasis Machine Learning.
Gunakan menu di sidebar untuk membaca informasi penyakit dan memprediksi gambar daun tomat.
""")

st.image("images/tomato_banner.jpg", use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; font-size:14px;'>
Dibuat oleh <b>Farhan</b><br>
🔗 <a href="https://id.linkedin.com/in/muhammad-sahrul-farhan/" target="_blank">LinkedIn</a> | 
<a href="https://instagram.com/eitcheien/" target="_blank">Instagram</a> | 
<a href="https://facebook.com/skywalkr12" target="_blank">Facebook</a>
</div>
""", unsafe_allow_html=True)
