
import streamlit as st
from PIL import Image

st.set_page_config(page_title="🍅 Klasifikasi Penyakit Tanaman Tomat", layout="wide")

# Header dengan logo dan judul
logo = Image.open("images/logo.png")
st.image(logo, width=100)
st.markdown("<h1 style='color:#b22222;'>🍅 Klasifikasi Penyakit Tanaman Tomat</h1>", unsafe_allow_html=True)

st.write("""
Selamat datang di aplikasi klasifikasi penyakit tanaman tomat menggunakan deep learning.
Gunakan menu di sidebar untuk membaca informasi terkait penyakit tanaman tomat dan mengklasifikasi penyakit tanaman tomat. 
""")

st.image("Beranda_tomat.jpeg", use_container_width=True)

st.header("🌱 Tentang Aplikasi Ini")
st.markdown("""
Aplikasi ini dikembangkan untuk membantu petani dan penggemar tanaman dalam mengidentifikasi penyakit pada daun tomat menggunakan Model Convolutional Neural Network (CNN) berbasis **ResNet**. 
Tujuan dibuatnya website ini adalah menyediakan visualisasi hasil klasifikasi **Penyakit Tanaman Tomat** secara informatif
""")
st.markdown("[Pelajari Lebih Lanjut tentang ResNet](https://pytorch.org/hub/pytorch_vision_resnet/)") # Ganti dengan link relevan

# Footer
st.markdown("---")

st.info( "Perlu diingat: Ini adalah alat diagnosis dengan bantuan Kecerdasan Buatan dan sebaiknya digunakan hanya sebagai panduan. Untuk diagnosis konklusif, konsultasikan dengan ahli patologi tanaman profesional."
)

st.markdown("""

<div style='text-align: center; font-size:14px;'>
<b>© - 2025 | Muhammad Sahrul Farhan | 51421076</b><br>
🔗 <a href="https://www.linkedin.com/in/muhammad-sahrul-farhan/" target="blank_">LinkedIn</a> | 
<a href="https://www.instagram.com/eitcheien/" target="blank_">Instagram</a> | 
<a href="https://www.facebook.com/skywalkr12" target="blank_">Facebook</a>
</div>
""", unsafe_allow_html=True)

















