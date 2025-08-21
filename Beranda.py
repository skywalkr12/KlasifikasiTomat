
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Deteksi Penyakit Tomat", layout="wide")

# Header dengan logo dan judul
logo = Image.open("images/logo.png")
st.image(logo, width=100)
st.markdown("<h1 style='color:#b22222;'>🍅 Aplikasi Deteksi Penyakit Tomat</h1>", unsafe_allow_html=True)

st.write("""
Selamat datang di aplikasi deteksi penyakit tomat berbasis Deep Learning.
Gunakan menu di sidebar untuk Membaca Informasi Penyakit Tanaman Tomat, Memprediksi Penyakit Tanaman Tomat, 
dan geser ke bawah di bagian beranda untuk Membaca Informasi Singkat tentang Tanaman Tomat 

""")

st.image("Beranda_tomat.jpeg", use_container_width=True)

st.write("""
Tomat atau rangam (Solanum lycopersicum) adalah tumbuhan dari keluarga Terong-terongan, tumbuhan asli Amerika Tengah dan Selatan, dari Meksiko sampai Peru. Tomat merupakan tumbuhan siklus hidup singkat, dapat tumbuh setinggi 1 sampai 3 meter. Tumbuhan ini memiliki buah berwarna hijau, kuning, dan merah yang biasa dipakai sebagai sayur dalam masakan atau dimakan secara langsung tanpa diproses. Tomat memiliki batang dan daun yang tidak dapat dikonsumsi karena masih sekeluarga dengan kentang dan terung yang mengadung alkaloid.
""")

st.write("""
Cara menanam tanaman tomat adalah disemai lebih dahulu, setelah tumbuh 4 daun sejati kemudian ditanam (dijadikan bibit terlebih dahulu). Panen dimulai usia 9 minggu setelah tanam selanjutnya setiap 5 hari.
""")

st.write("""
Salah satu produk buatan hasil olahan tomat yang digemari, yaitu saus tomat. Hampir di berbagai negara memproduksi saus tomat. Saus tomat sendiri banyak dimanfaatkan untuk bumbu tambahan dalam mengolah berbagai masakan. Saus tomat juga dapat disajikan langsung bersama burger, sandwich, dan banyak lagi yang lainnya. Selain untuk tambahan dalam mengolah makanan, tomat juga dapat diolah menjadi jus tomat dan sebagai bahan tambahan untuk membuat sambal.
""")

st.write("""
Dikutip dari sumber: https://id.wikipedia.org/wiki/Tomat
""")

# Footer
st.markdown("---")

st.info(
    "Sebagai Catatan: Ini adalah alat diagnosis dengan bantuan Kecerdasan Buatan dan sebaiknya digunakan hanya sebagai panduan. Untuk diagnosis konklusif, konsultasikan dengan ahli patologi tanaman profesional."
)

st.markdown("""

<div style='text-align: center; font-size:14px;'>
<b>© - 2025 | Muhammad Sahrul Farhan | 51421076</b><br>
🔗 <a href="https://www.linkedin.com/in/muhammad-sahrul-farhan/" target="blank_">LinkedIn</a> | 
<a href="https://www.instagram.com/eitcheien/" target="blank_">Instagram</a> | 
<a href="https://www.facebook.com/skywalkr12" target="blank_">Facebook</a>
</div>
""", unsafe_allow_html=True)









