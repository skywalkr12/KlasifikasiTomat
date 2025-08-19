
import streamlit as st

diseases = {
    "Bacterial Spot": {
        "desc": "Disebabkan oleh bakteri *Xanthomonas campestris*. Bercak kecil coklat atau hitam pada daun dan buah, tepi bercak sering berwarna kuning.",
        "handling": "Gunakan benih sehat, rotasi tanaman, semprot tembaga atau bakterisida sesuai anjuran.",
        "source": "https://plantvillage.psu.edu/topics/tomato"
    },
    "Early Blight": {
        "desc": "Disebabkan oleh jamur *Alternaria solani*. Gejalanya bercak coklat dengan lingkaran konsentris pada daun tua.",
        "handling": "Gunakan fungisida berbahan aktif klorotalonil atau mankozeb. Buang daun terinfeksi & rotasi tanaman.",
        "source": "https://extension.umn.edu/diseases/early-blight-tomato"
    },
    "Late Blight": {
        "desc": "Disebabkan oleh oomycete *Phytophthora infestans*. Daun muncul bercak air, kemudian membusuk.",
        "handling": "Semprot fungisida berbahan aktif metalaksil atau tembaga. Hindari kelembaban tinggi.",
        "source": "https://www.rhs.org.uk/disease/late-blight"
    },
    "Leaf Mold": {
        "desc": "Disebabkan oleh jamur *Passalora fulva*. Muncul bercak kuning di atas daun, dengan lapisan jamur hijau di bawah daun.",
        "handling": "Perbaiki sirkulasi udara, kurangi kelembaban. Fungisida berbahan aktif tembaga efektif.",
        "source": "https://extension.psu.edu/tomato-leaf-mold"
    },
    "Septoria Leaf Spot": {
        "desc": "Disebabkan oleh jamur *Septoria lycopersici*. Bercak kecil coklat kehitaman dengan tepi jelas.",
        "handling": "Buang daun terinfeksi, gunakan fungisida berbahan aktif klorotalonil atau mankozeb.",
        "source": "https://extension.umn.edu/diseases/septoria-leaf-spot"
    },
    "Spider Mites": {
        "desc": "Serangan tungau laba-laba (*Tetranychus urticae*). Daun menguning, ada jaring halus.",
        "handling": "Gunakan akarisida atau semprot air sabun. Pelihara predator alami seperti kumbang.",
        "source": "https://ipm.ucanr.edu/PMG/PESTNOTES/pn7405.html"
    },
    "Target Spot": {
        "desc": "Disebabkan oleh jamur *Corynespora cassiicola*. Bercak melingkar dengan pusat keabu-abuan.",
        "handling": "Gunakan fungisida berbahan aktif klorotalonil. Hindari penyiraman berlebihan.",
        "source": "https://www.daf.qld.gov.au/business-priorities/plants"
    },
    "Yellow Leaf Curl Virus": {
        "desc": "Virus yang ditularkan kutu kebul (*Bemisia tabaci*). Daun mengeriting & pertumbuhan terhambat.",
        "handling": "Kendalikan vektor (kutu kebul) dengan insektisida & perangkap kuning. Gunakan varietas tahan virus.",
        "source": "https://www.plantwise.org/knowledgebank"
    },
    "Mosaic Virus": {
        "desc": "Penyakit virus yang menyebabkan mosaik kuning-hijau pada daun.",
        "handling": "Cabut tanaman terinfeksi. Gunakan bibit sehat & bersihkan peralatan tanam.",
        "source": "https://www.rhs.org.uk/disease/tobacco-mosaic-virus"
    },
    "Healthy": {
        "desc": "Tanaman sehat dengan daun hijau tanpa bercak atau gejala penyakit.",
        "handling": "Pertahankan sanitasi kebun dan pemupukan seimbang.",
        "source": "-"
    }
}

st.title("🩺 Informasi Penyakit Tomat")

for name, data in diseases.items():
    st.subheader(name)
    st.image(f"images/{name}.jpg", width=200)
    st.write(f"**Penyebab & Gejala:** {data['desc']}")
    st.write(f"**Penanganan:** {data['handling']}")
    if data["source"] != "-":
        st.markdown(f"[Sumber]({data['source']})")
    st.markdown("---")

st.write("""
Catatan: Ini adalah alat diagnosis dengan bantuan Kecerdasan Buatan dan sebaiknya digunakan hanya sebagai panduan. Untuk diagnosis yang konklusif, konsultasikan dengan ahli patologi tanaman profesional.
""")

st.markdown("---")
st.markdown("""
<div style='text-align: center; font-size:14px;'>
<b>© - 2025 | Muhammad Sahrul Farhan | 51421076</b><br>
🔗 <a href="https://www.linkedin.com/in/muhammad-sahrul-farhan/" target="blank_">LinkedIn</a> | 
<a href="https://www.instagram.com/eitcheien/" target="blank_">Instagram</a> | 
<a href="https://www.facebook.com/skywalkr12" target="blank_">Facebook</a>
</div>
""", unsafe_allow_html=True)

