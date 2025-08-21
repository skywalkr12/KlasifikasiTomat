
import streamlit as st

diseases = {
	"Healthy": {
        "desc": " • Warna hijau merata (dari hijau sedang ke hijau tua) tanpa pola mosaik/klorosis yang aneh.
                  • Tidak ada bercak/lesi (tidak ada titik kecil keabu-abuan, bercak berlingkar “target”, tepi menghitam, atau bercak berminyak).
                  • Tidak menggulung/mengeriting dan tidak menyempit; helaian tetap rata dengan turgor baik (tidak layu).
                  • Tepi daun utuh (tidak sobek/nekrotik).
                  • Bagian bawah daun bersih—tidak ada serbuk/jamur (mis. lapisan zaitun khas leaf mold), jaring halus tungau, atau honeydew/sooty mold dari kutu kebul/aphid.
                  • Urat daun normal (tidak menebal/menonjol).
                  • Pertumbuhan tunas baru tampak segar dan simetris.",
        "handling": " • Siram di pangkal (drip) pada pagi hari; hindari membasahi daun. Ini memutus percikan patogen daun.
		     • Mulsa (organik/ plastik) untuk menahan percikan tanah & stabilkan kelembapan.
		     • Jarang-kanopi & ajir/trellis supaya sirkulasi udara bagus; pangkas daun bawah yang menyentuh tanah.
		     • Rotasi 2–3 tahun + sanitasi alat & lahan; jangan bekerja saat tanaman basah.
		     • pH & hara seimbang: target pH tanah ± 6.0–6.8, uji tanah, hindari N berlebih.
			 • Monitoring hama mingguan (bawah daun): jaga musuh alami; semprot sabun/minyak bila ambang terlampaui. Untuk daerah risiko TYLCV, mulsa reflektif saat awal musim menekan kedatangan kutu kebul.",

    },

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
Sebagai Catatan: Ini adalah alat diagnosis dengan bantuan Kecerdasan Buatan dan sebaiknya digunakan hanya sebagai panduan.
Untuk diagnosis konklusif, konsultasikan dengan ahli patologi tanaman profesional.
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




