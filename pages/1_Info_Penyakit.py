import streamlit as st

st.set_page_config(page_title="🩺 Informasi Penyakit Tomat", layout="centered")

# -------------------------- Data --------------------------
# Catatan: untuk tampilan vertikal rapi, "Healthy" memakai list.
# Item lain boleh string biasa; renderer di bawah akan menangani keduanya.
diseases = {
    "Healthy": {
        "desc": [
            "Warna hijau merata (dari hijau sedang ke hijau tua) tanpa pola mosaik/klorosis yang aneh.",
            "Tidak ada bercak/lesi (tidak ada titik kecil keabu-abuan, bercak berlingkar “target”, tepi menghitam, atau bercak berminyak).",
            "Tidak menggulung/mengeriting dan tidak menyempit; helaian tetap rata dengan turgor baik (tidak layu).",
            "Tepi daun utuh (tidak sobek/nekrotik).",
            "Bagian bawah daun bersih—tidak ada serbuk/jamur (mis. lapisan zaitun khas leaf mold), jaring halus tungau, atau honeydew/sooty mold dari kutu kebul/aphid.",
            "Urat daun normal (tidak menebal/menonjol).",
            "Pertumbuhan tunas baru tampak segar dan simetris."
        ],
        "handling": [
            "Siram di pangkal (drip) pada pagi hari; jangan membasahi daun.",
            "Gunakan mulsa (organik/plastik) untuk mengurangi percikan tanah & stabilkan kelembapan.",
            "Atur jarak tanam; ajir/trellis agar kanopi berangin; pangkas daun bawah yang menyentuh tanah.",
            "Rotasi 2–3 tahun + sanitasi alat & lahan; hindari bekerja saat tanaman basah.",
            "Jaga pH tanah 6.0–6.8, pemupukan seimbang (hindari N berlebih).",
            "Monitoring hama mingguan (cek bawah daun); kendalikan bila ambang terlampaui."
        ],
        "image": "Healthy.JPG",
        "source": "-"
    },

    "Bacterial Spot": {
        "desc": "Disebabkan oleh bakteri *Xanthomonas spp.* Bercak kecil berair lalu nekrotik pada daun; bercak kasar pada buah.",
        "handling": "Benih/transplan sehat, rotasi, hindari percikan. Semprot protektan berbasis tembaga + mankozeb sesuai label.",
        "source": "https://plantvillage.psu.edu/topics/tomato",
        "image": "Bacterial Spot.JPG"
    },

    "Early Blight": {
        "desc": "Jamur *Alternaria solani*. Bercak cokelat dengan lingkaran konsentris pada daun tua → defoliasi.",
        "handling": "Sanitasi & rotasi. Protektan/sistemik (mis. klorotalonil/mankozeb) sejak dini & terjadwal.",
        "source": "https://extension.umn.edu/diseases/early-blight-tomato",
        "image": "Early Blight.JPG"
    },

    "Late Blight": {
        "desc": "Oomycete *Phytophthora infestans*. Bercak berminyak cepat meluas; sporulasi putih di tepi bawah daun.",
        "handling": "Monitoring risiko; protektan + anti-oomycete sesuai label; eradikasi tanaman sangat terinfeksi.",
        "source": "https://www.rhs.org.uk/disease/late-blight",
        "image": "Late Blight.JPG"
    },

    "Leaf Mold": {
        "desc": "Jamur *Passalora fulva*. Bercak kuning di atas daun; lapisan jamur zaitun di bawah daun (kelembapan tinggi).",
        "handling": "Turunkan kelembapan/tingkatkan ventilasi; protektan (tembaga/klorotalonil) bila perlu.",
        "source": "https://extension.psu.edu/tomato-leaf-mold",
        "image": "Leaf Mold.JPG"
    },

    "Septoria Leaf Spot": {
        "desc": "Jamur *Septoria lycopersici*. Bercak kecil banyak berpusat pucat dengan titik hitam (pycnidia).",
        "handling": "Mulsa/rotasi; semprot protektan (klorotalonil/mankozeb/tembaga) preventif.",
        "source": "https://extension.umn.edu/diseases/septoria-leaf-spot",
        "image": "Septoria Leaf Spot.JPG"
    },

    "Spider Mites": {
        "desc": "Tungau laba-laba *Tetranychus urticae*. Stippling, bronzing, jaring halus di bawah daun.",
        "handling": "Semprot air kuat/sabun insektisida/minyak hortikultura; mitisida selektif; pelihara predator alami.",
        "source": "https://ipm.ucanr.edu/PMG/PESTNOTES/pn7405.html",
        "image": "Spider Mites.JPG"
    },

    "Target Spot": {
        "desc": "Jamur *Corynespora cassiicola*. Lesi bertarget (cincin konsentris) → defoliasi kanopi bagian dalam.",
        "handling": "Rotasi & aerasi kanopi; protektan (klorotalonil/tembaga/mankozeb) interval 10–14 hari.",
        "source": "https://www.daf.qld.gov.au/business-priorities/plants",
        "image": "Target Spot.JPG"
    },

    "Tomato Yellow Leaf Curl Virus (TYLCV)": {
        "desc": "Begomovirus vektor kutu kebul (*Bemisia tabaci*). Daun menguning & menggulung ke atas, tanaman kerdil.",
        "handling": "Bibit bebas virus; varietas tahan; kendalikan vektor (mulsa reflektif, sabun/insektisida selektif, sanitasi gulma).",
        "source": "https://www.plantwise.org/knowledgebank",
        "image": "Tomato Yellow Leaf Curl Virus (TYLCV).JPG"
    },

    "Tomato Mosaic Virus (TMV)": {
        "desc": "Virus mozaik; penularan mekanis & benih. Mozaik hijau–kuning; daun menyempit/keriting.",
        "handling": "Tidak ada kuratif. Cabut tanaman sakit; disinfeksi alat & tangan; benih bersertifikat/terdisinfeksi.",
        "source": "https://www.rhs.org.uk/disease/tobacco-mosaic-virus",
        "image": "Tomato Mosaic Virus (TMV).JPG"
    },
}

# -------------------------- Helpers --------------------------
def render_numbered(title: str, items):
    """Tampilkan list bernomor turun ke bawah (vertical)."""
    st.markdown(f"**{title}**")
    if isinstance(items, (list, tuple)):
        st.markdown("\n".join([f"{i}. {text}" for i, text in enumerate(items, start=1)]))
    else:
        st.markdown(f"{items}")

def render_section(name: str, data: dict):
    st.subheader(name)
    img_path = data.get("image")
    if img_path:
        try:
            st.image(f"images/{img_path}", width=240)
        except Exception:
            # Lewati bila file tidak ada; tetap render teks
            pass

    render_numbered("Penyebab & Gejala:", data.get("desc", "-"))
    render_numbered("Penanganan:", data.get("handling", "-"))

    src = data.get("source", "-")
    if src and src != "-":
        st.markdown(f"[Sumber]({src})")
    st.divider()

# -------------------------- UI --------------------------
st.title("🩺 Informasi Penyakit Tomat")

# Tampilkan semua penyakit sesuai urutan didefinisikan (Python >=3.7 menjaga insertion order)
for name, data in diseases.items():
    render_section(name, data)

st.info(
    "Sebagai Catatan: Ini adalah alat diagnosis dengan bantuan Kecerdasan Buatan dan sebaiknya digunakan hanya sebagai panduan. Untuk diagnosis konklusif, konsultasikan dengan ahli patologi tanaman profesional."
)

st.markdown(
    """
<div style='text-align:center; font-size:14px;'>
<b>© 2025 | Muhammad Sahrul Farhan | 51421076</b><br>
🔗 <a href="https://www.linkedin.com/in/muhammad-sahrul-farhan/" target="_blank">LinkedIn</a> |
<a href="https://www.instagram.com/eitcheien/" target="_blank">Instagram</a> |
<a href="https://www.facebook.com/skywalkr12" target="_blank">Facebook</a>
</div>
""",
    unsafe_allow_html=True,
)



