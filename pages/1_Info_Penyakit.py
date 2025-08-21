import streamlit as st

st.set_page_config(page_title="🩺 Informasi Penyakit Tomat", layout="centered")

# =========================
# DATA (Healthy: biarkan/ubah sesuai punyamu)
# =========================
diseases = {
    "Healthy": {
        "desc": [
            "Warna hijau merata (tidak mosaik/klorosis).",
            "Tidak ada bercak/lesi.",
            "Tidak menggulung/mengeriting; turgor baik (tidak layu).",
            "Tepi daun utuh (tidak sobek/nekrotik).",
            "Bagian bawah daun bersih—tidak ada lapisan jamur/jaring tungau/honeydew.",
            "Urat daun normal (tidak menebal/menonjol).",
            "Tunas baru segar dan simetris."
        ],
        "handling": [
            "Siram di pangkal (drip) pada pagi hari; jangan membasahi daun.",
            "Gunakan mulsa (organik/plastik) untuk mengurangi percikan tanah & stabilkan kelembapan.",
            "Atur jarak tanam; ajir/trellis agar kanopi berangin; pangkas daun bawah yang menyentuh tanah.",
            "Rotasi 2–3 tahun + sanitasi alat & lahan; hindari bekerja saat tanaman basah.",
            "Jaga pH tanah 6.0–6.8, pemupukan seimbang (hindari N berlebih).",
            "Monitoring hama mingguan (cek bawah daun); kendalikan bila ambang terlampaui."
        ],
        "severity": "—",
        "image": "Healthy.JPG",
        "sources": []
    },

    # ---------- 1) Spider mites — RENDAH → SEDANG ----------
    "Spider Mites": {
        "desc": [
            "Bintik keperakan (stippling) & bronzing pada daun, kadang kecokelatan.",
            "Jaring halus di bawah daun; populasi meledak saat panas–kering."
        ],
        "handling": [
            "Pencegahan: kurangi stres kekeringan & debu; jaga musuh alami; hindari insektisida spektrum luas.",
            "Jika sudah ada: semprot air kuat ke bawah daun; sabun insektisida/minyak hortikultura; mitisida selektif atau predator (mis. *Phytoseiulus*)."
        ],
        "severity": "Rendah → Sedang",
        "image": "Spider Mites.JPG",
        "sources": [
            "https://ipm.ucanr.edu/PMG/PESTNOTES/pn7405.html",
            "https://extension.umn.edu/yard-and-garden-insects/spider-mites"
        ]
    },

    # ---------- 2) Leaf mold — RENDAH → SEDANG ----------
    "Leaf Mold": {
        "desc": [
            "Bercak kuning di atas daun; bawah daun berlapis jamur zaitun/kehijauan (beludru).",
            "Parah pada RH >85% (rumah-tanam) dan kanopi lembap."
        ],
        "handling": [
            "Pencegahan: ventilasi/kurangi RH, jarak tanam & siram di pangkal; varietas toleran.",
            "Jika muncul: buang daun terinfeksi; fungisida protektan (klorotalonil/mankozeb/tembaga) sesuai label."
        ],
        "severity": "Rendah → Sedang",
        "image": "Leaf Mold.JPG",
        "sources": [
            "https://extension.psu.edu/tomato-leaf-mold",
            "https://vegetablemdonline.ppath.cornell.edu/factsheets/Tomato_LeafMold.htm"
        ]
    },

    # ---------- 3) Septoria leaf spot — SEDANG ----------
    "Septoria Leaf Spot": {
        "desc": [
            "Banyak bercak kecil (±1–3 mm) berpusat pucat dengan titik hitam (pycnidia) terutama di daun bawah.",
            "Dapat menyebabkan defoliasi berat jika dibiarkan."
        ],
        "handling": [
            "Pencegahan: rotasi 2–3 th, mulsa untuk cegah percikan, sanitasi sisa tanaman, siram pangkal.",
            "Jika muncul: protektan (klorotalonil/mankozeb/tembaga) terjadwal; buang daun bawah terinfeksi."
        ],
        "severity": "Sedang",
        "image": "Septoria Leaf Spot.JPG",
        "sources": [
            "https://extension.umn.edu/diseases/septoria-leaf-spot",
            "https://extension.psu.edu/septoria-leaf-spot-of-tomato"
        ]
    },

    # ---------- 4) TMV/ToMV — SEDANG → TINGGI ----------
    "Tomato Mosaic Virus (TMV)": {
        "desc": [
            "Mozaik hijau–kuning, daun menyempit/keriting, tanaman kerdil.",
            "Penularan utama: mekanis & benih/permukaan alat (sangat stabil)."
        ],
        "handling": [
            "Pencegahan: benih/bibit bebas virus atau perlakuan benih; disinfeksi alat & tangan; higienitas tinggi.",
            "Jika sudah ada: cabut tanaman sakit; cegah penyebaran (tidak ada kuratif)."
        ],
        "severity": "Sedang → Tinggi",
        "image": "Tomato Mosaic Virus (TMV).JPG",
        "sources": [
            "https://www.rhs.org.uk/disease/tobacco-mosaic-virus",
            "https://apsjournals.apsnet.org/doi/full/10.1094/PDIS-91-12-1513A"
        ]
    },

    # ---------- 5) Bacterial spot — SEDANG → TINGGI ----------
    "Bacterial Spot": {
        "desc": [
            "Bercak kecil berair → nekrotik pada daun; tepi kuning. Pada buah: bercak kasar/berlekuk.",
            "Sangat cocok kondisi hangat–basah."
        ],
        "handling": [
            "Pencegahan: benih/transplan sehat, rotasi, hindari percikan & bekerja saat tanaman basah.",
            "Jika muncul: semprot tembaga + mankozeb sebagai protektan (efektivitas bervariasi; waspadai resistensi)."
        ],
        "severity": "Sedang → Tinggi",
        "image": "Bacterial Spot.JPG",
        "sources": [
            "https://edis.ifas.ufl.edu/publication/pp121",
            "https://content.ces.ncsu.edu/bacterial-spot-of-tomato"
        ]
    },

    # ---------- 6) Target spot — TINGGI ----------
    "Target Spot": {
        "desc": [
            "Lesi bertarget (cincin konsentris) dengan pusat keabu-abuan; defoliasi terutama di kanopi bagian dalam.",
            "Dapat mengenai buah (lesi berlekuk)."
        ],
        "handling": [
            "Pencegahan: rotasi, aerasi kanopi baik; hindari kelembapan berlebih pada daun.",
            "Jika muncul: protektan (klorotalonil/tembaga/mankozeb) interval 10–14 hari sesuai label."
        ],
        "severity": "Tinggi",
        "image": "Target Spot.JPG",
        "sources": [
            "https://www.daf.qld.gov.au/__data/assets/pdf_file/0005/1526256/target-spot-of-tomato.pdf",
            "https://projects.sare.org/wp-content/uploads/Target-Spot-of-Tomato-UF-IFAS.pdf"
        ]
    },

    # ---------- 7) Early blight — TINGGI ----------
    "Early Blight": {
        "desc": [
            "Bercak cokelat ‘bullseye’ (lingkar konsentris) pada daun tua → defoliasi; dapat ke batang/buah.",
            "Berkembang pada kelembapan daun dan percikan tanah."
        ],
        "handling": [
            "Pencegahan: rotasi 2–3 th, buang sisa tanaman, pemupukan seimbang; siram pangkal & mulsa.",
            "Jika muncul: protektan/sistemik sejak dini dan terjadwal (mis. klorotalonil/mankozeb sesuai label)."
        ],
        "severity": "Tinggi",
        "image": "Early Blight.JPG",
        "sources": [
            "https://extension.umn.edu/diseases/early-blight-tomato",
            "https://vegetablemdonline.ppath.cornell.edu/factsheets/Tomato_EarlyBlight.htm"
        ]
    },

    # ---------- 8) Late blight — SANGAT TINGGI ----------
    "Late Blight": {
        "desc": [
            "Bercak berminyak cepat meluas; tepi bawah daun bersporulasi putih; menyerang daun, batang, buah.",
            "Sangat cepat pada cuaca sejuk–basah (dataran tinggi, musim hujan sejuk)."
        ],
        "handling": [
            "Pencegahan: sumber inokulum bersih, monitoring kondisi cuaca; protektan + anti-oomycete sesuai label.",
            "Jika parah: singkirkan tanaman sangat terinfeksi untuk memutus siklus."
        ],
        "severity": "Sangat Tinggi",
        "image": "Late Blight.JPG",
        "sources": [
            "https://usablight.org/educate/",
            "https://www.rhs.org.uk/disease/late-blight"
        ]
    },

    # ---------- 9) TYLCV — EKSTREM ----------
    "Tomato Yellow Leaf Curl Virus (TYLCV)": {
        "desc": [
            "Daun kecil menguning & menggulung ke atas; tanaman kerdil; gugur bunga → kehilangan hasil besar.",
            "Vektor: kutu kebul *Bemisia tabaci*; **tidak terbawa benih**."
        ],
        "handling": [
            "Pencegahan: varietas tahan; pengelolaan vektor (mulsa reflektif, sabun/insektisida selektif, sanitasi gulma inang).",
            "Jika muncul: cabut tanaman terinfeksi dini; fokus pada kontrol vektor & kebersihan persemaian."
        ],
        "severity": "Ekstrem",
        "image": "Tomato Yellow Leaf Curl Virus (TYLCV).JPG",
        "sources": [
            "https://www.plantwise.org/knowledgebank/",
            "https://www.cabi.org/isc/datasheet/56679",
            "https://edis.ifas.ufl.edu/publication/IN716"
        ]
    }
}

# =========================
# ORDER (Least -> Most severe; Healthy ditaruh di atas)
# =========================
ordered_keys = [
    "Healthy",
    "Spider Mites",
    "Leaf Mold",
    "Septoria Leaf Spot",
    "Tomato Mosaic Virus (TMV)",
    "Bacterial Spot",
    "Target Spot",
    "Early Blight",
    "Late Blight",
    "Tomato Yellow Leaf Curl Virus (TYLCV)"
]

# =========================
# HELPERS
# =========================
def render_numbered(title: str, items):
    st.markdown(f"**{title}**")
    if isinstance(items, (list, tuple)):
        st.markdown("\n".join([f"{i}. {text}" for i, text in enumerate(items, start=1)]))
    else:
        # fallback untuk string
        st.markdown(items)

def render_sources(srcs):
    if not srcs:
        return
    st.markdown("**Sumber:**")
    if isinstance(srcs, str):
        st.markdown(f"- [{srcs}]({srcs})")
    else:
        for s in srcs:
            st.markdown(f"- [{s}]({s})")

def render_section(name: str, data: dict):
    st.subheader(name)
    sev = data.get("severity", "")
    if sev:
        st.caption(f"Tingkat keparahan (lokal): {sev}")
    img_path = data.get("image")
    if img_path:
        try:
            st.image(f"images/{img_path}", width=260)
        except Exception:
            pass
    render_numbered("Ciri-ciri/Gejala & Catatan:", data.get("desc", "-"))
    render_numbered("Pencegahan & Penanganan:", data.get("handling", "-"))
    render_sources(data.get("sources", []))
    st.divider()

# =========================
# UI
# =========================
st.title("🩺 Informasi Penyakit Tomat — Urutan Keparahan (ID)")

for key in ordered_keys:
    if key in diseases:
        render_section(key, diseases[key])

st.info("Catatan: Ini alat bantu informasi. Untuk diagnosis konklusif, konsultasikan dengan ahli patologi tanaman/UPT proteksi setempat.")

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
