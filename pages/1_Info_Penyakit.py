import streamlit as st
import base64
from pathlib import Path
import html # <-- TAMBAHAN PENTING untuk membersihkan teks

st.set_page_config(page_title="🩺 Informasi Penyakit Tanaman Tomat", layout="centered")

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
        "image": "Healthy.jpg",
    },

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
        "image": "Spider Mites.jpg",
    },

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
        "image": "Leaf Mold.jpg",
    },

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
        "image": "Septoria Leaf Spot.jpg",
    },

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
    },

    "Bacterial Spot": {
        "desc": [
            "Bercak kecil berair → nekrotik pada daun; tepi kuning. Pada buah: bercak kasar/berlekuk.",
            "Sangat rawan di kondisi hangat–lembab."
        ],
        "handling": [
            "Pencegahan: benih/transplan sehat, rotasi, hindari percikan & bekerja saat tanaman basah.",
            "Jika muncul: semprot tembaga + mankozeb sebagai protektan (efektivitas bervariasi; waspadai resistensi)."
        ],
        "severity": "Sedang → Tinggi",
        "image": "Bacterial Spot.jpg",
    },

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
        "image": "Target Spot.jpg",
    },

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
        "image": "Early Blight.jpg",
    },

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
        "image": "Late Blight.jpg",
    },

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
        "image": "Tomato Yellow Leaf Curl Virus (TYLCV).jpg",
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
def image_to_base64(image_path):
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except Exception:
        return None

def render_section(name: str, data: dict):
    # 1. Siapkan semua konten dan BERSIHKAN (escape) teksnya
    severity = html.escape(data.get("severity", "—"))
    clean_name = html.escape(name)
    
    img_path = data.get("image")
    image_html = ""
    if img_path:
        # Asumsikan folder 'images' ada di direktori yang sama dengan skrip
        full_image_path = Path("images") / img_path
        base64_image = image_to_base64(full_image_path)
        if base64_image:
            image_html = f'<div style="text-align: center; margin-bottom: 20px;"><img src="data:image/jpeg;base64,{base64_image}" style="width: 300px; max-width: 100%; border-radius: 5px;"></div>'

    desc_items = data.get("desc", [])
    # PERBAIKAN: Gunakan html.escape() pada setiap item
    desc_html = "<ol>" + "".join([f"<li>{html.escape(item)}</li>" for item in desc_items]) + "</ol>"

    handling_items = data.get("handling", [])
    # PERBAIKAN: Gunakan html.escape() pada setiap item
    handling_html = "<ol>" + "".join([f"<li>{html.escape(item)}</li>" for item in handling_items]) + "</ol>"

    # 2. Gabungkan semua menjadi satu string HTML besar
    full_html = f"""
    <div style="
        background: linear-gradient(to right, #FFFFFF, #E0F2F1);
        border: 1px solid #CCCCCC;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 25px;
        color: #000000;
        font-family: sans-serif;
    ">
        <h3>{clean_name}</h3>
        <p style="font-size: 0.9em; color: #555; margin-top: -10px;">Tingkat keparahan (lokal): {severity}</p>
        
        {image_html}
        
        <b>Ciri-ciri/Gejala & Catatan:</b>
        {desc_html}
        
        <b>Pencegahan & Penanganan:</b>
        {handling_html}
    </div>
    """
    st.markdown(full_html, unsafe_allow_html=True)

# =========================
# UI
# =========================
st.title("🩺 Informasi Penyakit Tanaman Tomat (Beserta Tingkat Keparahan)")
st.markdown("---")

for key in ordered_keys:
    if key in diseases:
        render_section(key, diseases[key])

st.info( "Perlu diingat: Ini adalah alat diagnosis dengan bantuan Kecerdasan Buatan dan sebaiknya digunakan hanya sebagai panduan. Untuk diagnosis konklusif, konsultasikan dengan ahli patologi tanaman profesional."
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
