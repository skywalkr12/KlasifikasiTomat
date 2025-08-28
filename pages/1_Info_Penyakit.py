import streamlit as st

st.set_page_config(page_title="🩺 Informasi Penyakit Tanaman Tomat", layout="wide")

# ============================================================
# Daftar penyakit daun tomat tersusun dari yang paling bisa
# dipulihkan tanpa pencabutan → hingga yang wajib eradikasi.
# Rasional singkat (ringkas & akademik):
# - Kelompok “dikelola tanpa cabut”: Spider Mites, Leaf Mold,
#   Septoria Leaf Spot, Target Spot, Early Blight.
# - “Cabut selektif (kontekstual)”: Bacterial Spot—terutama
#   pada pembibitan atau saat penyebaran luas.
# - “Wajib eradikasi”: TMV/ToMV, TYLCV, Late Blight.
# ============================================================

diseases = {
    "Healthy": {
        "info": [
            "Tanaman sehat menunjukkan daun hijau merata, tanpa bercak, dan kanopi berangin. Praktik dasar seperti mengurangi kebasahan daun, rotasi ≥2 tahun, dan inspeksi rutin memang terbukti menurunkan risiko penyakit utama pada tomat."
        ],
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

    # ---------- 1) Spider mites — RENDAH → SEDANG ----------
    "Spider Mites": {
        "desc": [
            "Bintik keperakan (stippling) & bronzing pada daun, kadang kecokelatan.",
            "Jaring halus di bawah daun; populasi meningkat pada kondisi panas–kering/berdebu."
        ],
        "handling": [
            "Pencegahan: kurangi stres kekeringan & debu; pelihara musuh alami; hindari insektisida spektrum luas.",
            "Jika sudah ada: semprot air kuat ke bawah daun; sabun insektisida/minyak hortikultura; mitisida selektif atau predator (mis. *Phytoseiulus*)."
        ],
        "severity": "Rendah → Sedang",
        "image": "Spider Mites.jpg",
        "sources": "https://www.vegetables.cornell.edu/pest-management/disease-factsheets/managing-tomato-diseases-successfully"
    },

    # ---------- 2) Leaf mold — RENDAH → SEDANG ----------
    "Leaf Mold": {
        "desc": [
            "Bercak kuning di atas daun; bawah daun berlapis jamur zaitun/kehijauan (beludru).",
            "Dominan pada RH >85% (greenhouse/kanopi lembap)."
        ],
        "handling": [
            "Pencegahan: ventilasi/kurangi RH, jarak tanam & siram di pangkal; gunakan varietas toleran jika tersedia.",
            "Jika muncul: buang daun terinfeksi; fungisida protektan (klorotalonil/mankozeb/tembaga) sesuai label."
        ],
        "severity": "Rendah → Sedang",
        "image": "Leaf Mold.jpg",
    },

    # ---------- 3) Septoria leaf spot — SEDANG ----------
    "Septoria Leaf Spot": {
        "desc": [
            "Bercak kecil banyak (±1–3 mm) berpusat pucat dengan titik hitam (pycnidia), terutama di daun bawah.",
            "Berpotensi defoliasi progresif bila tidak dikendalikan."
        ],
        "handling": [
            "Pencegahan: rotasi 2–3 th, mulsa untuk cegah percikan, sanitasi sisa tanaman, siram pangkal.",
            "Jika muncul: protektan (klorotalonil/mankozeb/tembaga) terjadwal; buang selektif daun bawah terinfeksi (hindari memangkas >⅓ tajuk)."
        ],
        "severity": "Sedang",
        "image": "Septoria Leaf Spot.jpg",
    },

    # ---------- 4) Target spot — TINGGI ----------
    "Target Spot": {
        "desc": [
            "Lesi bertarget (cincin konsentris) dengan pusat keabu-abuan; defoliasi di kanopi bagian dalam.",
            "Dapat mengenai buah (lesi berlekuk)."
        ],
        "handling": [
            "Pencegahan: rotasi, aerasi kanopi baik; minimalkan kelembapan daun.",
            "Jika muncul: protektan (klorotalonil/tembaga/mankozeb) interval 10–14 hari sesuai label."
        ],
        "severity": "Tinggi",
        "image": "Target Spot.jpg",
    },

    # ---------- 5) Early blight — TINGGI ----------
    "Early Blight": {
        "desc": [
            "Bercak cokelat dengan lingkar konsentris ‘bullseye’ pada daun tua → defoliasi; dapat menjalar ke batang/buah.",
            "Dipicu kelembapan daun & percikan tanah."
        ],
        "handling": [
            "Pencegahan: rotasi 2–3 th, buang sisa tanaman, pemupukan seimbang; siram pangkal & mulsa.",
            "Jika muncul: protektan/sistemik sejak dini dan terjadwal (mis. klorotalonil/mankozeb sesuai label; interval 7–10 hari saat tekanan tinggi)."
        ],
        "severity": "Tinggi",
        "image": "Early Blight.jpg",
    },

    # ---------- 6) Bacterial spot — SEDANG → TINGGI (CABUT SELEKTIF KONTEKSTUAL) ----------
    "Bacterial Spot": {
        "desc": [
            "Bercak kecil berair → nekrotik pada daun; tepi klorotik. Pada buah: bercak kasar/berlekuk.",
            "Sangat rawan di kondisi hangat–lembap; tidak ada kuratif spesifik."
        ],
        "handling": [
            "Pencegahan: benih/transplan sehat, rotasi, hindari percikan & bekerja saat tanaman basah; sanitasi intensif.",
            "Protektan: tembaga ± mankozeb sebagai pencegah (efektivitas bervariasi; waspadai resistensi).",
            "Cabut selektif: transplan/bibit bergejala atau bila penyebaran luas untuk memutus sumber inokulum."
        ],
        "severity": "Sedang → Tinggi (cabut selektif pada pembibitan/penyebaran luas)",
        "image": "Bacterial Spot.jpg",
    },

    # ---------- 7) TMV/ToMV — TINGGI (ERADIKASI DIANJURKAN) ----------
    "Tomato Mosaic Virus (TMV)": {
        "desc": [
            "Mozaik hijau–kuning, daun menyempit/keriting, tanaman kerdil.",
            "Penularan utama: mekanis & benih/permukaan alat (stabil dan menular tinggi)."
        ],
        "handling": [
            "Pencegahan: benih/bibit bebas virus atau perlakuan benih; disinfeksi alat & tangan; higienitas tinggi.",
            "Jika sudah ada: cabut & musnahkan tanaman sakit; cegah penyebaran (tidak ada kuratif)."
        ],
        "severity": "Tinggi (eradikasi/rogueing dianjurkan)",
        "image": "Tomato Mosaic Virus (TMV).JPG",
    },

    # ---------- 8) TYLCV — SANGAT TINGGI → EKSTREM ----------
    "Tomato Yellow Leaf Curl Virus (TYLCV)": {
        "desc": [
            "Daun kecil menguning & menggulung ke atas; tanaman kerdil; gugur bunga → kehilangan hasil signifikan.",
            "Vektor: kutu kebul *Bemisia tabaci*; tidak terbawa benih."
        ],
        "handling": [
            "Pencegahan: varietas tahan; pengelolaan vektor (mulsa reflektif, sabun/insektisida selektif, sanitasi gulma inang).",
            "Jika muncul: cabut tanaman terinfeksi sedini mungkin (bungkus plastik saat dicabut), fokus pada kontrol vektor & kebersihan persemaian."
        ],
        "severity": "Sangat Tinggi → Ekstrem (eradikasi cepat + kendali vektor)",
        "image": "Tomato Yellow Leaf Curl Virus (TYLCV).jpg",
    },

    # ---------- 9) Late blight — EKSTREM ----------
    "Late Blight": {
        "desc": [
            "Bercak berminyak cepat meluas; tepi bawah daun bersporulasi putih; menyerang daun, batang, hingga buah.",
            "Penyakit komunitas yang sangat cepat menyebar pada cuaca sejuk–basah."
        ],
        "handling": [
            "Pencegahan: sumber inokulum bersih, monitoring kondisi cuaca; protektan + anti-oomycete sesuai label.",
            "Eradikasi: singkirkan tanaman terinfeksi dan tanaman sekitar untuk memutus siklus (jangan dikompos)."
        ],
        "severity": "Ekstrem (eradikasi + penghapusan tanaman sekitar)",
        "image": "Late Blight.jpg",
    },
}

# =========================
# ORDER (Least -> Most severe; Healthy di atas)
# =========================
ordered_keys = [
    "Healthy",
    "Spider Mites",
    "Leaf Mold",
    "Septoria Leaf Spot",
    "Target Spot",
    "Early Blight",
    "Bacterial Spot",
    "Tomato Mosaic Virus (TMV)",
    "Tomato Yellow Leaf Curl Virus (TYLCV)",
    "Late Blight",
]

# ============================================================
# FEEDBACK LOOP (komentar untuk transparansi penalaran)
# - Asumsi utama: urutan ditentukan oleh dua sumbu:
#   (1) laju kerusakan & risiko epidemi, (2) ketegasan rekomendasi
#       tindakan (apakah cukup protektan/sanitasi vs. wajib cabut).
# - Potensi jebakan: salah diagnosis “leaf spots” (Septoria/Alternaria/
#   Target spot) dapat menggeser keputusan; verifikasi morfologi
#   (pycnidia, pola “bullseye”, dll.) penting.
# - Alternatif: Bacterial Spot bisa bergeser lebih tinggi jika terjadi
#   pada pembibitan intensif (karena eradikasi batch sering lebih
#   efektif daripada perawatan bertahap).
# - Cara verifikasi praktis:
#   a) Observasi 48–72 jam pada kasus dicurigai virus/late blight; cek
#      progres cepat atau sporulasi putih (late blight).
#   b) Terapkan disinfeksi alat/tangan (pemutih 1:9 ±1 menit) pada
#      setiap pemangkasan atau kontak jaringan untuk menekan transmisi
#      mekanis virus.
#   c) Dokumentasikan gejala (atas/bawah daun & buah) dan cocokkan
#      dengan panduan extension setempat sebelum mengeksekusi “cabut”.
# ============================================================

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
    render_text("Informasi Singkat:", data.get("info"))
    render_numbered("Ciri-ciri/Gejala & Catatan:", data.get("desc", "-"))
    render_numbered("Pencegahan & Penanganan:", data.get("handling", "-"))
    render_sources(data.get("sources", []))
    st.divider()

# =========================
# UI
# =========================
st.title("🩺 Informasi Penyakit Tanaman Tomat (Beserta Tingkat Keparahan)")

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






