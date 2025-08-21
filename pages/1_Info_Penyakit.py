import streamlit as st

st.set_page_config(page_title="🩺 Informasi Penyakit Tomat", layout="centered")

# -------------------------- Data --------------------------
# Catatan: untuk tampilan vertikal rapi, "Healthy" memakai list.
# Item lain boleh string biasa; renderer di bawah akan menangani keduanya.
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
        "image": "Healthy.JPG",
        "source": "-"
    },

    "Bacterial Spot": {
        "desc": "Disebabkan oleh bakteri *Xant*
