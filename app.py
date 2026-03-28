import streamlit as st
import torch
import cv2
import numpy as np
import os
from model_utils import load_model, predict_large_image_smooth, colorize_mask, CLASS_MAP

# Configuration de la page
st.set_page_config(page_title="Analyse Drone - Segmentation", layout="wide")

# --- MÉMOIRE (SESSION STATE) ---
if 'segmentation_done' not in st.session_state:
    st.session_state.segmentation_done = False
if 'last_mask' not in st.session_state:
    st.session_state.last_mask = None

# --- BARRE LATÉRALE (GUIDE & PARAMÈTRES) ---
with st.sidebar:
    st.header("📖 Guide d'utilisation")
    st.info("""
    1. **Charger** une image drone.
    2. **Cliquer** sur 'Lancer l'analyse'.
    3. **Ajuster** le GSD (le calcul s'adapte).
    4. **Télécharger** le rapport final.
    """)
    st.divider()
    st.header("⚙️ Paramètres")
    gsd = st.slider("Résolution (GSD en cm/px)", 1.0, 10.0, 3.0, step=0.1)
    st.caption("Le GSD modifie le calcul des surfaces (m²) sans changer le dessin.")

st.title("🏠 Analyse Drone & Segmentation IA")

@st.cache_resource
def get_model():
    return load_model("best_model.pth")

model = get_model()
uploaded_file = st.file_uploader("Charger une image...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    with open("temp_input.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("🖼️ Originale")
        st.image(uploaded_file, use_container_width=True)
    
    if st.button("🚀 Lancer l'analyse"):
        with st.spinner('Analyse en cours...'):
            mask = predict_large_image_smooth(model, "temp_input.jpg", "temp_output.jpg")
            st.session_state.last_mask = mask
            st.session_state.segmentation_done = True

    if st.session_state.segmentation_done:
        mask = st.session_state.last_mask
        with c2:
            st.subheader("📊 Segmentation")
            color_mask = colorize_mask(mask)
            st.image(color_mask, use_container_width=True)

        # --- SECTION RÉSULTATS (Formaté pour éviter les pointillés) ---
        st.divider()
        st.subheader(f"📈 Rapport des Surfaces (GSD : {gsd} cm/px)")
        
        pixel_area_m2 = (gsd / 100) ** 2
        
        # On crée deux lignes de 3 colonnes pour avoir plus d'espace
        rows = [st.columns(3), st.columns(3)]
        classes = list(CLASS_MAP.items())

        for i, (class_id, info) in enumerate(classes):
            pixels = np.sum(mask == class_id)
            area = pixels * pixel_area_m2
            
            # Sélection de la ligne (0 ou 1) et de la colonne (0, 1 ou 2)
            target_row = i // 3
            target_col = i % 3
            
            with rows[target_row][target_col]:
                color_hex = '#%02x%02x%02x' % tuple(info['color'])
                # Texte propre sans étoiles
                st.markdown(f"<div style='display: flex; align-items: center;'><div style='width:15px;height:15px;background:{color_hex};margin-right:10px;border:1px solid #ddd'></div><span style='font-weight: bold;'>{info['name']}</span></div>", unsafe_allow_html=True)
                st.metric(label="Surface calculée", value=f"{area:.2f} m²")

        # Bouton Téléchargement
        from model_utils import save_visual_report
        save_visual_report("temp_input.jpg", mask, "report.jpg", gsd_cm=gsd)
        with open("report.jpg", "rb") as f:
            st.download_button("📥 Télécharger le rapport Image", f, "rapport.jpg", "image/jpeg")

st.markdown("---")
st.markdown("<p style='text-align: center;'>Développé par <b>Wassi</b> | Télédétection & IA</p>", unsafe_allow_html=True)
