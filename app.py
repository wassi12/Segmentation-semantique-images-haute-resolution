import streamlit as st
import torch
import cv2
import numpy as np
import os
from model_utils import load_model, predict_large_image_smooth, colorize_mask, CLASS_MAP

# Configuration
st.set_page_config(page_title="Analyse Drone - Segmentation", layout="wide")

# --- MÉMOIRE ---
if 'segmentation_done' not in st.session_state:
    st.session_state.segmentation_done = False
if 'last_mask' not in st.session_state:
    st.session_state.last_mask = None

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Paramètres")
    gsd = st.slider("Résolution (GSD en cm/px)", 1.0, 10.0, 3.0, step=0.1)
    st.info("Le GSD modifie le calcul des surfaces (m²) sans changer le dessin.")

st.title("🏠 Analyse Drone & Segmentation IA")

@st.cache_resource
def get_model():
    return load_model("best_model.pth")

model = get_model()
uploaded_file = st.file_uploader("Charger une image...", type=["jpg", "png"])

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

        # --- NOUVELLE SECTION : RÉSULTATS EN BAS ---
        st.divider()
        st.subheader("📈 Rapport des Surfaces (GSD : {} cm/px)".format(gsd))
        
        # Calcul des surfaces
        pixel_area_m2 = (gsd / 100) ** 2
        cols = st.columns(len(CLASS_MAP)) # Une colonne par classe
        
        for i, (class_id, info) in enumerate(CLASS_MAP.items()):
            pixels = np.sum(mask == class_id)
            area = pixels * pixel_area_m2
            
            with cols[i]:
                # Petit carré de couleur HTML
                color_hex = '#%02x%02x%02x' % tuple(info['color'])
                st.markdown(f"<div style='width:20px;height:20px;background:{color_hex};border:1px solid black;display:inline-block'></div> **{info['name']}**", unsafe_allow_html=True)
                st.metric(label="Surface", value=f"{area:.2f} m²")

        # Bouton Téléchargement (on garde la sauvegarde image pour le rapport)
        from model_utils import save_visual_report
        save_visual_report("temp_input.jpg", mask, "report.jpg", gsd_cm=gsd)
        with open("report.jpg", "rb") as f:
            st.download_button("📥 Télécharger le rapport Image", f, "rapport.jpg", "image/jpeg")

st.markdown("---")
st.markdown("<p style='text-align: center;'>Développé par <b>Wassi</b> | Télédétection & IA</p>", unsafe_allow_html=True)
