import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
from model_utils import load_model, predict_large_image_smooth, save_visual_report

# Configuration de la page
st.set_page_config(page_title="Analyse Drone - Segmentation", layout="wide", page_icon="Base_icon.png")

# --- BARRE LATÉRALE (GUIDE) ---
with st.sidebar:
    st.header("📖 Guide d'utilisation")
    st.info("""
    1. **Charger** une image drone (haute résolution).
    2. **Cliquer** sur 'Lancer l'analyse'.
    3. **Attendre** la fin du traitement (découpage en patchs).
    4. **Télécharger** le rapport final au format JPG.
    """)
    st.divider()
    st.markdown("### 🛠️ Paramètres")
    gsd = st.slider("Résolution (GSD en cm/px)", 1.0, 10.0, 3.0)
    st.caption("Le GSD permet de calculer les surfaces exactes en m².")

# --- CORPS DE L'APPLICATION ---
st.title("🏠 Détection Automatique des éléments d'occupation du sol")
st.write("Cet outil utilise un réseau de neurones **DeepLabV3+** pour segmenter les images aériennes.")

# 1. Chargement du modèle
@st.cache_resource
def get_model():
    return load_model("best_model.pth")

model = get_model()

# 2. Interface de téléchargement
uploaded_file = st.file_uploader("Choisir une image drone (JPG/PNG)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Sauvegarde temporaire
    with open("temp_input.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🖼️ Image Originale")
        st.image(uploaded_file, use_container_width=True)

    if st.button("🚀 Lancer l'analyse"):
        with st.spinner('Analyse Deep Learning en cours...'):
            # 3. Exécution de l'inférence
            mask = predict_large_image_smooth(model, "temp_input.jpg", "temp_output.jpg")
            save_visual_report("temp_input.jpg", mask, "report.jpg", gsd_cm=gsd)
            
            with col2:
                st.subheader("📊 Résultat de Segmentation")
                st.image("report.jpg", use_container_width=True)

        # 4. Bouton de téléchargement
        with open("report.jpg", "rb") as file:
            st.download_button(
                label="📥 Télécharger le rapport complet (JPG)",
                data=file,
                file_name=f"rapport_{uploaded_file.name}",
                mime="image/jpg"
            )

# --- PIED DE PAGE ---
st.divider()
st.markdown(
    """
    <div style="text-align: center; color: grey;">
        <p>Développé par <strong>Hubert SALANON</strong> |Télédétection & IA</p>
    </div>
    """, 
    unsafe_allow_html=True
)
