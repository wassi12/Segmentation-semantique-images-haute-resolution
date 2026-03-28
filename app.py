import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
# Importe tes fonctions de ton fichier actuel
from model_utils import load_model, predict_large_image_smooth, save_visual_report

st.set_page_config(page_title="Analyse Drone - Segmentation", layout="wide")

st.title("🏠 Détection Automatique des éléments d'occupation du sol par Drone")
st.write("Téléchargez une image aérienne pour obtenir le rapport de segmentation et les surfaces.")

# 1. Chargement du modèle (mis en cache pour la vitesse)
@st.cache_resource
def get_model():
    return load_model("best_model.pth")

model = get_model()

# 2. Interface de téléchargement
uploaded_file = st.file_uploader("Choisir une image drone (JPG/PNG)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Sauvegarde temporaire du fichier
    with open("temp_input.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    col1, col2 = st.columns(2)
    
    with col1:
        #st.image(uploaded_file, caption="Image originale", use_column_width=True)
        st.image(uploaded_file, caption="Image originale", use_container_width=True)
    
    if st.button("Lancer l'analyse"):
        with st.spinner('Analyse en cours... cela peut prendre 1 à 2 minutes selon la taille.'):
            # 3. Exécution de ton code existant
            mask = predict_large_image_smooth(model, "temp_input.jpg", "temp_output.jpg")
            save_visual_report("temp_input.jpg", mask, "report.jpg", gsd_cm=3.0)
            
        with col2:
            st.image("report.jpg", caption="Résultat et Statistiques", use_column_width=True)
            
        # 4. Bouton de téléchargement du rapport
        with open("report.jpg", "rb") as file:
            st.download_button(
                label="📥 Télécharger le rapport complet",
                data=file,
                file_name="rapport_segmentation.jpg",
                mime="image/jpg"
            )
