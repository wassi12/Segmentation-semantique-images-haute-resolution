import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import segmentation_models_pytorch as smp
import albumentations as A
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device :", device)

IMAGE_DIR = "images_originales"
MASK_DIR = "images_labelisees"

PATCH_SIZE = 512
BATCH_SIZE = 4
EPOCHS = 10
N_CLASSES = 11


def load_model(checkpoint_path):

    print("Loading checkpoint:", checkpoint_path)

    # Reconstruction du modèle (doit être identique à celui de l'entraînement)
    model = smp.DeepLabV3Plus(
        encoder_name="resnet34",
        encoder_weights=None,
        classes=11,
        activation=None
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Si checkpoint complet
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        if "epoch" in checkpoint:
            print("Checkpoint epoch:", checkpoint["epoch"])
    else:
        # Si seulement state_dict
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    print("Model loaded successfully")

    return model



import matplotlib.patches as mpatches

# Configuration des classes réelles
CLASS_MAP = {
    0: {"name": "Background", "color": [0, 0, 0]},       # Noir
    1: {"name": "Végétation", "color": [34, 139, 34]},   # Vert
    3: {"name": "Sol nu", "color": [160, 82, 45]},       # Marron
    5: {"name": "Eau", "color": [0, 0, 255]},            # Bleu
    6: {"name": "Voiture", "color": [255, 255, 0]},      # Jaune
    10: {"name": "Bâtiment", "color": [255, 0, 0]}       # Rouge
}


def colorize_mask(mask):
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    
    for class_id, info in CLASS_MAP.items():
        color_mask[mask == class_id] = info["color"]
        
    return color_mask



#Fonction de prédiction
import torch.nn.functional as F

def predict_large_image_smooth (model , image_path , output_path , patch_size=512, stride=256) :
    print ("Predicting with smooth blending:" , image_path )
    image = cv2 . imread (image_path )
    image = cv2 . cvtColor (image , cv2 . COLOR_BGR2RGB )
    h , w = image . shape [: 2 ]

    # -- NOUVEAU : On initialise des matrices pour accumuler les probabilités --
    # Shape: (N_CLASSES, H, W)
    sum_probs = np.zeros((N_CLASSES, h, w), dtype=np.float32)
    # Compteur pour diviser et faire la moyenne
    count_map = np.zeros((h, w), dtype=np.float32)

    # 1. Inférence par patchs avec accumulation
    # Ajout de tqdm pour la barre de progression sur l'image entière
    pbar = tqdm(total=((h-patch_size)//stride + 1) * ((w-patch_size)//stride + 1), desc="Processing Patches")

    for y in range (0 , h - patch_size + 1, stride ) :
        for x in range (0 , w - patch_size + 1, stride ) :
            patch = image [y:y + patch_size , x:x + patch_size ]
            
            # Prétraitement (doit être identique à l'entraînement !)
            inp = torch . tensor (patch ) . permute (2 , 0 , 1 ) . float () / 255
            # Si tu as utilisé normalization ImageNet à l'entraînement, AJOUTE LA ICI
            # mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]
            # inp = (inp - torch.tensor(mean).view(3, 1, 1)) / torch.tensor(std).view(3, 1, 1)

            inp = inp . unsqueeze (0 ) . to (device )

            with torch . no_grad () :
                output = model (inp )
                # On convertit les logits en probabilités (Softmax)
                probs = F.softmax(output, dim=1).cpu().numpy()[0] # Shape (11, 512, 512)

            # Accumulation
            sum_probs[:, y:y + patch_size, x:x + patch_size] += probs
            count_map[y:y + patch_size, x:x + patch_size] += 1.0
            pbar.update(1)
    pbar.close()

    # 2. Moyennage et Argmax final
    # On évite la division par zéro
    count_map[count_map == 0] = 1.0
    # Shape: (N_CLASSES, H, W) / (1, H, W) -> Broadcasting
    final_probs = sum_probs / count_map

    # On prend la classe avec la probabilité moyenne la plus élevée
    result_labels = np.argmax(final_probs, axis=0).astype(np.uint8)

    # 3. Transformation en image colorée et sauvegarde (ton code existant)
    color_result = colorize_mask (result_labels )
    color_result_bgr = cv2 . cvtColor (color_result , cv2 . COLOR_RGB2BGR )
    cv2 . imwrite (output_path , color_result_bgr )
    
    print ("Smooth color prediction saved:" , output_path )
    return result_labels



import matplotlib.patches as mpatches

def show_prediction(image_path, pred_mask):
    # 1. Chargement et conversion de l'image originale
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 2. Colorisation du masque (utilise votre CLASS_MAP de la cellule [25])
    color_pred = colorize_mask(pred_mask)
    
    # 3. Création de l'affichage
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    
    ax[0].imshow(image)
    ax[0].set_title("Image originale")
    ax[0].axis("off")
    
    ax[1].imshow(color_pred)
    ax[1].set_title("Prédiction (6 classes réelles)")
    ax[1].axis("off")
    
    # 4. Génération de la légende avec les bonnes couleurs et noms
    legend_elements = [
        mpatches.Patch(color=np.array(info["color"])/255, label=info["name"])
        for class_id, info in CLASS_MAP.items()
    ]
    
    # Placement de la légende à droite de l'image
    ax[1].legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    
    plt.tight_layout()
    plt.show()
    

def calculate_areas(mask, gsd_cm):
    """
    Calcule la surface de chaque classe en m²
    gsd_cm : résolution au sol en centimètres par pixel (ex: 2.5)
    """
    # 1 pixel représente (gsd_cm / 100)^2 mètres carrés
    pixel_area_m2 = (gsd_cm / 100) ** 2
    
    print("\n📊 Rapport des Surfaces Réelles :")
    print(f"{'Classe':<18} | {'Pixels':<12} | {'Surface (m²)':<12}")
    print("-" * 45)
    
    for class_id, info in CLASS_MAP.items():
        # Compter le nombre de pixels pour cette classe
        num_pixels = np.sum(mask == class_id)
        area_m2 = num_pixels * pixel_area_m2
        
        print(f"{info['name']:<18} | {num_pixels:<12} | {area_m2:<12.2f}")
    
    # Focus sur les bâtiments
    total_batiment = np.sum(mask == 10) * pixel_area_m2
    print(f"\n🏠 Surface Totale Bâtie : {total_batiment:.2f} m²")


def save_visual_report(image_path, mask, output_path, gsd_cm=3.0):
    # 1. Calcul des surfaces
    pixel_area_m2 = (gsd_cm / 100) ** 2
    
    # 2. Préparation de l'image segmentée
    color_mask = colorize_mask(mask)
    color_mask_bgr = cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR)
    h, w, _ = color_mask_bgr.shape
    
    # 3. Création d'un panneau latéral blanc (largeur 400px)
    panel_width = 450
    report_img = np.ones((h, w + panel_width, 3), dtype=np.uint8) * 255
    report_img[:, :w] = color_mask_bgr # On place le masque à gauche
    
    # 4. Écriture du titre et des infos
    font = cv2.FONT_HERSHEY_SIMPLEX
    y_offset = 50
    cv2.putText(report_img, "RAPPORT DE SEGMENTATION", (w + 20, y_offset), font, 0.8, (0, 0, 0), 2)
    
    y_offset += 40
    cv2.putText(report_img, f"Image: {os.path.basename(image_path)}", (w + 20, y_offset), font, 0.5, (50, 50, 50), 1)
    y_offset += 25
    cv2.putText(report_img, f"GSD: {gsd_cm} cm/px", (w + 20, y_offset), font, 0.5, (50, 50, 50), 1)
    
    # 5. Dessin de la légende et des surfaces
    y_offset += 50
    cv2.putText(report_img, "LEGENDE & SURFACES :", (w + 20, y_offset), font, 0.6, (0, 0, 0), 2)
    y_offset += 30
    
    total_built_area = 0
    for class_id, info in CLASS_MAP.items():
        # Couleur de la classe (conversion RGB -> BGR pour OpenCV)
        color = info['color'][::-1] 
        num_pixels = np.sum(mask == class_id)
        area_m2 = num_pixels * pixel_area_m2
        
        if class_id == 10: total_built_area = area_m2

        # Dessiner le carré de couleur
        cv2.rectangle(report_img, (w + 20, y_offset - 15), (w + 40, y_offset + 5), color, -1)
        cv2.rectangle(report_img, (w + 20, y_offset - 15), (w + 40, y_offset + 5), (0,0,0), 1) # Bordure
        
        # Texte de la surface
        text = f"{info['name']}: {area_m2:.2f} m2"
        cv2.putText(report_img, text, (w + 55, y_offset), font, 0.5, (0, 0, 0), 1)
        y_offset += 35

    # 6. Résumé final en bas
    y_offset += 20
    cv2.rectangle(report_img, (w + 10, y_offset), (w + panel_width - 10, y_offset + 60), (240, 240, 240), -1)
    cv2.putText(report_img, "SURFACE BATIE TOTALE :", (w + 20, y_offset + 25), font, 0.6, (0, 0, 255), 2)
    cv2.putText(report_img, f"{total_built_area:.2f} m2", (w + 20, y_offset + 50), font, 0.8, (0, 0, 255), 2)

    # Sauvegarde
    cv2.imwrite(output_path, report_img)
    print(f"✅ Rapport visuel complet sauvegardé : {output_path}")