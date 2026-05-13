import numpy as np
import cv2

def recentrer_et_redimensionner(image, taille_finale=28, taille_interne=20):
    """
    Recadre l'image sur le chiffre, le redimensionne pour qu'il tienne dans 
    une zone de 20x20, puis le centre dans un canevas de 28x28.
    Applique un léger flou pour imiter la diffusion de l'encre (MNIST).
    """

    img_float = image.astype(np.float32)

    coords = cv2.findNonZero((img_float > 0.1).astype(np.uint8))
    if coords is None:
        return np.zeros((taille_finale, taille_finale), dtype=np.float32)
    
    # Bounding Box
    x, y, w, h = cv2.boundingRect(coords)
    decoupe = img_float[y:y+h, x:x+w]
    
    # Redimensionnement
    ratio = taille_interne / max(w, h)
    nouvelle_largeur = max(1, int(w * ratio))
    nouvelle_hauteur = max(1, int(h * ratio))
    
    decoupe_redimensionnee = cv2.resize(decoupe, (nouvelle_largeur, nouvelle_hauteur), 
                                        interpolation=cv2.INTER_AREA)
    
    # Centrage sur le canevas
    canevas = np.zeros((taille_finale, taille_finale), dtype=np.float32)
    debut_x = (taille_finale - nouvelle_largeur) // 2
    debut_y = (taille_finale - nouvelle_hauteur) // 2

    canevas[debut_y:debut_y+nouvelle_hauteur, debut_x:debut_x+nouvelle_largeur] = decoupe_redimensionnee
    
    # Flou Gaussien, noyau de (3, 3), '0' --> laisse OpenCV calculer la variance idéale
    canevas_adouci = cv2.GaussianBlur(canevas, (3, 3), 0)
    
    # Renormalisation de l'intensité du trait
    max_val = np.max(canevas_adouci)
    if max_val > 0:
        canevas_adouci = canevas_adouci / max_val
        
    return canevas_adouci