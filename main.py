import tkinter as tk
import numpy as np
from tensorflow.keras.models import load_model
from utils import recentrer_et_redimensionner
import os

GRID, SCALE = 28, 20
SIZE = GRID * SCALE

class MiniPaint:
    def __init__(self, root, model):
        self.root = root
        self.model = model
        root.title("Reconnaissance De Chiffres")
        root.resizable(False, False)

        # canvas de dessin
        self.canvas = tk.Canvas(root, bg="white", width=SIZE, height=SIZE, highlightthickness=1, highlightbackground="black")
        self.canvas.pack(pady=5, padx=5)
        self.dessin = [[0 for _ in range(GRID)] for _ in range(GRID)]

        # canvas du bas
        self.bottom_frame = tk.Frame(root)
        self.bottom_frame.pack(fill=tk.X, padx=10, pady=5)

        # Boutons 
        self.btn_frame = tk.Frame(self.bottom_frame)
        self.btn_frame.pack(side=tk.LEFT)
        
        tk.Button(self.btn_frame, text="Prédire (Space)", command=self.predict_drawing, width=15).pack(pady=2)
        tk.Button(self.btn_frame, text="Effacer (c)", command=self.clear_canvas, width=15).pack(pady=2)

        # Prédiction
        self.result_frame = tk.Frame(self.bottom_frame)
        self.result_frame.pack(side=tk.RIGHT, padx=10)
        self.lbl_result = tk.Label(self.result_frame, text="Prédictions :\n", font=("Arial", 16, "bold"), fg="#333333", justify=tk.CENTER)
        self.lbl_result.pack()

        # Evenements

        # Souris
        self.canvas.bind("<Button-1>", self.paint_cell)
        self.canvas.bind("<B1-Motion>", self.paint_cell)
        
        # Clavier
        root.bind("<Return>", lambda e: self.predict_drawing())
        root.bind("<space>", lambda e: self.predict_drawing())
        root.bind("<BackSpace>", lambda e: self.clear_canvas())
        root.bind("<c>", lambda e: self.clear_canvas())
        root.bind("<Escape>", lambda e: self.close_program())

        # Variable pour savoir s'il faut effacer le texte au prochain coup de crayon
        self.vient_de_predire = False

    def paint_cell(self, event):
        # Si on dessine juste après une prédiction, on réinitialise l'affichage
        if self.vient_de_predire:
            self.lbl_result.config(text="Prediction :\n", fg="#333333")
            self.vient_de_predire = False

        i, j = event.x // SCALE, event.y // SCALE
        if 0 <= i < GRID and 0 <= j < GRID:
            self.canvas.create_rectangle(i*SCALE, j*SCALE, (i+1)*SCALE, (j+1)*SCALE, fill="black", outline="")
            
            voisins = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]
            for di, dj in voisins:
                ni, nj = i + di, j + dj
                if 0 <= ni < GRID and 0 <= nj < GRID:
                    self.dessin[nj][ni] = 1

    def clear_canvas(self):
        self.canvas.delete("all")
        self.dessin = [[0 for _ in range(GRID)] for _ in range(GRID)]
        self.lbl_result.config(text="Prédictions :\n", fg="#333333")
        self.vient_de_predire = False

    def predict_drawing(self):
        dessin_brut = np.array(self.dessin).astype('float32')
        
        if np.max(dessin_brut) == 0:
            self.lbl_result.config(text="-", fg="black")
            return

        dessin_recentre = recentrer_et_redimensionner(dessin_brut)
        
        # Prédiction
        img_input = dessin_recentre.reshape(1, 28, 28, 1)
        predictions = self.model.predict(img_input, verbose=0)[0]
        
        idx1, p1 = np.argmax(predictions), max(predictions)
        
        # Mise à jour du texte avec le chiffre prédit
        self.lbl_result.config(text=f"Prediction :\n{idx1}", fg="blue")
        self.vient_de_predire = True

    def close_program(self):
        self.root.quit()
        self.root.destroy()

if __name__ == "__main__":

    model_path = 'cnn_model_mnist.keras' 
    
    if os.path.exists(model_path):
        trained_model = load_model(model_path)
        root = tk.Tk()
        app = MiniPaint(root, trained_model)
        root.mainloop()
    else:
        print(f"Erreur : Le modèle '{model_path}' est introuvable")