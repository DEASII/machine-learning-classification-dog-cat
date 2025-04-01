import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf

import os
model_path = os.path.abspath("dog_cat_classification_model.h5")
print("Loading model from:", model_path)
model = tf.keras.models.load_model("/Users/cactus/Desktop/machine-lerning/dog_cat_classification_model.h5")


class DogCatApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Dog vs Cat Predictor")
        
        # องค์ประกอบ GUI
        self.label = tk.Label(root, text="เลือกภาพสุนัขหรือแมว", font=('Helvetica', 16))
        self.label.pack(pady=20)
        
        self.btn_select = tk.Button(root, text="เลือกไฟล์", command=self.load_image)
        self.btn_select.pack(pady=10)
        
        self.image_label = tk.Label(root)
        self.image_label.pack()
        
        self.result_label = tk.Label(root, text="", font=('Helvetica', 14))
        self.result_label.pack(pady=20)
    
    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="เลือกภาพ",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
        )
        
        if file_path:
            img = Image.open(file_path)
            img = img.resize((300, 300))
            img_tk = ImageTk.PhotoImage(img)
            self.image_label.config(image=img_tk)
            self.image_label.image = img_tk
            
            # predic
            prediction = self.predict_image(file_path)
            pred_value = prediction[0][0]
            class_name = "สุนัข 🐶" if pred_value > 0.5 else "แมว 🐱"
            confidence = max(pred_value, 1 - pred_value)
            
            self.result_label.config(
                text=f"ผลลัพธ์: {class_name}\nความมั่นใจ: {confidence:.2%}",
                fg="green" if class_name == "สุนัข 🐶" else "orange"
            )
    
    def predict_image(self, image_path):
        img = Image.open(image_path).resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # predic
        prediction = model.predict(img_array)
        return prediction

# GUI
root = tk.Tk()
app = DogCatApp(root)
root.geometry("500x600")
root.mainloop()