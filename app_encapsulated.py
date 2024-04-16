import numpy as np
import tkinter as tk
from PIL import Image, ImageDraw
import mlflow

class DigitRecognizerApp:
    def __init__(self, window, loaded_model):
        self.window = window
        self.loaded_model = loaded_model
        self.image = Image.new("L", (280, 280), 255)
        self.draw = ImageDraw.Draw(self.image)
        self.title = tk.Label(window)
        self.title.pack()
        
        self.canvas = tk.Canvas(window, width=280, height=280, bg="white")
        self.canvas.pack()
        self.canvas.bind("<B1-Motion>", self.draw_lines)
        self.canvas.bind("<ButtonRelease-1>", lambda _: self.predict_value())

        # Create a button to predict the value
        self.save_button = tk.Button(window, text="Predict value", command=self.predict_value)
        self.save_button.pack()

        # Create a button to clean the canvas
        self.clean_button = tk.Button(window, text="Clean", command=self.clean_canvas)
        self.clean_button.pack()

    def draw_lines(self, event):
        x = event.x
        y = event.y
        self.canvas.create_oval(x-10, y-10, x+10, y+10, fill="black")
        self.draw.ellipse((x-10, y-10, x+10, y+10), fill=0)

    def predict_value(self):
        grayscale = self.image.convert('L')
        resized = grayscale.resize((28,28))
        transformed = np.array(resized, dtype=np.float32) / 255.
        reshaped = np.reshape(transformed, (1, 28, 28, 1))
        result = self.loaded_model.predict(reshaped)

        prediction = result.argmax()
        self.title.config(text=f"Predicted value: {prediction}")

    def clean_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (280, 280), 255)
        self.draw = ImageDraw.Draw(self.image)
        self.title.config(text="Draw a digit from 0 to 9")

if __name__ == "__main__":
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    loaded_model = mlflow.pyfunc.load_model('runs:/c2ee72bd1e9b442a87bae86e74aade20/model')

    window = tk.Tk()
    app = DigitRecognizerApp(window, loaded_model)
    window.mainloop()