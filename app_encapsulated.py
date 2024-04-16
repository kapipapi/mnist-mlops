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

        size = 20
        self.canvas.create_oval(x-size//2, y-size//2, x+size//2, y+size//2, fill="black")
        self.draw.ellipse((x-size//2, y-size//2, x+size//2, y+size//2), fill=0)

    def predict_value(self):
        grayscale = self.image.convert('L')
        img_data = np.array(grayscale)

        # Find the rows and columns where the image is not white
        non_white_pixels = np.where(img_data != 255)
        min_y, max_y = np.min(non_white_pixels[0]), np.max(non_white_pixels[0])
        min_x, max_x = np.min(non_white_pixels[1]), np.max(non_white_pixels[1])

        img_data = img_data[min_y:max_y, min_x:max_x]

        # Add padding to the image
        side = max(img_data.shape[0], img_data.shape[1])
        pad_x = max(side - img_data.shape[0], 0)
        pad_y = max(side - img_data.shape[1], 0)
        
        img_data = np.pad(img_data, ((pad_x//2, pad_x//2), (pad_y//2, pad_y//2)), 'constant', constant_values=255)

        from PIL import Image
        im = Image.fromarray(img_data)
        im.save("test.png")

        resized = np.resize(img_data, (28, 28))
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
    loaded_model = mlflow.pyfunc.load_model('runs:/cd39b8ed1b3a4f579b712181fa3576f5/mnist-model')

    window = tk.Tk()
    app = DigitRecognizerApp(window, loaded_model)
    window.mainloop()