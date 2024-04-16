import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import mlflow
import numpy as np
import tensorflow as tf
import pandas as pd

# Create a window
window = tk.Tk()
window.title("MNIST Data Generator")

# Create a canvas for drawing
canvas = tk.Canvas(window, width=280, height=280, bg="white")
canvas.pack()

# Create an image and a draw object
image = Image.new("L", (280, 280), 255)
draw = ImageDraw.Draw(image)

# Function to handle mouse events
def on_mouse(event):
    x = event.x
    y = event.y
    canvas.create_oval(x-10, y-10, x+10, y+10, fill="black")
    draw.ellipse((x-10, y-10, x+10, y+10), fill=0)

# Bind mouse events to the canvas
canvas.bind("<B1-Motion>", on_mouse)

title = tk.Label(window, text="Draw a digit from 0 to 9")
title.pack()

# Function to save the drawn image
def predict_value():
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    logged_model = 'runs:/c2ee72bd1e9b442a87bae86e74aade20/model'

    loaded_model = mlflow.pyfunc.load_model(logged_model)

    resized = image.resize((28,28))
    grayscale = ImageOps.grayscale(resized)
    transformed = np.array(grayscale, dtype=np.float32) / 255.
    reshaped = np.reshape(transformed, (1, 28, 28, 1))
    result = loaded_model.predict(reshaped)

    prediction = result.argmax()
    title.config(text=f"Predicted value: {prediction}")

# Create a button to save the image
save_button = tk.Button(window, text="Predict value", command=predict_value)
save_button.pack()

def clean_canvas():
    canvas.delete("all")
    global image, draw
    image = Image.new("L", (280, 280), 255)
    draw = ImageDraw.Draw(image)
    title.config(text="Draw a digit from 0 to 9")

# Create a button to save the image
clean_button = tk.Button(window, text="Clean", command=clean_canvas)
clean_button.pack()

# Start the main event loop
window.mainloop()