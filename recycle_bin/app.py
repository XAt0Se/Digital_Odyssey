import pathlib
from fastai.vision.all import load_learner, PILImage
from flask import Flask, jsonify, render_template
import os
from recycle_bin import capture_photo_android



temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

model = load_learner('result-resnet50.pkl')

pathlib.PosixPath = temp

app = Flask(__name__)



def get_image_from_folder(folder_path):
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        raise FileNotFoundError("No image files found in the folder")

    return os.path.join(folder_path, image_files[0])

@app.route('/', methods=['GET', "SET"])
def predict():
    try:
        folder_path = 'C:/Users/yusif/desktop/recycle_bin/folders/'
        
        image_path = get_image_from_folder(folder_path)
        
        img = PILImage.create(image_path)

        pred_class, pred_idx, outputs = model.predict(img)

        if pred_class == "Cardboard":
            pred_class = "Paper"
        
        os.remove(image_path)

        return render_template('index.html', pred_class=pred_class)
        

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
