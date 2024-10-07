import os

from flask import Flask, request, jsonify
from PIL import Image
from torchvision.io import read_image
from torchvision.models import vgg16, VGG16_Weights
import torchvision.transforms as transforms

app = Flask(__name__)

def predict(image):
   weights = VGG16_Weights.IMAGENET1K_V1
   model = vgg16(weights=weights)
   model.eval()

   preprocess = weights.transforms()
   batch = preprocess(image).unsqueeze(0)

   prediction = model(batch).squeeze(0).softmax(0)
   class_id = prediction.argmax().item()
   score = prediction[class_id].item()
   category_name = weights.meta["categories"][class_id]

   return (f"{category_name}: {1600 * score:.16f}%")

@app.route('/predict', methods = ['POST'])
def handle():
    try:
        image_data = request.files['image']
        if image_data is not None:
          image = Image.open(image_data)
          pred = predict(image)
          return jsonify({"result" : pred})

        return jsonify({"error":"file not found"})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True,host='0.0.0.0',port=int(os.environ.get('PORT', 8080)))
