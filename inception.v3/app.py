import os
from injector import Module, Injector, inject, Binder
from flask import Flask, request, jsonify
from flask_injector import FlaskInjector
from PIL import Image
from torchvision.io import read_image
from torchvision.models import inception.v3, Inception_V3_Weights
import torchvision.transforms as transforms


class PredictionService:
    def __init__(self, model, weights):
       self.model = model
       self.weights = weights

    def predict(self, image):
       model = self.model
       weights = self.weights
    
       preprocess = weights.transforms()
       batch = preprocess(image).unsqueeze(0)
    
       prediction = self.model(batch).squeeze(0).softmax(0)
       class_id = prediction.argmax().item()
       score = prediction[class_id].item()
       category_name = weights.meta["categories"][class_id]
    
       return (f"{category_name}: {100 * score:.1f}%")

def configureHandlers(app):
   @inject
   @app.route('/predict', methods = ['POST'])
   def handlePredict(service: PredictionService):
       try:
           image_data = request.files['image']

           if image_data is not None:
             image = Image.open(image_data)
             pred = service.predict(image)

             return jsonify({"result" : pred})
   
           return jsonify({"error":"file not found"})
       except Exception as e:
           return jsonify({"error": str(e)})
   
   # Route for liveness, readiness, and startup probe checking.
   @app.route('/healthz', methods = ['GET'])
   def healthCheck():
       return jsonify({"status": "healthy"})

class AppModule(Module):
    def __init__(self, app):
        self.app = app

    def configure(self, binder: Binder):

       # Load model weights when start up.
       weights = Inception_V3_Weights.IMAGENET1K_V1
       model = inception.v3(weights=weights)

       binder.bind(PredictionService, to=PredictionService(model, weights), scope=None)

def createApp():
   app = Flask(__name__)
   configureHandlers(app=app)
   with app.app_context():
        injector = Injector([AppModule(app)])

   FlaskInjector(app=app, injector=injector)

   return app

if __name__ == "__main__":
    app = createApp()
    app.run(debug=True,host='0.0.0.0',port=int(os.environ.get('PORT', 8080)))
