import os
import logging
import datetime
import time
from injector import Module, Injector, inject, Binder
from flask import Flask, request, jsonify, Response
from flask_injector import FlaskInjector
from PIL import Image
from torchvision.io import read_image
from torchvision.models import resnet50, ResNet50_Weights
import torchvision.transforms as transforms
from prometheus_client import multiprocess
from prometheus_client import generate_latest, CollectorRegistry, CONTENT_TYPE_LATEST, Gauge

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
           start = time.time()
           image_data = request.files['image']

           if image_data is not None:
             image = Image.open(image_data)
             pred = service.predict(image)

             end = time.time()
             print("total request time: ", end - start)
             return jsonify({"result" : pred})
   
           return jsonify({"error":"file not found"})
       except Exception as e:
           return jsonify({"error": str(e)})
   
   # Route for liveness, readiness, and startup probe checking.
   @app.route('/healthz', methods = ['GET'])
   def healthCheck():
       return jsonify({"status": "healthy"})
    
   @app.route("/metrics")
   def metrics():
       registry = CollectorRegistry()
       multiprocess.MultiProcessCollector(registry)
       data = generate_latest(registry)
       return Response(data, mimetype=CONTENT_TYPE_LATEST)


class AppModule(Module):
    def __init__(self, app):
        self.app = app

    def configure(self, binder: Binder):

       # Load model weights when start up.
       weights = ResNet50_Weights.IMAGENET1K_V1
       model = resnet50(weights=weights)

       binder.bind(PredictionService, to=PredictionService(model, weights), scope=None)

createAppTime = Gauge('app_create_time', 'Timestamp when app was created')
def createApp():
   app = Flask(__name__)

   gunicorn_logger = logging.getLogger('gunicorn.error')
   
   app.logger.handlers = gunicorn_logger.handlers    
   app.logger.setLevel(gunicorn_logger.level) 

   configureHandlers(app=app)
   with app.app_context():
        injector = Injector([AppModule(app)])

   FlaskInjector(app=app, injector=injector)
   createAppTime.set(datetime.datetime.now().timestamp())

   return app

if __name__ == "__main__":
    app = createApp()
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
