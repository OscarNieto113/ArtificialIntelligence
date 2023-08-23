from flask import Flask
from flask_restful import Resource, Api, reqparse, abort  

import json

app = Flask("VideoAPI")
api = Api(app)

class Index(Resource):
    def get(self):
        return "Hello World!", 200
    
api.add_resource(Index, "/")

if __name__ == "__main__":
    app.run()
