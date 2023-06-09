from flask import Flask, request
from flask_restful import Resource, Api
import pickle
import numpy as np
import pandas as pd
from flask_cors import CORS


app = Flask(__name__)
#
CORS(app)
# creating an API object
api = Api(app)

#prediction api call
class prediction(Resource):
    def get(self, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16):
        #budget = request.args.get('budget')
        # print(f"A1: {A1}, A2: {A16}")
        # Let's load the package
        values = np.array([[float(A1), float(A2), float(A3), float(A4), float(A5), float(A6),float(A7),float(A8),float(A9), float(A10),float(A11),float(A12),float(A13), float(A14), float(A15), float(A16)]])
        print(values)
        df = pd.DataFrame(values, columns=['texture_mean', 'area_mean', 'smoothness_mean', 'concavity_mean', 'symmetry_mean', 'fractal_dimension_mean', 'texture_se', 'area_se', 'smoothness_se', 'concavity_se', 'symmetry_se', 'fractal_dimension_se','smoothness_worst', 'concavity_worst', 'symmetry_worst','fractal_dimension_worst'])
        # print(df)
        model = pickle.load(open("D:/NU/NU Semster 8/data analysis/project/Deploy%20Model%20as%20API/Deploy%20Model%20as%20API/logistic_reg_model.pkl", 'rb'))
        prediction = model.predict(df)
        prediction = int(prediction[0])
        return str(prediction)


#data api
class getData(Resource):
    def get(self):
            df = pd.read_excel("D:/NU/NU Semster 8/data analysis/project/Deploy%20Model%20as%20API/Deploy%20Model%20as%20API/data.xlsx")
            df =  df.rename({'Marketing Budget': 'budget', 'Actual Sales': 'sale'}, axis=1)  # rename columns
            #print(df.head())
            #out = {'key':str}
            res = df.to_json(orient='records')
            #print( res)
            return res

class landingPage(Resource):
    def get(self):
        return str('server running')

api.add_resource(landingPage, '/')
api.add_resource(getData, '/api')
api.add_resource(prediction, '/prediction/<float:A1>-<float:A2>-<float:A3>-<float:A4>-<float:A5>-<float:A6>-<float:A7>-<float:A8>-<float:A9>-<float:A10>-<float:A11>-<float:A12>-<float:A13>-<float:A14>-<float:A15>-<float:A16>')

if __name__ == '__main__':
    app.run(debug=True)