import numpy as np
import re
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
class model:
  def __init__(self):
    print("HEllo")
  def scaling(self,dataframe):
    scaler=StandardScaler()
    prep_data=scaler.fit_transform(dataframe.iloc[:,6:15].to_numpy())
    return prep_data,scaler

  def nn_predictor(self,prep_data):
    neigh = NearestNeighbors(metric='cosine',algorithm='brute')
    neigh.fit(prep_data)
    return neigh

  def build_pipeline(self,neigh,scaler,params):
    transformer = FunctionTransformer(neigh.kneighbors,kw_args=params)
    pipeline=Pipeline([('std_scaler',scaler),('NN',transformer)])
    return pipeline

  def extract_data(self,dataframe,ingredients):
    extracted_data=dataframe.copy()
    extracted_data=self.extract_ingredient_filtered_data(extracted_data,ingredients)
    return extracted_data
    
  def extract_ingredient_filtered_data(self,dataframe,ingredients):
    extracted_data=dataframe.copy()
    regex_string=''.join(map(lambda x:f'(?=.*{x})',ingredients))
    extracted_data=extracted_data[extracted_data['RecipeIngredientParts'].str.contains(regex_string,regex=True,flags=re.IGNORECASE)]
    return extracted_data

  def apply_pipeline(self,pipeline,_input,extracted_data):
    _input=np.array(_input).reshape(1,-1)
    return extracted_data.iloc[pipeline.transform(_input)[0]]

  def recommend(self,dataframe,_input,ingredients=[],params={'n_neighbors':5,'return_distance':False}):
        extracted_data=self.extract_data(dataframe,ingredients)
        if extracted_data.shape[0]>=params['n_neighbors']:
            prep_data,scaler=self.scaling(extracted_data)
            neigh=self.nn_predictor(prep_data)
            pipeline=self.build_pipeline(neigh,scaler,params)
            return self.apply_pipeline(pipeline,_input,extracted_data)
        else:
            return None

  def extract_quoted_strings(self,s):
    # Find all the strings inside double quotes
    strings = re.findall(r'"([^"]*)"', s)
    # Join the strings with 'and'
    return strings

  def output_recommended_recipes(self,dataframe):
    if dataframe is not None:
        output=dataframe.copy()
        output=output.to_dict("records")
        for recipe in output:
            recipe['RecipeIngredientParts']=self.extract_quoted_strings(recipe['RecipeIngredientParts'])
            recipe['RecipeInstructions']=self.extract_quoted_strings(recipe['RecipeInstructions'])
    else:
        output=None
    return output

from fastapi import FastAPI
from pydantic import BaseModel,conlist
from typing import List,Optional
import pandas as pd
from typing import List, Dict



dataset=pd.read_csv('./dataset (1).csv',compression='gzip',header=0, sep=',', quotechar='"')

# with open("/content/drive/MyDrive/diet/dataset (1).csv",encoding='latin1',error_bad_lines=False) as f:
#       train = pd.read_csv(f, header=0, delimiter="\t")
#       print(train.head())



#app=Flask(__name__)


class params(BaseModel):
    n_neighbors:int=5
    return_distance:bool=False

class PredictionIn(BaseModel):
    nutrition_input:conlist(float, min_items=9, max_items=9)
    ingredients:List[str]=[]
    params:Optional[params]

class Recipe(BaseModel):
    Name:str
    CookTime:str
    PrepTime:str
    TotalTime:str
    RecipeIngredientParts:List[str]
    Calories:float
    FatContent:float
    SaturatedFatContent:float
    CholesterolContent:float
    SodiumContent:float
    CarbohydrateContent:float
    FiberContent:float
    SugarContent:float
    ProteinContent:float
    RecipeInstructions:List[str]


class PredictionOut(BaseModel):
    output: Optional[List[Recipe]] = None


app = FastAPI()

@app.get("/")
def home():
    return {"health_check": "OK"}

@app.post("/predict/",response_model=PredictionOut)

def update_item(prediction_input:PredictionIn):
    model1=model()
    recommendation_dataframe=model1.recommend(dataset,prediction_input.nutrition_input,prediction_input.ingredients,prediction_input.params.dict())
    output=model1.output_recommended_recipes(recommendation_dataframe)
    if output is None:
        return {"output":None}
    else:
        return {"output":output}

  
