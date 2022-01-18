import logging
from fastapi import FastAPI
import api.modules.algorithms as algo
import api.modules.hsi as hsi
import numpy as np
from json import dumps
from pydantic import BaseModel
import os
import time

CWD = "/data"

logger = logging.getLogger("hsi-api")
logging.basicConfig(level=logging.DEBUG)


class Inspection(BaseModel):
    crop: str
    date: str
    altitude: str
    filename: str


app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}
    
@app.get("/crops")
async def list_crops():
    dates = os.listdir(CWD)
    return {"crops": dumps(dates)}


@app.get("/crops/{crop}")
async def list_inspections(crop: str):
    dates = os.listdir(os.path.join(CWD, crop))
    return {"dates": dumps(dates)}


@app.get("/crops/{crop}/{date}")
async def list_altitudes(crop: str, date: str):
    altitudes = os.listdir(os.path.join(CWD, crop, date))
    return {"sets": dumps(altitudes)}


@app.get("/crops/{crop}/{date}/{altitude}")
async def list_folders(crop: str, date: str, altitude: str):
    folders = os.listdir(os.path.join(CWD, crop, date, altitude))
    return {"folders": dumps(folders)}


@app.get("/crops/{crop}/{date}/{altitude}/{folder}")
async def list_images(crop: str, date: str, altitude: str, folder: str):
    images = os.listdir(os.path.join(CWD, crop, date, altitude, folder))
    return {"images": dumps(images)}


@app.get("/crops/{crop}/{date}/{altitude}/{folder}/{filename}/{kind}")
async def get_image(crop: str, date: str, altitude: str, folder: str, filename: str, kind: str):
    # Get image
    cube, _ = hsi.load_image(path=os.path.join(CWD, crop, date, altitude, folder), filename=filename)
    if kind in ["rgb", "cir", "inv"]:
        img_out = hsi.set_mode(cube, mode=kind)
    elif kind == "severity":
        img_out = algo.estimate_severity_level(cube)
    return {"image": dumps(img_out.tolist())}


# Results

@app.get("/vegetation_indexes")
async def list_indexes():
    return {"vis": dumps(algo.VEGETATION_INDEXES)}


@app.get("/vegetation_indexes/{crop}/{date}/{altitude}/{filename}/{index}")
async def get_index(crop: str, date: str, altitude: str, index: str, filename: str):
    cube, _ = hsi.load_image(path=os.path.join(CWD, crop, date, altitude, "ref"), filename=filename)
    indxr = algo.VIndexer(cube)
    out = indxr.get_index(index_name=index)
    interval = indxr.index_minmax[index]
    return {"vi": dumps(out.tolist()), "minmax": dumps(interval)}


@app.post("/predict")
async def predict(item: Inspection):
    clf = algo.Classifier(os.path.join(os.getcwd(), "api", "model", "vegetation-clf.joblib"))
    logger.info("Model Loaded.")
    cube, _ = hsi.load_image(path=os.path.join(CWD,
                                               item.crop,
                                               item.date,
                                               item.altitude, "ref"),
                             filename=item.filename)
    logger.info("Making prediction")
    it = time.time()
    pred = clf.classify(cube)
    logger.info(f"Prediction made in {time.time() -it} seconds.")
    return {"mask": dumps(np.clip(pred, 0, 1).tolist())}

    # Insert to MongoDB
    # try:
    #     client = pymongo.MongoClient("localhost", 27017)
    #     db = client.crops
    #     collection = db[item["crop"]]
    #     # Insert
    #     inspection = {
    #         "path": item["path"]
    #     }
    #     collection.insert_one(inspection)
    #     return True
    # except Exception as e:
    #     logger.error(str(e))
    #     return False

###

