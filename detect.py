from io import BytesIO
import cv2
from fastapi import APIRouter, File,UploadFile,HTTPException
from PIL import Image
import numpy as np
from logic import emotion_detector
detect_router= APIRouter()





@detect_router.post('/detect')
async def detect(im:UploadFile=File(...)):
    if im.filename.split('.')[-1] in ('jpg','jpeg','.png'):
        pass
    else:
        raise HTTPException(
            status_code=415 ,detail="Image not found"
        )
    contents = await im.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return emotion_detector(img)

