from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io

import torch
from torchvision import models, transforms

app = FastAPI(title="Image Classification API")

model = models.mobilenet_v2(
    weights=models.MobileNet_V2_Weights.DEFAULT
)
model.eval()  

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

labels = models.MobileNet_V2_Weights.DEFAULT.meta["categories"]

@app.get("/")
def health():
    return {
        "status": "ok",
        "message": "ML API is running"
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    input_tensor = preprocess(image)
    input_tensor = input_tensor.unsqueeze(0)  # Batch dimension

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, dim=1)

    return {
        "filename": file.filename,
        "predicted_class": labels[predicted_idx.item()],
        "confidence": float(confidence.item()),
    }
