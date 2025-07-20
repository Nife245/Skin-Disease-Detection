from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
import tensorflow as tf
from PIL import Image
import numpy as np
import io

# Load model
model = tf.keras.models.load_model("model/skin_diseases_detection.h5")
print("Model loaded successfully.")

# Define class names (adjust based on your model)
class_names = ['Acne And Rosacea Photos',
 'Actinic Keratosis Basal Cell Carcinoma And Other Malignant Lesions',
 'Atopic Dermatitis Photos', 'Ba  Cellulitis', 'Ba Impetigo', 'Benign',
 'Bullous Disease Photos',
 'Cellulitis Impetigo And Other Bacterial Infections' ,'Eczema Photos',
 'Exanthems And Drug Eruptions', 'Fu Athlete Foot', 'Fu Nail Fungus',
 'Fu Ringworm', 'Hair Loss Photos Alopecia And Other Hair Diseases',
 'Heathy', 'Herpes Hpv And Other Stds Photos',
 'Light Diseases And Disorders Of Pigmentation',
 'Lupus And Other Connective Tissue Diseases', 'Malignant',
 'Melanoma Skin Cancer Nevi And Moles',
 'Nail Fungus And Other Nail Disease', 'Pa Cutaneous Larva Migrans',
 'Poison Ivy Photos And Other Contact Dermatitis',
 'Psoriasis Pictures Lichen Planus And Related Diseases' ,'Rashes',
 'Scabies Lyme Disease And Other Infestations And Bites',
 'Seborrheic Keratoses And Other Benign Tumors', 'Systemic Disease',
 'Tinea Ringworm Candidiasis And Other Fungal Infections',
 'Urticaria Hives', 'Vascular Tumors' ,'Vasculitis Photos' ,'Vi Chickenpox',
 'Vi Shingles', 'Warts Molluscum And Other Viral Infections']

app = FastAPI()

# If using UI
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).resize((224, 224))  # adjust to your input size
    image = np.array(image) / 255.0  # normalize
    if image.shape[-1] == 4:  # Remove alpha channel if present
        image = image[..., :3]
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    predictions = model.predict(image)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = float(np.max(predictions))

    return {"prediction": predicted_class, "confidence": confidence}
