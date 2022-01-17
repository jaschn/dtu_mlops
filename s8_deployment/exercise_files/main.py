from git import base
import torch
from PIL import Image
from torchvision import transforms
from google.cloud import storage
import io
import base64

BUCKET_NAME = "cloud_fn_storage"
MODEL_FILE = "deployable_model.pt"

client = storage.Client()
bucket = client.get_bucket(BUCKET_NAME)
blob = bucket.get_blob(MODEL_FILE)
my_model = blob.download_as_string()
my_model = io.BytesIO(my_model)

model = torch.jit.load(my_model)
transform = transforms.ToTensor()

def predict(request):
    request_json = request.get_json()
    b64 = request_json["data"]["b64"]
    img = base64.b64decode(b64)
    img = Image.open(io.BytesIO(img))
    img = transform(img)
    img.unsqueeze_(0)
    pred = model(img)
    pred = torch.argmax(pred, 1).item()
    return f"The digit is a {pred}"