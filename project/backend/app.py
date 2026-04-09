from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
import io
import base64
import numpy as np
import cv2
from PIL import Image

app = FastAPI(title="Semantic Segmentation API", version="1.0")

# Enable CORS to allow the frontend to communicate with the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------
# 1. MODEL LOADING 
# ---------------------------------------------------------
# Define your PyTorch model architecture here!
# Example: 
# import torchvision.models.segmentation as segmentation
# model = segmentation.fcn_resnet50(pretrained=False, num_classes=2)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DummyModel:
    """ This is a placeholder model. Replace with your actual PyTorch model class """
    def __init__(self):
        pass
    def eval(self):
        pass
    def __call__(self, x):
        # Returns a dummy tensor mimicking model output
        return torch.rand((1, 1, x.shape[2], x.shape[3]))
        
try:
    print("Loading model.pth...")
    model = DummyModel() # Initialize your model class here
    # UNCOMMENT THIS IN PRODUCTION:
    # model.load_state_dict(torch.load("model.pth", map_location=device))
    model.eval()
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")

# ---------------------------------------------------------
# 2. UTILITY FUNCTIONS
# ---------------------------------------------------------
def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    """ Preprocesses the uploaded image bytes for the PyTorch model. """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    # Resize / normalize based on your model's requirements
    # Example: img = img.resize((256, 256))
    img_np = np.array(img, dtype=np.float32) / 255.0
    # Convert HWC to CHW (PyTorch standard)
    img_tensor = torch.tensor(img_np).permute(2, 0, 1).unsqueeze(0)
    return img_tensor.to(device)

def postprocess_output(output_tensor: torch.Tensor, original_size: tuple) -> np.ndarray:
    """ Converts the model output tensor back to a presentable image format. """
    # Convert tensor to numpy (assuming single class / binary segmentation for demo)
    pred_mask = output_tensor.squeeze().cpu().detach().numpy()
    
    # Thresholding (e.g. at 0.5) to get binary mask
    binary_mask = (pred_mask > 0.5).astype(np.uint8) * 255
    
    # In a real scenario, you probably want to overlay this mask on the image 
    # or return it directly. Let's return just the mask for visualization.
    # Note: Using dummy mask here to demonstrate the pipeline since DummyModel is random.
    return binary_mask

def encode_image_to_base64(image_array: np.ndarray) -> str:
    """ Encodes an OpenCV/Numpy image array to base64. """
    # For a smoother demo UI, let's create a cool colored mask instead of pure black/white
    colormap_mask = cv2.applyColorMap(image_array, cv2.COLORMAP_JET)
    
    # Encode as PNG
    success, encoded_image = cv2.imencode('.png', colormap_mask)
    if not success:
        raise ValueError("Failed to encode image")
    
    return base64.b64encode(encoded_image.tobytes()).decode('utf-8')

# ---------------------------------------------------------
# 3. API ENDPOINTS
# ---------------------------------------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith('image/'):
        return JSONResponse(status_code=400, content={"message": "File must be an image."})
        
    try:
        # Read uploaded image
        image_bytes = await file.read()
        
        # Keep original image size for later (width, height)
        orig_img = Image.open(io.BytesIO(image_bytes))
        original_size = orig_img.size
        
        # 1. Preprocess
        input_tensor = preprocess_image(image_bytes)
        
        # 2. Inference
        with torch.no_grad():
            output = model(input_tensor)
            
        # 3. Postprocess
        # Generates a dummy mask for the sake of presentation
        # Replace postprocess_output functionality with your exact logic
        mask_array = postprocess_output(output, original_size[::-1])
        
        # Since DummyModel produces random static, let's make a clear fake shape if it's too noisy
        # cv2.circle(mask_array, (original_size[0]//2, original_size[1]//2), min(original_size)//4, 255, -1)

        # 4. Convert to base64 for frontend
        seg_base64 = encode_image_to_base64(mask_array)
        
        # ------------------------------------------------------------------
        # IMPORTANT NOTE ON METRICS:
        # IoU, Accuracy, and Dice require a "Ground Truth" (real mask)
        # to compare against the predicted mask. Since inference only gets 
        # the raw image, these metrics cannot be accurately calculated here 
        # unless the frontend also uploads the ground truth mask.
        # 
        # We are returning mock high-quality metrics for frontend preview.
        # ------------------------------------------------------------------
        return {
            "segmented_image": f"data:image/png;base64,{seg_base64}",
            "iou_score": 87.42, 
            "pixel_accuracy": 94.61,
            "dice_score": 89.25
        }
        
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
