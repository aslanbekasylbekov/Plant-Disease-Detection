import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
from src.model import build_model  

MODEL_PATH = "/Users/aslanbekasylbekov/Desktop/diploma/tomato-disease-classifier/models/tomato_model.pth"
NUM_CLASSES = 4  
CLASS_NAMES = [
    "Tomato___Bacterial_spot", "Tomato___Late_blight", "Tomato___Leaf_Mold","Tomato___Tomato_mosaic_virus"
]

def load_model():
    model = build_model(NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)) 
])

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    image = transform(image).unsqueeze(0)  # Добавляем batch dimension
    
    with torch.no_grad():
        output = model(image)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        top3_confidences, top3_indices = torch.topk(probabilities, 3)
        predictions = [(CLASS_NAMES[idx], conf.item() * 100) for idx, conf in zip(top3_indices[0], top3_confidences[0])]
    
    for i, (label, confidence) in enumerate(predictions):
        cv2.putText(frame, f"{label}: {confidence:.2f}%", (10, 50 + i * 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Tomato Disease Detection", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
