import os
import csv
import torch
from torchvision import transforms
from PIL import Image
from cnn_model_multiconv import ConvNet  # import model

# path configurations
test_folder = "/home/francesco/Downloads/dl2425_challenge_dataset/test"
model_path = "/home/francesco/Documents/Materiale Università/Third year/Deep Learning/Exercises/best_model_convnet_97_34.pth"
output_csv_path = "/home/francesco/Downloads/output_predictions.csv"

# Trasformazioni immagini
transform = transforms.Compose([
    transforms.ToTensor()
])

# load the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ConvNet()
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# prediction function
def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        prediction = (output.item() > 0.5)  # class 1 if p > 0.5, otherwise 0
    return int(prediction)

# create CSV file and make predictions
with open(output_csv_path, mode='w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["Image Name", "Prediction"])
    
    for image_name in os.listdir(test_folder):
        image_path = os.path.join(test_folder, image_name)
        if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            prediction = predict_image(image_path)
            csv_writer.writerow([image_name, prediction])
            print(f"Predicted {prediction} for {image_name}")

print(f"Predictions saved to {output_csv_path}")
