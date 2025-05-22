from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader

# Define the path to your dataset
dataset_path = 'Datasetplant1/New_Plant Diseases_Dataset/New_Plant_Diseases_Dataset/train'

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Load dataset
dataset = ImageFolder(root=dataset_path, transform=transform)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Display the class names
print(dataset.classes)
