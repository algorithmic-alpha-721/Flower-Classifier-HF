# # import os
# # import torch
# # import torch.nn.functional as F
# # from transformers import AutoImageProcessor, AutoModelForImageClassification
# # from PIL import Image

# # MODEL_PATH = "model/flower_model"

# # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # print("Loading model...")
# # processor = AutoImageProcessor.from_pretrained(MODEL_PATH)
# # model = AutoModelForImageClassification.from_pretrained(MODEL_PATH)

# # model.to(device)
# # model.eval()
# # print("Model loaded successfully!\n")


# # def predict_image(image):
# #     inputs = processor(image, return_tensors="pt")
# #     inputs = {k: v.to(device) for k, v in inputs.items()}

# #     with torch.no_grad():
# #         outputs = model(**inputs)

# #     probs = F.softmax(outputs.logits, dim=-1)
# #     pred = probs.argmax(-1).item()
# #     confidence = probs[0][pred].item()

# #     label = model.config.id2label[pred]

# #     return label, confidence


# # def main():
# #     folder = "test_images"

# #     if not os.path.exists(folder):
# #         print("⚠ 'test_images' folder not found.")
# #         print("Create it and add some images.")
# #         return

# #     for file in os.listdir(folder):
# #         if file.lower().endswith((".jpg", ".png", ".jpeg")):
# #             path = os.path.join(folder, file)
# #             image = Image.open(path).convert("RGB")
# #             label, conf = predict_image(image)
# #             print(f"{file} → {label} ({conf*100:.2f}%)")


# # if __name__ == "__main__":
# #     main()
# import os
# import torch
# import torch.nn.functional as F
# from transformers import AutoImageProcessor, AutoModelForImageClassification
# from PIL import Image

# MODEL_PATH = "model/flower_model"

# LABELS = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# processor = AutoImageProcessor.from_pretrained(MODEL_PATH)
# model = AutoModelForImageClassification.from_pretrained(MODEL_PATH)

# model.to(device)
# model.eval()


# def predict_image(image):
#     inputs = processor(image, return_tensors="pt")
#     inputs = {k: v.to(device) for k, v in inputs.items()}

#     with torch.no_grad():
#         outputs = model(**inputs)

#     probs = F.softmax(outputs.logits, dim=-1)
#     pred = probs.argmax(-1).item()
#     confidence = probs[0][pred].item()

#     label = LABELS[pred]

#     return label, confidence


# def main():
#     folder = "test_images"

#     if not os.path.exists(folder):
#         print("Create 'test_images' folder and add images.")
#         return

#     for file in os.listdir(folder):
#         if file.lower().endswith((".jpg", ".png", ".jpeg")):
#             path = os.path.join(folder, file)
#             image = Image.open(path).convert("RGB")
#             label, conf = predict_image(image)
#             print(f"{file} → {label} ({conf*100:.2f}%)")


# if __name__ == "__main__":
#     main()


import os
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
import matplotlib.pyplot as plt

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model + processor
model_path = "C:\\Users\\ayush\\Deep_learning_works\\Hugging_Face\\Flower_Classifier\\model\\convnext_flower102_model"

processor = AutoImageProcessor.from_pretrained(model_path)
model = AutoModelForImageClassification.from_pretrained(model_path)
model.to(device)
model.eval()

# If you saved id2label during training, this auto-loads
# id2label = model.config.id2label
flower_names = [
    "pink primrose", "hard-leaved pocket orchid", "canterbury bells",
    "sweet pea", "english marigold", "tiger lily", "moon orchid",
    "bird of paradise", "monkshood", "globe thistle",
    "snapdragon", "colt's foot", "king protea",
    "spear thistle", "yellow iris", "globe-flower",
    "purple coneflower", "peruvian lily", "balloon flower",
    "giant white arum lily", "fire lily", "pincushion flower",
    "fritillary", "red ginger", "grape hyacinth",
    "corn poppy", "prince of wales feathers", "stemless gentian",
    "artichoke", "sweet william", "carnation",
    "garden phlox", "love in the mist", "mexican aster",
    "alpine sea holly", "ruby-lipped cattleya", "cape flower",
    "great masterwort", "siam tulip", "lenten rose",
    "barbeton daisy", "daffodil", "sword lily",
    "poinsettia", "bolero deep blue", "wallflower",
    "marigold", "buttercup", "oxeye daisy",
    "common dandelion", "petunia", "wild pansy",
    "primula", "sunflower", "pelargonium",
    "bishop of llandaff", "gaura", "geranium",
    "orange dahlia", "pink-yellow dahlia", "cautleya spicata",
    "japanese anemone", "black-eyed susan", "silverbush",
    "californian poppy", "osteospermum", "spring crocus",
    "bearded iris", "windflower", "tree poppy",
    "gazania", "azalea", "water lily",
    "rose", "thorn apple", "morning glory",
    "passion flower", "lotus", "toad lily",
    "anthurium", "frangipani", "clematis",
    "hibiscus", "columbine", "desert-rose",
    "tree mallow", "magnolia", "cyclamen",
    "watercress", "canna lily", "hippeastrum",
    "bee balm", "ball moss", "foxglove",
    "bougainvillea", "camellia", "mallow",
    "mexican petunia", "bromelia", "blanket flower",
    "trumpet creeper", "blackberry lily"
]
# label = flower_names[pred_class]
id2label = {i: label for i, label in enumerate(flower_names)}


# Folder containing test images
test_folder = "C:\\Users\\ayush\\Deep_learning_works\\Hugging_Face\\Flower_Classifier\\test_images"

for img_name in os.listdir(test_folder):
    if img_name.lower().endswith((".png", ".jpg", ".jpeg")):
        img_path = os.path.join(test_folder, img_name)

        image = Image.open(img_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()

        confidence = probs[0][pred_class].item() * 100
        label = id2label[pred_class]

        print(f"\nImage: {img_name}")
        print(f"Predicted: {label}")
        print(f"Confidence: {confidence:.2f}%")

        # Visual display
        plt.imshow(image)
        plt.title(f"{label} ({confidence:.2f}%)")
        plt.axis("off")
        plt.show()
