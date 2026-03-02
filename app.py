# import gradio as gr
# import torch
# import torch.nn.functional as F
# from transformers import AutoImageProcessor, AutoModelForImageClassification

# MODEL_PATH = "model/flower_model"
# LABELS = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# processor = AutoImageProcessor.from_pretrained(MODEL_PATH)
# model = AutoModelForImageClassification.from_pretrained(MODEL_PATH)

# model.to(device)
# model.eval()


# def predict(image):
#     inputs = processor(image, return_tensors="pt")
#     inputs = {k: v.to(device) for k, v in inputs.items()}

#     with torch.no_grad():
#         outputs = model(**inputs)

#     probs = F.softmax(outputs.logits, dim=-1)[0]

#     top3 = torch.topk(probs, 3)

#     results = {}
#     for score, idx in zip(top3.values, top3.indices):
#         results[LABELS[idx.item()]] = float(score.item())

#     return image, results


# with gr.Blocks(theme=gr.themes.Soft()) as demo:
    
#     gr.Markdown(
#         """
#         # 🌸 AI Flower Classifier  
#         Upload a flower image and let the model predict the species.
#         """
#     )

#     with gr.Row():
#         with gr.Column():
#             input_image = gr.Image(type="pil", label="Upload Flower Image")
#             predict_btn = gr.Button("🔍 Classify", variant="primary")

#         with gr.Column():
#             output_image = gr.Image(label="Preview")
#             output_label = gr.Label(num_top_classes=3, label="Prediction Confidence")

#     predict_btn.click(
#         fn=predict,
#         inputs=input_image,
#         outputs=[output_image, output_label]
#     )

#     gr.Markdown(
#         """
#         ---
#         Built using 🤗 Transformers + PyTorch + Gradio
#         """
#     )


# if __name__ == "__main__":
#     demo.launch(share=True)


# import torch
# import gradio as gr
# from PIL import Image
# from transformers import AutoImageProcessor, AutoModelForImageClassification

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model_path = "C:\\Users\\ayush\\Deep_learning_works\\Hugging_Face\\Flower_Classifier\\model\\convnext_flower102_model"

# processor = AutoImageProcessor.from_pretrained(model_path)
# model = AutoModelForImageClassification.from_pretrained(model_path)
# model.to(device)
# model.eval()

# flower_names = [
#     "pink primrose", "hard-leaved pocket orchid", "canterbury bells",
#     "sweet pea", "english marigold", "tiger lily", "moon orchid",
#     "bird of paradise", "monkshood", "globe thistle",
#     "snapdragon", "colt's foot", "king protea",
#     "spear thistle", "yellow iris", "globe-flower",
#     "purple coneflower", "peruvian lily", "balloon flower",
#     "giant white arum lily", "fire lily", "pincushion flower",
#     "fritillary", "red ginger", "grape hyacinth",
#     "corn poppy", "prince of wales feathers", "stemless gentian",
#     "artichoke", "sweet william", "carnation",
#     "garden phlox", "love in the mist", "mexican aster",
#     "alpine sea holly", "ruby-lipped cattleya", "cape flower",
#     "great masterwort", "siam tulip", "lenten rose",
#     "barbeton daisy", "daffodil", "sword lily",
#     "poinsettia", "bolero deep blue", "wallflower",
#     "marigold", "buttercup", "oxeye daisy",
#     "common dandelion", "petunia", "wild pansy",
#     "primula", "sunflower", "pelargonium",
#     "bishop of llandaff", "gaura", "geranium",
#     "orange dahlia", "pink-yellow dahlia", "cautleya spicata",
#     "japanese anemone", "black-eyed susan", "silverbush",
#     "californian poppy", "osteospermum", "spring crocus",
#     "bearded iris", "windflower", "tree poppy",
#     "gazania", "azalea", "water lily",
#     "rose", "thorn apple", "morning glory",
#     "passion flower", "lotus", "toad lily",
#     "anthurium", "frangipani", "clematis",
#     "hibiscus", "columbine", "desert-rose",
#     "tree mallow", "magnolia", "cyclamen",
#     "watercress", "canna lily", "hippeastrum",
#     "bee balm", "ball moss", "foxglove",
#     "bougainvillea", "camellia", "mallow",
#     "mexican petunia", "bromelia", "blanket flower",
#     "trumpet creeper", "blackberry lily"

# ]


# def predict(image):
#     image = image.convert("RGB")
#     inputs = processor(images=image, return_tensors="pt").to(device)

#     with torch.no_grad():
#         outputs = model(**inputs)
#         probs = torch.softmax(outputs.logits, dim=1)

#     top5 = torch.topk(probs, 5)

#     results = {}

#     for i in range(5):
#         class_idx = int(top5.indices[0][i])
#         score = float(top5.values[0][i])

#         # Safe guard (should never trigger now)
#         if 0 <= class_idx < len(flower_names):
#             label_name = flower_names[class_idx]
#         else:
#             label_name = f"class_{class_idx}"

#         results[label_name] = score

#     return results


# demo = gr.Interface(
#     fn=predict,
#     inputs=gr.Image(type="pil"),
#     outputs=gr.Label(num_top_classes=5),
#     title="Flower Classifier 🌸",
#     description="Upload a flower image to classify it."
# )

# demo.launch()

import torch
import gradio as gr
import pandas as pd
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
import os

# ----------------------------
# Device Setup
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path = r"C:\Users\ayush\Deep_learning_works\Hugging_Face\Flower_Classifier\model\convnext_flower102_model"

# ----------------------------
# Load Model (Spinner Enabled Later in UI)
# ----------------------------
processor = AutoImageProcessor.from_pretrained(model_path)
model = AutoModelForImageClassification.from_pretrained(model_path)
model.to(device)
model.eval()

# ----------------------------
# Flower Labels
# ----------------------------
flower_names = [
    "pink primrose","hard-leaved pocket orchid","canterbury bells","sweet pea",
    "english marigold","tiger lily","moon orchid","bird of paradise","monkshood",
    "globe thistle","snapdragon","colt's foot","king protea","spear thistle",
    "yellow iris","globe-flower","purple coneflower","peruvian lily","balloon flower",
    "giant white arum lily","fire lily","pincushion flower","fritillary","red ginger",
    "grape hyacinth","corn poppy","prince of wales feathers","stemless gentian",
    "artichoke","sweet william","carnation","garden phlox","love in the mist",
    "mexican aster","alpine sea holly","ruby-lipped cattleya","cape flower",
    "great masterwort","siam tulip","lenten rose","barbeton daisy","daffodil",
    "sword lily","poinsettia","bolero deep blue","wallflower","marigold",
    "buttercup","oxeye daisy","common dandelion","petunia","wild pansy",
    "primula","sunflower","pelargonium","bishop of llandaff","gaura","geranium",
    "orange dahlia","pink-yellow dahlia","cautleya spicata","japanese anemone",
    "black-eyed susan","silverbush","californian poppy","osteospermum",
    "spring crocus","bearded iris","windflower","tree poppy","gazania",
    "azalea","water lily","rose","thorn apple","morning glory",
    "passion flower","lotus","toad lily","anthurium","frangipani",
    "clematis","hibiscus","columbine","desert-rose","tree mallow",
    "magnolia","cyclamen","watercress","canna lily","hippeastrum",
    "bee balm","ball moss","foxglove","bougainvillea","camellia",
    "mallow","mexican petunia","bromelia","blanket flower",
    "trumpet creeper","blackberry lily"
]

# ----------------------------
# Prediction Function
# ----------------------------
def predict(image):
    if image is None:
        return None, "", None

    image = image.convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)

    top5 = torch.topk(probs, 5)

    results = {}
    rows = []

    for i in range(5):
        idx = int(top5.indices[0][i])
        score = float(top5.values[0][i])
        label = flower_names[idx] if idx < len(flower_names) else f"class_{idx}"

        confidence = round(score * 100, 2)
        results[label] = score
        rows.append([label, confidence])

    best_label = rows[0][0]
    best_score = rows[0][1]

    summary = f"Top Prediction: {best_label} ({best_score}%)"

    # Create downloadable CSV report
    df = pd.DataFrame(rows, columns=["Flower", "Confidence (%)"])
    report_path = "prediction_report.csv"
    df.to_csv(report_path, index=False)

    return results, summary, report_path


# ----------------------------
# Clean Professional CSS
# ----------------------------
custom_css = """
body {
    background: linear-gradient(135deg, #1f2937, #111827);
}

.gradio-container {
    max-width: 1000px !important;
}

.glow-button {
    border-radius: 8px !important;
    font-weight: 600 !important;
    transition: 0.3s ease !important;
}

.glow-button:hover {
    transform: scale(1.04);
    box-shadow: 0 0 12px #3b82f6;
}

footer {visibility: hidden;}
"""

# ----------------------------
# UI
# ----------------------------
with gr.Blocks(css=custom_css, title="Flower AI Classifier") as demo:

    gr.Markdown("# Flower Image Classification System")
    gr.Markdown("Upload a flower image to classify it using a trained ConvNeXt deep learning model.")

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Upload Image")
            analyze_btn = gr.Button("Analyze Image", elem_classes="glow-button")
            clear_btn = gr.Button("Clear Results")

        with gr.Column():
            output_label = gr.Label(num_top_classes=5, label="Top 5 Predictions")
            summary_text = gr.Markdown()
            download_file = gr.File(label="Download Prediction Report")

    analyze_btn.click(
        predict,
        inputs=image_input,
        outputs=[output_label, summary_text, download_file],
        show_progress=True  # Spinner while predicting
    )

    clear_btn.click(
        lambda: (None, None, "", None),
        outputs=[image_input, output_label, summary_text, download_file]
    )

demo.launch()
