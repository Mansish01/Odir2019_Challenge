import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
from src.dataloader import val_dataloader
from models.models import Dense161Model
from sklearn.metrics import precision_score
from sklearn.metrics import precision_score

device = torch.device("cpu")

def evaluate_model(model, val_loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    for test_images, test_labels in tqdm(val_loader):
        test_images, test_labels = test_images.to(device), test_labels
        #.to(device)
        test_model_out = model(test_images)
        test_labels = test_labels.to(test_model_out.device)
        test_pred = torch.argmax(test_model_out, dim=1)

        all_preds.extend(test_pred.cpu().numpy())
        all_labels.extend(test_labels.cpu().numpy())

    return all_labels, all_preds

def calculate_metrics(all_labels, all_preds):
   
    precision = precision_score(all_labels, all_preds, average='micro')
    conf_matrix = confusion_matrix(all_labels, all_preds)
    class_report = classification_report(all_labels, all_preds)

    return conf_matrix, class_report

def plot_confusion_matrix(conf_matrix):
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=True,
        xticklabels=["N", "D", "A", "G", "C", "H", "M", "O"],
        yticklabels=["N", "D", "A", "G", "C", "H", "M", "O"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig(f"artifacts/confusion_matrix.png")
    plt.close()

def save_classification_report(class_report):
    report_path = f"artifacts/classification_report.txt"
    with open(report_path, "w") as report_file:
        report_file.write("Classification Report:\n\n")
        report_file.write(class_report)

if __name__ == "__main__":
    # Load model
    model = Dense161Model(num_labels=8)

    # Load checkpoint
    checkpoint_path = r"artifacts/best_model.pth"
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Evaluate the model
    all_labels, all_preds = evaluate_model(model, val_dataloader, device)

    # Calculate metrics
    conf_matrix, class_report = calculate_metrics(all_labels, all_preds)

    # Plot confusion matrix
    plot_confusion_matrix(conf_matrix)

    # Save classification report
    save_classification_report(class_report)