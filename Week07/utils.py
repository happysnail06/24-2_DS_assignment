import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import numpy as np
import torch
from PIL import Image



device = "cuda" if torch.cuda.is_available() else "cpu"
# Preprocess data
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, preprocess):
        self.dataset = dataset
        self.preprocess = preprocess

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.dataset[idx]['image']
        label = self.dataset[idx]['label']
        image = self.preprocess(Image.fromarray(np.array(image)))  # Adjust if not PIL Image
        return image, label
    
def plot_confidence_and_accuracy(probs, labels, n_bins=100):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
    bin_acc = []
    bin_conf = []

    for bin_lower, bin_upper in zip(bin_boundaries[:-1], bin_boundaries[1:]):
        # Determine whether each probability falls in the current bin
        max_probs = probs.max(axis=1)  # Get the maximum probability for each sample
        in_bin = (max_probs >= bin_lower) & (max_probs < bin_upper)

        if np.sum(in_bin) > 0:
            # Compute average confidence and accuracy for the current bin
            avg_confidence_in_bin = np.mean(max_probs[in_bin])
            accuracy_in_bin = np.mean(labels[in_bin] == np.argmax(probs[in_bin], axis=1))
            bin_conf.append(avg_confidence_in_bin)
            bin_acc.append(accuracy_in_bin)
        else:
            bin_conf.append(0)
            bin_acc.append(0)

    plt.figure(figsize=(12, 5))
    
    # Plot confidence
    plt.subplot(1, 2, 1)
    plt.bar(bin_centers, bin_conf, width=0.05, alpha=0.7, label="Confidence")
    plt.xlabel("Confidence")
    plt.ylabel("Average Confidence")
    plt.title("Model Confidence Distribution")
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.bar(bin_centers, bin_acc, width=0.05, alpha=0.7, label="Accuracy")
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title("Model Accuracy per Confidence Bin")
    plt.legend()

    plt.show()
    plt.savefig('confidence_accuracy.png')
    plt.close()


def visualize_embeddings_with_tsne(model, dataloader):
    model.eval()
    embeddings = []
    labels_list = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            features = model.clip_model.encode_image(images).cpu().numpy()
            embeddings.append(features)
            labels_list.extend(labels.numpy())

    embeddings = np.concatenate(embeddings)
    labels_list = np.array(labels_list)

    # Standardize embeddings before applying t-SNE
    scaler = StandardScaler()
    embeddings = scaler.fit_transform(embeddings)

    tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
    reduced_embeddings = tsne.fit_transform(embeddings)

    # Plot t-SNE results
    plt.figure(figsize=(8, 8))
    scatter = plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=labels_list, cmap='tab10', alpha=0.7)
    plt.colorbar(scatter, label='Class')
    plt.title('t-SNE Visualization of Embeddings')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.show()
    plt.savefig('tsne_embeddings.png')
    plt.close()


def compute_ece(probs, labels, n_bins=100):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (probs >= bin_lower) & (probs < bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            # Ensure the indexing matches the array dimensions
            in_bin_indices = np.where(in_bin)[0]  # Get the indices where in_bin is True
            
            # Calculate accuracy and confidence only for valid indices
            if len(in_bin_indices) > 0:
                accuracy_in_bin = np.mean(labels[in_bin_indices] == probs[in_bin_indices].argmax(axis=1))
                avg_confidence_in_bin = np.mean(probs[in_bin])
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return ece