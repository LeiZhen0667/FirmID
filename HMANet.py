import os
import re
import torch
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_curve, auc, precision_recall_fscore_support, accuracy_score
import torch.nn as nn
import torch.optim as optim
from gensim.models import KeyedVectors
from bs4 import BeautifulSoup
from torch.nn import functional as F
import random
from transformers import BertTokenizer, BertModel
import warnings
import pandas as pd
from datetime import datetime
from torch.utils.data import DataLoader, TensorDataset
import math
warnings.filterwarnings('ignore')

# Set all random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Check for available GPU
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

# Text preprocessing
def preprocess_text_for_html(text):
    cleaned_text = text.replace('\n', ' ')  # Remove newline characters
    cleaned_text = re.sub(r'\s{2,}', ' ', cleaned_text)  # Remove consecutive spaces
    return cleaned_text

# Load pretrained FastText model
def load_pretrained_fasttext_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model = KeyedVectors.load_word2vec_format(model_path, binary=False)
    return model

# Get weighted FastText and character TF-IDF embeddings
def get_weighted_embeddings(texts, fasttext_model, max_features=1000, ngram_range=(1, 5)):
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range, analyzer='char')
    tfidf_matrix = vectorizer.fit_transform(texts)

    embeddings = []
    for i, text in enumerate(texts):
        words = text.split()
        embedding = np.zeros(fasttext_model.vector_size)
        tfidf_vector = tfidf_matrix[i]
        for word in words:
            if word in fasttext_model.key_to_index:
                # Weight word using TF-IDF weight
                tfidf_weight = tfidf_vector[
                    0, vectorizer.vocabulary_.get(word, 0)] if word in vectorizer.vocabulary_ else 0
                embedding += fasttext_model[word] * tfidf_weight
        embeddings.append(embedding)
    return torch.tensor(embeddings, dtype=torch.float32)

# DOM structure extraction
def extract_dom_structure(html_content):
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        nodes = soup.find_all(True)
        node_depths = [len(list(node.parents)) for node in nodes]
        node_types = [node.name for node in nodes]

        node_type_set = list(set(node_types))
        node_type_to_id = {node_type: i for i, node_type in enumerate(node_type_set)}
        node_type_ids = [node_type_to_id[node_type] for node_type in node_types]

        # Ensure dom_feature_vector length is 300
        combined = node_depths + node_type_ids
        if len(combined) < 300:
            combined += [0] * (300 - len(combined))
        else:
            combined = combined[:300]
        dom_feature_vector = np.array(combined)
        return dom_feature_vector
    except Exception as e:
        print(f"DOM structure extraction failed: {e}")
        return np.zeros(300)  # Default length

# Extract code snippets
def extract_code_snippets(html_content):
    """
    Extract JS and CSS code snippets from HTML content.
    """
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        scripts = soup.find_all('script')
        styles = soup.find_all('style')
        code_snippets = []

        for script in scripts:
            if script.string:
                code_snippets.append(script.string)

        for style in styles:
            if style.string:
                code_snippets.append(style.string)

        return '\n'.join(code_snippets)
    except Exception as e:
        print(f"Code snippet extraction failed: {e}")
        return ""

# Encode code snippets
def encode_code_with_bert(code_snippets, tokenizer, model, device):
    """
    Encode code snippets using BERT, return fixed-dimension embeddings.
    """
    # Handle empty code snippets
    non_empty = [code for code in code_snippets if code.strip()]
    if not non_empty:
        # Return zero vectors
        return torch.zeros(len(code_snippets), model.config.hidden_size).to(device)

    # Tokenize the code snippets
    inputs = tokenizer(non_empty, return_tensors='pt', padding=True, truncation=True, max_length=512)
    inputs = {key: val.to(device) for key, val in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        # Use [CLS] token hidden state as embedding
        embeddings = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_dim]

    # Create full embeddings tensor, empty snippets as zero vectors
    full_embeddings = torch.zeros(len(code_snippets), model.config.hidden_size).to(device)
    j = 0
    for i, code in enumerate(code_snippets):
        if code.strip():
            full_embeddings[i] = embeddings[j]
            j += 1
    return full_embeddings

# Define bilinear pooling function
def bilinear_pooling(x, y):
    """
    x: [batch_size, seq_len, hidden_dim]
    y: [batch_size, seq_len, hidden_dim]
    Returns: [batch_size, hidden_dim * hidden_dim]
    """
    bilinear = torch.bmm(x.transpose(1, 2), y)  # [batch_size, hidden_dim, hidden_dim]
    bilinear = bilinear.view(x.size(0), -1)  # [batch_size, hidden_dim * hidden_dim]
    return bilinear

# Weight initialization function
def initialize_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

class DynamicMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(DynamicMultiHeadAttention, self).__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Linear transformation matrices
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        self.W_o = nn.Linear(embed_dim, embed_dim)
        
        # Dynamic scaling factor β network
        self.beta_net = nn.Sequential(
            nn.Linear(embed_dim, num_heads),
            nn.Sigmoid()
        )
        
        # Scaling factor
        self.scale_factor = math.sqrt(self.head_dim)
        
    def forward(self, query, key=None, value=None):
        # Use query if key/value not provided
        if key is None:
            key = query
        if value is None:
            value = key
            
        batch_size = query.size(0)
        
        # Compute dynamic scaling factor β
        pooled = torch.mean(query, dim=1)  # Global average pooling [batch_size, embed_dim]
        betas = self.beta_net(pooled)  # [batch_size, num_heads]
        betas = betas.unsqueeze(-1).unsqueeze(-1)  # [batch_size, num_heads, 1, 1]
        
        # Linear transformation and split into multiple heads
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale_factor
        
        # Apply dynamic scaling factor
        scaled_scores = scores * betas
        
        # Compute attention weights
        attn_weights = F.softmax(scaled_scores, dim=-1)
        
        # Apply attention weights to value
        attn_output = torch.matmul(attn_weights, V)
        
        # Combine multiple heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.embed_dim)
        
        # Output projection
        output = self.W_o(attn_output)
        
        return output

class SCMA(nn.Module):
    def __init__(self, embed_dim, num_heads, ffn_hidden_dim):
        super(SCMA, self).__init__()
        # Replace original self-attention with dynamic multi-head attention
        self.self_attn_mod1 = DynamicMultiHeadAttention(embed_dim, num_heads)
        self.self_attn_mod2 = DynamicMultiHeadAttention(embed_dim, num_heads)

        self.cross_attn_mod1_to_mod2 = DynamicMultiHeadAttention(embed_dim, num_heads)
        self.cross_attn_mod2_to_mod1 = DynamicMultiHeadAttention(embed_dim, num_heads)

        self.linear = nn.Linear(embed_dim * 2, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_hidden_dim),
            nn.ReLU(),
            nn.Linear(ffn_hidden_dim, embed_dim)
        )
        self.ffn_layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, mod1, mod2):
        # Self-Attention for each modality
        mod1_self = self.self_attn_mod1(mod1)  # (batch, seq_mod1, embed_dim)
        mod2_self = self.self_attn_mod2(mod2)  # (batch, seq_mod2, embed_dim)

        # Cross-Attention
        cross_mod1 = self.cross_attn_mod1_to_mod2(mod1_self, mod2_self, mod2_self)  # Mod1 queries, Mod2 keys/values
        cross_mod2 = self.cross_attn_mod2_to_mod1(mod2_self, mod1_self, mod1_self)  # Mod2 queries, Mod1 keys/values

        # Combine cross-attention outputs
        combined = torch.cat([cross_mod1, cross_mod2], dim=-1)  # (batch, seq, embed_dim * 2)
        combined = self.linear(combined)  # (batch, seq, embed_dim)
        combined = self.layer_norm(combined + mod1_self + mod2_self)  # Residual connection

        # Feed Forward Network
        fused = self.ffn(combined)  # (batch, seq, embed_dim)
        fused = self.ffn_layer_norm(fused + combined)  # Residual connection

        return fused  # (batch, seq, embed_dim)

class DCMF(nn.Module):
    def __init__(self, embed_dim, ffn_hidden_dim):
        super(DCMF, self).__init__()
        self.bilinear = nn.Bilinear(embed_dim, embed_dim, embed_dim)
        self.sigmoid = nn.Sigmoid()

        # Use dynamic multi-head attention
        self.cross_attn1 = DynamicMultiHeadAttention(embed_dim, num_heads=8)
        self.cross_attn2 = DynamicMultiHeadAttention(embed_dim, num_heads=8)

        self.linear = nn.Linear(embed_dim * 2, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_hidden_dim),
            nn.ReLU(),
            nn.Linear(ffn_hidden_dim, embed_dim)
        )
        self.ffn_layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, z1, z2):
        """
        z1: (batch, seq, embed_dim)
        z2: (batch, seq, embed_dim)
        """
        # Bilinear Pooling (Outer Product)
        bilinear_feat = z1.unsqueeze(2) * z2.unsqueeze(1)  # (batch, seq1, seq2, embed_dim)
        # Average pooling over seq1 and seq2
        bilinear_feat = bilinear_feat.mean(dim=1).mean(dim=1)  # (batch, embed_dim)

        # Generate K_b and V_b using bilinear pooling and bilinear_feat
        K_b = self.sigmoid(self.bilinear(z1.mean(dim=1), z2.mean(dim=1)) + bilinear_feat)  # (batch, embed_dim)
        V_b = self.sigmoid(self.bilinear(z1.mean(dim=1), z2.mean(dim=1)) + bilinear_feat)  # (batch, embed_dim)

        # Expand K_b and V_b to match sequence length
        seq_length = z1.size(1)
        K_b = K_b.unsqueeze(1).repeat(1, seq_length, 1)  # (batch, seq, embed_dim)
        V_b = V_b.unsqueeze(1).repeat(1, seq_length, 1)  # (batch, seq, embed_dim)

        # High-order Attention
        cross_s1 = self.cross_attn1(z1, K_b, V_b)  # (batch, seq, embed_dim)
        cross_s2 = self.cross_attn2(z2, K_b, V_b)  # (batch, seq, embed_dim)

        # Combine cross-attention outputs
        combined = torch.cat([cross_s1, cross_s2], dim=-1)  # (batch, seq, embed_dim * 2)
        combined = self.linear(combined)  # (batch, seq, embed_dim)
        combined = self.layer_norm(combined + z1 + z2)  # Residual connection

        # Feed Forward Network
        fused = self.ffn(combined)  # (batch, seq, embed_dim)
        fused = self.ffn_layer_norm(fused + combined)  # Residual connection

        return fused  # (batch, seq, embed_dim)

class CMFL(nn.Module):
    def __init__(self, embed_dim, num_heads, ffn_hidden_dim):
        super(CMFL, self).__init__()
        # Two SCMA modules
        self.scma1 = SCMA(embed_dim, num_heads, ffn_hidden_dim)
        self.scma2 = SCMA(embed_dim, num_heads, ffn_hidden_dim)

        # Two DCMF modules
        self.dcmf1 = DCMF(embed_dim, ffn_hidden_dim)
        self.dcmf2 = DCMF(embed_dim, ffn_hidden_dim)

    def forward(self, mod1, mod2):
        # Forward pass for two SCMA modules
        scma1 = self.scma1(mod1, mod2)  # SCMA-1
        scma2 = self.scma2(mod2, mod1)  # SCMA-2

        # Forward pass for two DCMF modules
        dcmf1 = self.dcmf1(scma1, scma2)  # DCMF-1
        dcmf2 = self.dcmf2(scma2, scma1)  # DCMF-2

        # Element-wise addition for current CMFL output
        output = dcmf1 + dcmf2  # (batch, seq, embed_dim)

        return output  # (batch, seq, embed_dim)

class HCMA_Net(nn.Module):
    def __init__(self, dom_dim, text_dim, code_dim, embed_dim=256, num_heads=8, ffn_hidden_dim=512):
        super(HCMA_Net, self).__init__()
        # Projection layers to map different modalities to same dimension
        self.dom_proj = nn.Linear(dom_dim, embed_dim)
        self.text_proj = nn.Linear(text_dim, embed_dim)
        self.code_proj = nn.Linear(code_dim, embed_dim)

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ffn_hidden_dim = ffn_hidden_dim

        # Define 4 CMFL layers
        self.cmfl1 = CMFL(embed_dim, num_heads, ffn_hidden_dim)
        self.cmfl2 = CMFL(embed_dim, num_heads, ffn_hidden_dim)
        self.cmfl3 = CMFL(embed_dim, num_heads, ffn_hidden_dim)
        self.cmfl4 = CMFL(embed_dim, num_heads, ffn_hidden_dim)

    def forward(self, dom, text, code):
        # Projection
        dom = self.dom_proj(dom)    # [batch, embed_dim]
        text = self.text_proj(text)  # [batch, embed_dim]
        code = self.code_proj(code)  # [batch, embed_dim]

        # Add sequence dimension
        dom = dom.unsqueeze(1)      # [batch, 1, embed_dim]
        text = text.unsqueeze(1)    # [batch, 1, embed_dim]
        code = code.unsqueeze(1)    # [batch, 1, embed_dim]

        # Pass through CMFL layers with modality pairs
        z1 = self.cmfl1(dom, text)  # [batch, 1, embed_dim]
        z2 = self.cmfl2(z1, text)   # [batch, 1, embed_dim]
        z3 = self.cmfl3(z2, code)   # [batch, 1, embed_dim]
        z4 = self.cmfl4(z3, code)   # [batch, 1, embed_dim]

        # Final fusion
        final_features = z4.squeeze(1)  # [batch, embed_dim]
        return final_features

# Define fully connected neural network classifier
class Classifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(Classifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, num_classes)
            # Removed Softmax for numerical stability with CrossEntropyLoss
        )

    def forward(self, x):
        return self.fc(x)

# Define combined model including HCMA_Net and Classifier
class CombinedModel(nn.Module):
    def __init__(self, hcma_net, classifier):
        super(CombinedModel, self).__init__()
        self.hcma_net = hcma_net
        self.classifier = classifier

    def forward(self, dom, text, code):
        features = self.hcma_net(dom, text, code)  # [batch, embed_dim]
        class_output = self.classifier(features)    # [batch, num_classes]
        return features, class_output

# Hard Negative Mining Contrastive Loss (HNMCL)
class HNMCLoss(nn.Module):
    def __init__(self, tau=0.07):
        super(HNMCLoss, self).__init__()
        self.tau = tau
        self.cosine_sim = nn.CosineSimilarity(dim=2)
    
    def forward(self, anchor_features, positive_features, negative_features):
        """
        anchor_features: [batch, hidden_dim]
        positive_features: [batch, hidden_dim]
        negative_features: [batch, hidden_dim]
        """
        # Calculate positive sample similarity
        pos_sim = self.cosine_sim(anchor_features.unsqueeze(1), positive_features.unsqueeze(0))  # [batch, batch]
        pos_sim = torch.diag(pos_sim) / self.tau  # [batch]
        
        # Calculate negative sample similarity
        neg_sim = self.cosine_sim(anchor_features.unsqueeze(1), negative_features.unsqueeze(0))  # [batch, batch]
        neg_sim = torch.diag(neg_sim) / self.tau  # [batch]
        
        # Calculate loss
        numerator = torch.exp(pos_sim)
        denominator = numerator + torch.exp(neg_sim)
        loss = -torch.log(numerator / denominator)
        
        return loss.mean()


def save_metrics(y_true, y_pred, y_pred_proba, label_encoder, output_dir):
    """
    Calculate and save various metrics
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')

    # Calculate ROC curves (one-vs-rest)
    n_classes = len(label_encoder.classes_)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    # New: Variables for macro-average ROC curve
    all_fpr = np.unique(np.concatenate([np.linspace(0, 1, 100)]))
    mean_tpr = np.zeros_like(all_fpr)

    # One-hot encode true labels
    y_true_onehot = np.zeros((len(y_true), n_classes))
    for i in range(len(y_true)):
        y_true_onehot[i, y_true[i]] = 1

    # Calculate ROC curves and ROC AUC for each class
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_onehot[:, i], y_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        
        # New: Interpolate and accumulate TPR values for macro average
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # New: Calculate macro-average ROC curve
    mean_tpr /= n_classes
    macro_auc = auc(all_fpr, mean_tpr)
    
    # New: Calculate micro-average ROC curve
    micro_fpr, micro_tpr, _ = roc_curve(y_true_onehot.ravel(), y_pred_proba.ravel())
    micro_auc = auc(micro_fpr, micro_tpr)

    # Save ROC curve data for each class
    for i in range(n_classes):
        roc_data = pd.DataFrame({
            'FPR': fpr[i],
            'TPR': tpr[i],
            'AUC': [roc_auc[i]] * len(fpr[i])
        })
        roc_data.to_csv(f'{output_dir}/roc_curve_class_{label_encoder.classes_[i]}_{timestamp}.csv', index=False)
    
    # New: Save macro-average ROC curve
    macro_roc_data = pd.DataFrame({
        'FPR': all_fpr,
        'TPR': mean_tpr,
        'AUC': [macro_auc] * len(all_fpr)
    })
    macro_roc_data.to_csv(f'{output_dir}/roc_curve_macro_avg_{timestamp}.csv', index=False)
    
    # New: Save micro-average ROC curve
    micro_roc_data = pd.DataFrame({
        'FPR': micro_fpr,
        'TPR': micro_tpr,
        'AUC': [micro_auc] * len(micro_fpr)
    })
    micro_roc_data.to_csv(f'{output_dir}/roc_curve_micro_avg_{timestamp}.csv', index=False)

    # Save overall metrics
    metrics_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1', 'Macro AUC', 'Micro AUC'],
        'Value': [accuracy, precision, recall, f1, macro_auc, micro_auc]
    })
    metrics_df.to_csv(f'{output_dir}/overall_metrics_{timestamp}.csv', index=False)

    # Print metrics
    print("\nClassification Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Macro AUC: {macro_auc:.4f}")
    print(f"Micro AUC: {micro_auc:.4f}")

    # Save per-class metrics
    per_class_precision, per_class_recall, per_class_f1, _ = precision_recall_fscore_support(y_true, y_pred)
    per_class_metrics = pd.DataFrame({
        'Class': label_encoder.classes_,
        'Precision': per_class_precision,
        'Recall': per_class_recall,
        'F1': per_class_f1,
        'AUC': [roc_auc[i] for i in range(n_classes)]
    })
    per_class_metrics.to_csv(f'{output_dir}/per_class_metrics_{timestamp}.csv', index=False)

def save_checkpoint(model, optimizer, epoch, metrics, model_name, output_dir):
    """
    Save model checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = os.path.join(output_dir, f'{model_name}_checkpoint_{timestamp}.pt')

    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")

    return checkpoint_path

# Create hard negative sample pairs
def create_hnm_pairs(features, labels, file_ids, batch_size, device):
    """
    Create hard negative sample pairs
    features: Current batch feature vectors [batch_size, hidden_dim]
    labels: Current batch labels [batch_size]
    file_ids: Current batch file identifiers [batch_size]
    batch_size: Batch size
    device: Device
    """
    # Normalize feature vectors for cosine similarity calculation
    features_norm = F.normalize(features, p=2, dim=1)
    
    # Compute similarity matrix
    sim_matrix = torch.mm(features_norm, features_norm.t())
    
    # Create masks
    same_file_mask = (file_ids.unsqueeze(1) == file_ids.unsqueeze(0))  # Same file
    same_class_mask = (labels.unsqueeze(1) == labels.unsqueeze(0))     # Same class
    diff_class_mask = ~same_class_mask                                  # Different class
    
    # Initialize storage
    anchors = []
    positives = []
    negatives = []
    
    for i in range(batch_size):
        # Positive samples: Same class, different file
        pos_mask = same_class_mask[i] & ~same_file_mask[i]
        if pos_mask.any():
            # Randomly select a positive sample
            pos_indices = torch.where(pos_mask)[0]
            j = random.choice(pos_indices.tolist())
            positives.append(j)
        else:
            # Skip current anchor if no positive found
            continue
        
        # Negative samples: Prefer hard negatives from same file but different class
        neg_mask_same_file = same_file_mask[i] & diff_class_mask[i]
        if neg_mask_same_file.any():
            # Select most similar negative from same file, different class
            neg_indices = torch.where(neg_mask_same_file)[0]
            sim_scores = sim_matrix[i, neg_indices]
            k = neg_indices[torch.argmax(sim_scores)]
            negatives.append(k)
        else:
            # If no same-file negatives, select globally most similar negative
            neg_mask_global = diff_class_mask[i]
            if neg_mask_global.any():
                neg_indices = torch.where(neg_mask_global)[0]
                sim_scores = sim_matrix[i, neg_indices]
                k = neg_indices[torch.argmax(sim_scores)]
                negatives.append(k)
            else:
                # Skip current anchor if no negative found
                continue
        
        anchors.append(i)
    
    # Convert to tensors
    if anchors:
        anchors = torch.tensor(anchors, dtype=torch.long, device=device)
        positives = torch.tensor(positives, dtype=torch.long, device=device)
        negatives = torch.tensor(negatives, dtype=torch.long, device=device)
        return anchors, positives, negatives
    else:
        return None, None, None

def train_classification_model(model, text_embeddings, dom_embeddings, code_embeddings, labels,
                               label_encoder, file_identifiers, hidden_dim=256, num_epochs=50, 
                               learning_rate=0.0001, batch_size=32, output_dir='metrics', 
                               checkpoint_dir='model_checkpoints', model_name='softmax_net',
                               tau=0.07, class_weight=1.0, contrastive_weight=0.5):
    """
    Train model with classification loss and HNMCL loss
    """
    # Create output directories
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Data splitting with stratification
    (X_train_text, X_test_text, 
     X_train_dom, X_test_dom, 
     X_train_code, X_test_code, 
     y_train, y_test, 
     train_file_ids, test_file_ids) = train_test_split(
        text_embeddings, dom_embeddings, code_embeddings, 
        labels, file_identifiers, test_size=0.2, 
        random_state=SEED, stratify=labels  # Use fixed seed
    )

    # Move data to device
    X_train_text, X_test_text = X_train_text.to(device), X_test_text.to(device)
    X_train_dom, X_test_dom = X_train_dom.to(device), X_test_dom.to(device)
    X_train_code, X_test_code = X_train_code.to(device), X_test_code.to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)
    
    # Convert file identifiers to tensors
    train_file_ids_tensor = torch.tensor(train_file_ids, dtype=torch.long).to(device)
    test_file_ids_tensor = torch.tensor(test_file_ids, dtype=torch.long).to(device)

    num_classes = len(label_encoder.classes_)

    # Instantiate classifier and combined model
    classifier = Classifier(input_dim=hidden_dim, num_classes=num_classes).to(device)
    combined_model = CombinedModel(model, classifier).to(device)

    # Define loss functions
    criterion_cls = nn.CrossEntropyLoss().to(device)  # Classification loss
    criterion_hnmc = HNMCLoss(tau=tau).to(device)    # HNMCL loss

    # Optimizer
    optimizer = optim.AdamW(combined_model.parameters(), lr=learning_rate, weight_decay=0.01)
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5,
                                                     patience=5, verbose=True)

    # Create data loader
    train_dataset = TensorDataset(X_train_text, X_train_dom, X_train_code, 
                                 y_train_tensor, train_file_ids_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    best_accuracy = 0.0
    best_checkpoint_path = None
    training_history = []
    no_improve_count = 0
    patience = 10  # Early stopping patience

    for epoch in range(num_epochs):
        combined_model.train()
        total_loss_cls = 0.0
        total_loss_hnmc = 0.0
        total_loss = 0.0
        
        for batch_idx, (batch_text, batch_dom, batch_code, batch_labels, batch_file_ids) in enumerate(train_loader):
            optimizer.zero_grad()

            # Forward pass
            features, class_outputs = combined_model(batch_dom, batch_text, batch_code)
            
            # Calculate classification loss
            loss_cls = criterion_cls(class_outputs, batch_labels)
            
            # Create hard negative sample pairs
            anchors, positives, negatives = create_hnm_pairs(
                features, batch_labels, batch_file_ids, 
                len(batch_labels), device
            )
            
            # Calculate HNMCL loss
            loss_hnmc = 0.0
            if anchors is not None:
                anchor_features = features[anchors]
                positive_features = features[positives]
                negative_features = features[negatives]
                loss_hnmc = criterion_hnmc(anchor_features, positive_features, negative_features)
            
            # Combine losses
            loss_total = class_weight * loss_cls + contrastive_weight * loss_hnmc
            
            # Backward pass
            loss_total.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(combined_model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Record losses
            total_loss_cls += loss_cls.item()
            total_loss_hnmc += loss_hnmc.item() if anchors is not None else 0.0
            total_loss += loss_total.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] Batch [{batch_idx}/{len(train_loader)}] - "
                      f"CLS Loss: {loss_cls.item():.4f} | "
                      f"HNMCL Loss: {loss_hnmc.item() if anchors is not None else 0.0:.4f} | "
                      f"Total Loss: {loss_total.item():.4f}")

        # Calculate average losses
        avg_loss_cls = total_loss_cls / len(train_loader)
        avg_loss_hnmc = total_loss_hnmc / len(train_loader)
        avg_loss = total_loss / len(train_loader)
        
        # Evaluation
        combined_model.eval()
        with torch.no_grad():
            _, test_outputs = combined_model(X_test_dom, X_test_text, X_test_code)
            _, predicted = torch.max(test_outputs, 1)
            current_accuracy = (predicted == y_test_tensor).float().mean().item()
            
            # Record detailed metrics
            metrics = {
                'epoch': epoch + 1,
                'loss_cls': avg_loss_cls,
                'loss_hnmc': avg_loss_hnmc,
                'total_loss': avg_loss,
                'test_accuracy': current_accuracy
            }
            training_history.append(metrics)
            
            print(f"Epoch [{epoch + 1}/{num_epochs}] - "
                  f"Avg CLS Loss: {avg_loss_cls:.4f} | "
                  f"Avg HNMCL Loss: {avg_loss_hnmc:.4f} | "
                  f"Avg Total Loss: {avg_loss:.4f} | "
                  f"Test Acc: {current_accuracy:.4f}")
            
            # Adjust learning rate
            scheduler.step(current_accuracy)
            
            # Save model if performance improves
            if current_accuracy > best_accuracy:
                best_accuracy = current_accuracy
                checkpoint_path = save_checkpoint(
                    combined_model,
                    optimizer,
                    epoch + 1,
                    metrics,
                    model_name,
                    checkpoint_dir
                )
                best_checkpoint_path = checkpoint_path
                no_improve_count = 0
            else:
                no_improve_count += 1
                
            # Early stopping
            if no_improve_count >= patience:
                print("Early stopping triggered")
                break

    # Save final model
    final_checkpoint_path = save_checkpoint(
        combined_model,
        optimizer,
        num_epochs,
        metrics,
        f"{model_name}_final",
        checkpoint_dir
    )

    # Save training history
    history_df = pd.DataFrame(training_history)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    history_df.to_csv(os.path.join(output_dir, f'training_history_{timestamp}.csv'), index=False)

    # Load best model for final evaluation
    if best_checkpoint_path:
        checkpoint = torch.load(best_checkpoint_path)
        combined_model.load_state_dict(checkpoint['model_state_dict'])

    # Final evaluation
    combined_model.eval()
    with torch.no_grad():
        _, test_outputs = combined_model(X_test_dom, X_test_text, X_test_code)
        y_pred_proba = F.softmax(test_outputs, dim=1).cpu().numpy()
        _, predicted = torch.max(test_outputs, 1)
        y_pred = predicted.cpu().numpy()
        y_true = y_test_tensor.cpu().numpy()

        # Save evaluation metrics
        save_metrics(y_true, y_pred, y_pred_proba, label_encoder, output_dir)

    print(f"\nTraining completed.")
    print(f"Best model saved at: {best_checkpoint_path}")
    print(f"Final model saved at: {final_checkpoint_path}")
    print(f"Training history saved in: {output_dir}")

    return combined_model, best_checkpoint_path, final_checkpoint_path

# Get web file list
def get_web_files(root_folder):
    web_files = []
    for root, _, files in os.walk(root_folder):
        for file in files:
            file_path = os.path.join(root, file)
            if file_path.endswith(('.html', '.htm', '.php', '.asp', '.js', '.css')):
                web_files.append(file_path)
    return web_files

# Filter out empty files
def filter_empty_files(files):
    return [file_path for file_path in files if os.path.getsize(file_path) > 0]

# Extract file identifier (relative path)
def extract_file_identifier(file_path, base_folder):
    return os.path.relpath(file_path, base_folder)

if __name__ == "__main__":
    # Load and preprocess data
    folder_paths = [
        # './dataset_demo/JNR3210_V1.1.0.14_1.0.14',
        # './dataset_demo/DIR816L_FW100b09',
        # './dataset_demo/DIR-300_fw_revb_205b03_ALL_de_20101123',
        # './dataset_demo/Archer_c5v2_us-up-ver3-17-1-P1[20150908-rel43260]',
        # './dataset_demo/ArcherC5v1_en_3_14_1_up_boot(141126)',
        # './dataset_demo/FW_EA6700_1.1.42.203057',
        # './dataset_demo/FW_RT_AC1900P_300438432799',
        # './dataset_demo/R7000_V1.0.4.18_1.1.52',
        # './dataset_demo/FW_EA6700_1.1.40.176451',
        # './dataset_demo/FW_EA6900_1.1.42.174776_prod.img'
        './10-dataset-test/JNR3210_V1.1.0.14_1.0.14',
        './10-dataset-test/DIR816L_FW100b09',
        './10-dataset-test/DIR-300_fw_revb_205b03_ALL_de_20101123',
        './10-dataset-test/Archer_c5v2_us-up-ver3-17-1-P1[20150908-rel43260]',
        './10-dataset-test/ArcherC5v1_en_3_14_1_up_boot(141126)',
        './10-dataset-test/FW_EA6700_1.1.42.203057',
        './10-dataset-test/FW_RT_AC1900P_300438432799',
        # './10-dataset-test/R7000_V1.0.4.18_1.1.52',
        './10-dataset-test/FW_EA6700_1.1.40.176451',
        # './10-dataset-test/FW_EA6900_1.1.42.174776_prod.img'
    ]

    fasttext_model_path = 'pretrain_model/wiki-news-300d-1M.vec'

    fasttext_model = load_pretrained_fasttext_model(fasttext_model_path)

    # Load pretrained BERT model
    BERT_MODEL_PATH = "/mnt/data/leizhen/FirmID-Web/Firm-ID/scripts/bert-base-uncase"
    bert_tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_PATH)
    bert_model = BertModel.from_pretrained(BERT_MODEL_PATH).to(device)

    texts = []
    dom_features = []
    code_snippets_list = []
    labels = []
    file_identifiers = []  # Store file identifiers
    file_id_mapping = {}   # File path to integer mapping
    next_file_id = 0       # Next file ID

    for path in folder_paths:
        files = get_web_files(path)
        files = filter_empty_files(files)

        for file in files:
            # Extract file identifier (relative path)
            file_identifier = extract_file_identifier(file, path)
            
            # Assign new ID if file identifier not seen
            if file_identifier not in file_id_mapping:
                file_id_mapping[file_identifier] = next_file_id
                next_file_id += 1
                
            file_identifiers.append(file_id_mapping[file_identifier])

            with open(file, 'r', encoding='iso-8859-1') as f:
                text = f.read()
            text = preprocess_text_for_html(text)
            texts.append(text)

            dom_feature = extract_dom_structure(text)
            dom_features.append(dom_feature)

            # Extract code snippets
            code_snippets = extract_code_snippets(text)
            code_snippets_list.append(code_snippets)

            labels.append(os.path.basename(path))

    # Encode labels
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)

    # Get text embeddings
    text_embeddings = get_weighted_embeddings(texts, fasttext_model)

    # Encode code snippets
    batch_size_bert = 16
    code_embeddings = []

    for i in range(0, len(code_snippets_list), batch_size_bert):
        batch_code = code_snippets_list[i:i + batch_size_bert]
        # Encode with BERT
        encoded_batch = encode_code_with_bert(batch_code, bert_tokenizer, bert_model, device)
        code_embeddings.append(encoded_batch)

    code_embeddings = torch.cat(code_embeddings, dim=0)  # [num_samples, hidden_dim]

    # Pad DOM features to consistent length
    max_dom_length = 300  # extract_dom_structure returns fixed 300-dim vector
    padded_dom_features = []
    for feature in dom_features:
        if len(feature) == 0:
            padded_feature = np.zeros(max_dom_length)
        else:
            padded_feature = feature  # Already fixed to 300
        padded_dom_features.append(padded_feature)

    dom_embeddings = torch.tensor(padded_dom_features, dtype=torch.float32)

    # Move all embeddings to device
    text_embeddings = text_embeddings.to(device)
    dom_embeddings = dom_embeddings.to(device)
    code_embeddings = code_embeddings.to(device)

    # Instantiate HCMA_Net
    hcma_net = HCMA_Net(
        dom_dim=dom_embeddings.size(1),
        text_dim=text_embeddings.size(1),
        code_dim=code_embeddings.size(1),
        embed_dim=256,
        num_heads=8,
        ffn_hidden_dim=512
    )
    hcma_net.apply(initialize_weights)  # Apply weight initialization
    hcma_net.to(device)

    # Train the model with classification and HNMCL
    trained_model = train_classification_model(
        model=hcma_net,
        text_embeddings=text_embeddings,
        dom_embeddings=dom_embeddings,
        code_embeddings=code_embeddings,
        labels=labels_encoded,
        label_encoder=label_encoder,
        file_identifiers=file_identifiers,  # Add file identifiers
        hidden_dim=256,
        num_epochs=50,
        learning_rate=0.0001,
        batch_size=32,
        output_dir='FirmID-metrics_output',
        checkpoint_dir='FirmID-model_checkpoints',
        tau=0.07,  # Temperature parameter
        class_weight=1.0,
        contrastive_weight=0.5
    )