import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pickle
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="Neural Storyteller", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
.main {background-color: #f8f9fa;}
.stButton>button {
    width: 100%;
    background-color: #2c3e50;
    color: white;
    border-radius: 8px;
    padding: 12px;
    font-size: 16px;
    border: none;
    transition: 0.3s;
}
.stButton>button:hover {background-color: #34495e;}
.metric-card {
    background: white;
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    margin: 10px 0;
}
.caption-box {
    background: #ecf0f1;
    padding: 15px;
    border-radius: 8px;
    border-left: 4px solid #3498db;
    margin: 10px 0;
    font-size: 16px;
}
</style>
""", unsafe_allow_html=True)

class Encoder(nn.Module):
    def __init__(self, feature_size=2048, hidden_size=512):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.fc = nn.Linear(feature_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
    def forward(self, features):
        hidden = self.fc(features)
        hidden = self.relu(hidden)
        hidden = self.dropout(hidden)
        return hidden

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=1, pad_idx=0):
        super(Decoder, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True, dropout=0.5 if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)
    def forward(self, captions, hidden):
        embeddings = self.embedding(captions)
        embeddings = self.dropout(embeddings)
        lstm_out, hidden = self.lstm(embeddings, hidden)
        outputs = self.fc(lstm_out)
        return outputs, hidden

class Seq2SeqModel(nn.Module):
    def __init__(self, encoder, decoder, hidden_size, num_layers):
        super(Seq2SeqModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.hidden_size = hidden_size
        self.num_layers = num_layers
    def forward(self, image_features, captions):
        batch_size = image_features.size(0)
        encoder_hidden = self.encoder(image_features)
        h_0 = encoder_hidden.unsqueeze(0).repeat(self.num_layers, 1, 1)
        c_0 = torch.zeros_like(h_0)
        outputs, _ = self.decoder(captions, (h_0, c_0))
        return outputs

def beam_search(model, image_features, max_length, start_token, end_token, idx2word, beam_width=3, device='cpu'):
    model.eval()
    with torch.no_grad():
        encoder_hidden = model.encoder(image_features)
        h = encoder_hidden.unsqueeze(0).repeat(model.num_layers, 1, 1)
        c = torch.zeros_like(h)
        sequences = [[[start_token], 0.0, h, c]]
        for _ in range(max_length):
            all_candidates = []
            for seq, score, h_state, c_state in sequences:
                if seq[-1] == end_token:
                    all_candidates.append([seq, score, h_state, c_state])
                    continue
                current_word = torch.tensor([[seq[-1]]]).to(device)
                output, (new_h, new_c) = model.decoder(current_word, (h_state, c_state))
                probs = torch.log_softmax(output[0, 0], dim=0)
                top_probs, top_indices = probs.topk(beam_width)
                for i in range(beam_width):
                    candidate = [seq + [top_indices[i].item()], score + top_probs[i].item(), new_h, new_c]
                    all_candidates.append(candidate)
            ordered = sorted(all_candidates, key=lambda x: x[1], reverse=True)
            sequences = ordered[:beam_width]
        return sequences[0][0]

@st.cache_resource
def load_model():
    device = torch.device('cpu')
    try:
        checkpoint = torch.load('final_model.pth', map_location=device)
        word2idx = checkpoint['word2idx']
        idx2word = checkpoint['idx2word']
        vocab = checkpoint['vocab']
        max_length = checkpoint['max_length']
        vocab_size = len(vocab)
        encoder = Encoder(2048, 512)
        decoder = Decoder(vocab_size, 256, 512, 1, pad_idx=0)
        model = Seq2SeqModel(encoder, decoder, 512, 1)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model, word2idx, idx2word, max_length, device
    except:
        return None, None, None, None, None

def generate_caption(image, model, word2idx, idx2word, max_length, device):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    img_tensor = transform(image).unsqueeze(0).to(device)
    resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    resnet = nn.Sequential(*list(resnet.children())[:-1]).to(device)
    resnet.eval()
    with torch.no_grad():
        features = resnet(img_tensor).view(1, -1)
    caption_indices = beam_search(model, features, max_length, word2idx['<start>'], word2idx['<end>'], idx2word, beam_width=5, device=device)
    caption = ' '.join([idx2word[idx] for idx in caption_indices if idx not in [0, word2idx['<start>'], word2idx['<end>']]])
    return caption

st.title("🎨 Neural Storyteller")
st.markdown("### Image Captioning with Seq2Seq Architecture")

model, word2idx, idx2word, max_length, device = load_model()

if model is None:
    st.error("⚠️ Model not found. Please ensure 'final_model.pth' is in the same directory.")
else:
    tab1, tab2, tab3 = st.tabs(["📸 Generate Caption", "📊 Model Performance", "ℹ️ About"])
    
    with tab1:
        col1, col2 = st.columns([1, 1])
        with col1:
            uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'], label_visibility="collapsed")
            if uploaded_file:
                image = Image.open(uploaded_file).convert('RGB')
                st.image(image, caption="Uploaded Image", use_container_width=True)
        with col2:
            if uploaded_file:
                with st.spinner("Generating caption..."):
                    caption = generate_caption(image, model, word2idx, idx2word, max_length, device)
                st.markdown(f'<div class="caption-box">📝 <b>Generated Caption:</b><br>{caption}</div>', unsafe_allow_html=True)
    
    with tab2:
        st.markdown("### 📈 Training Results")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("BLEU-4 (Greedy)", "0.2041", help="Measures n-gram overlap with reference captions")
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("BLEU-4 (Beam)", "0.2229", help="Improved score using beam search")
            st.markdown('</div>', unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Vocabulary Size", f"{len(idx2word)}", help="Total unique words in vocabulary")
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("### 🔧 Model Architecture")
        st.markdown("""
        <div class="metric-card">
        <ul style="margin: 0; padding-left: 20px;">
            <li><b>Encoder:</b> ResNet50 → Linear(2048 → 512)</li>
            <li><b>Decoder:</b> Embedding(256) → LSTM(512) → Linear(vocab_size)</li>
            <li><b>Training:</b> 10 epochs, Adam optimizer, CrossEntropyLoss</li>
            <li><b>Dataset:</b> Flickr30k (31,783 images, 158,915 captions)</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with tab3:
        st.markdown("### 📖 Project Overview")
        st.markdown("""
        <div class="metric-card">
        <h4>Neural Storyteller - Image Captioning System</h4>
        <p>This project implements a Sequence-to-Sequence (Seq2Seq) model for automatic image captioning using PyTorch.</p>
        
        <h4>Key Features:</h4>
        <ul>
            <li>Pre-trained ResNet50 for image feature extraction</li>
            <li>LSTM-based decoder for caption generation</li>
            <li>Beam search for improved caption quality</li>
            <li>Trained on Flickr30k dataset</li>
        </ul>
        
        <h4>Technologies Used:</h4>
        <ul>
            <li>PyTorch, Torchvision</li>
            <li>NLTK for text processing</li>
            <li>Streamlit for web interface</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("<p style='text-align: center; color: #7f8c8d;'>Generative AI Assignment | AI4009 | Spring 2026</p>", unsafe_allow_html=True)
