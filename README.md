# Neural Storyteller - Image Captioning App

A minimalistic Streamlit web app for image captioning using Seq2Seq architecture.

## Features
- Clean, modern UI
- Real-time caption generation
- Model performance metrics
- Project documentation

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Place your trained model file (`final_model.pth`) in the same directory as `app.py`

3. Run the app:
```bash
streamlit run app.py
```

## Usage
1. Navigate to the "Generate Caption" tab
2. Upload an image (JPG, JPEG, or PNG)
3. View the generated caption

## Model Details
- **Architecture**: ResNet50 Encoder + LSTM Decoder
- **Dataset**: Flickr30k (31,783 images)
- **BLEU-4 Score**: 0.2229 (Beam Search)
- **Vocabulary**: ~8000-10000 words

## Project Structure
```
.
├── app.py                 # Streamlit application
├── requirements.txt       # Python dependencies
├── final_model.pth        # Trained model (you need to add this)
└── README.md             # This file
```