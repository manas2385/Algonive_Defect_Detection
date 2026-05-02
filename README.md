## Defect Detection in Manufacturing using AI

### Overview
This project is a computer vision system that detects and classifies surface defects in manufacturing products using a Convolutional Neural Network (CNN).

### Dataset
NEU Metal Surface Defects dataset with multiple defect classes:
- crazing
- inclusion
- patches
- pitted_surface
- rolled_in_scale
- scratches

### Approach
- Converted Pascal VOC (XML) annotations into classification folders
- Trained a CNN model on labeled images
- Used softmax for multi-class classification
- Built a Streamlit app for real-time predictions

### Features
- Multi-class defect classification
- Image upload interface
- Confidence score display
- Top predictions output

### How to Run
pip install -r requirements.txt
python prepare_data.py
python train_defect.py
streamlit run app_defect.py

### Project Structure
Defect_Detection/
├── data/
├── train/
├── validation/
├── prepare_data.py
├── train_defect.py
├── app_defect.py
├── defect_model.h5
├── requirements.txt
├── README.md


### Model File

The trained model (.h5) is not included due to GitHub size limits.

To generate it locally, run:

python train_defect.py
