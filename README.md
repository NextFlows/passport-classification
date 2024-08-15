---
title: Passport Classification
emoji: 📊
colorFrom: gray
colorTo: gray
sdk: gradio
sdk_version: 4.41.0
app_file: app.py
pinned: false
license: cc-by-2.0
---

## Passport Classification

### Installation
```bash
git clone https://github.com/axcelerateai/passport_classification.git

python3 -m venv .env
source .env/bin/activate

pip install -r requirements.txt

```
### Code integration
```python
import cv2
from model.classify import PassportClassifier

# Load model
classifier = PassportClassifier()

# Run
img = cv2.imread(img_path)
output = classifier.classify(img)

# Print
print(output)
```
The output will either be `passport` or `non-passport`

### Inference on a single image or a directory of images
```
python main.py path/to/image/or/directory/of/images
```

### Compute metrics
To compute recall, precision, f1_score and accuracy, run:
```bash
python main.py path/to/images/folder --mode test
```
Note that the images folder passed above should contain two sub-folders named `passport` and `non-passport`

### Normalize Image Rotation
```
python image_rotation.py path/to/image/or/directory/of/images --save_dir output
```
