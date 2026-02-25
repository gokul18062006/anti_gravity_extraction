# Dataset Instructions

## Download the Tamil Handwritten Character Recognition Dataset

1. Go to [Kaggle - Tamil Handwritten Character Recognition](https://www.kaggle.com/datasets/sudalairajkumar/tamil-handwritten-character-recognition)
2. Download and extract the dataset
3. Place the training images in `data/train/<class_label>/` folders
4. Place the test images in `data/test/<class_label>/` folders

## Expected Structure

```
data/
├── train/
│   ├── class_0/
│   │   ├── img_001.bmp
│   │   ├── img_002.bmp
│   │   └── ...
│   ├── class_1/
│   └── ...
├── test/
│   ├── class_0/
│   ├── class_1/
│   └── ...
└── README.md
```

## Notes
- The dataset contains BMP images of handwritten Tamil characters
- Each class folder corresponds to a Tamil character
- The label mapping will be generated during training
