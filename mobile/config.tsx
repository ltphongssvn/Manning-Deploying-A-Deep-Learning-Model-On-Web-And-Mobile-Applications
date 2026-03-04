// mobile/config.tsx
// App-wide configuration constants — no hardcoded values in components

export const APP_CONFIG = {
  title: "Food Classifier",
  imageSize: 224,
  numPredictions: 3,
  probabilityPrecision: 2,
  aboutText: `
# Food Classifier Mobile App

This app classifies food images using a MobileNetV2 deep learning model
trained on the Food-101 dataset (3 classes).

## Supported Classes
- Apple Pie
- Caesar Salad
- Falafel

## How It Works
1. Select an image from your gallery or take a photo
2. The TF.js model runs inference entirely on-device
3. Predictions display with confidence scores

## Model Details
- Architecture: MobileNetV2 (transfer learning)
- Input: 224×224 RGB images
- Quantization: float16 (6.8MB)
- Framework: TensorFlow.js
`,
};
