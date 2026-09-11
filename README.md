# Flower Species Classification: CNN vs MobileNetV2

## About this project

This is my first end-to-end computer vision project. I used the Oxford 17 Flowers dataset to compare two ways of classifying flower images:

- a CNN trained from scratch; and
- MobileNetV2 using transfer learning.

The main question was:

> How does a CNN trained from scratch compare with a pretrained transfer-learning model for multiclass flower-species classification?

## 🌸 Try the Live Flower Classifier

A live version of the trained MobileNetV2 model is available for testing.

**[Open the Flower Classifier](https://flower-species-classification.streamlit.app/)**

Upload a flower image and the application will show the model's predicted flower class, confidence score and top three predictions.

> **Note:** The model was trained to recognise only the 17 flower classes in the Oxford 17 Flowers dataset. If an image outside these classes is uploaded, the model will still choose one of the 17 known classes.

## Dataset

The Oxford 17 Flowers dataset contains 1,360 images from 17 flower classes, with 80 images per class.

I used official Split 1:

- Training: 680 images (40 per class)
- Validation: 340 images (20 per class)
- Test: 340 images (20 per class)

Both models used the same split and 224 × 224 image size so that the comparison was fair.

## What the two models mean

### CNN from scratch

The first CNN started with no previous image knowledge. It had to learn useful visual patterns directly from the 680 training images.

### MobileNetV2 transfer learning

The second model used MobileNetV2 with ImageNet pretrained weights. In simple terms, it already had experience learning useful patterns from many other images. I froze the pretrained base and trained a new classifier for the 17 flower classes.

## Results

| Measure | CNN from scratch | MobileNetV2 |
|---|---:|---:|
| Test accuracy | 52.65% | **96.18%** |
| Test loss | 1.4553 | **0.1848** |
| Macro precision | 54.31% | **96.30%** |
| Macro recall | 52.65% | **96.18%** |
| Macro F1-score | 51.59% | **96.17%** |
| Training time | **70.55 s** | 89.74 s |
| Trainable parameters | 392,785 | **21,777** |

MobileNetV2 improved test accuracy by **43.53 percentage points**. It correctly classified 327 of the 340 test images, compared with 179 correct predictions from the scratch CNN.

## What I learned from the results

The scratch CNN did learn: its accuracy increased and its loss decreased during training. However, its training accuracy itself remained around 50%, showing that it struggled to learn the 17-class problem strongly from the small training set.

MobileNetV2 learned much faster and performed much better. Its validation accuracy stayed around the low-90% range and its final test accuracy was 96.18%. Its later training curves showed a small gap between training and validation performance, suggesting some mild overfitting, but its performance on unseen test images remained very strong.

The class-by-class results showed the same pattern. The scratch CNN performed well on some classes but failed to correctly identify any of the 20 Iris test images. MobileNetV2 achieved an F1-score above 0.90 for every flower class.

## Main conclusion

For this experiment, **MobileNetV2 transfer learning clearly outperformed the CNN trained from scratch**.

The main reason is that the scratch CNN had to learn visual features from only 680 training images, while MobileNetV2 could reuse visual knowledge learned previously from ImageNet. This demonstrates why transfer learning can be especially useful when the new training dataset is relatively small.

## Important limitation

The 96.18% result should not be interpreted as meaning that the model will classify all flower images with 96% accuracy. The result comes from 340 test images in one official Oxford split and covers only the 17 flower classes in this dataset.

Future work could repeat the experiment using the other official Oxford splits or test the model on a separate flower dataset.

## Repository contents

A simple repository can contain:

```text
flower-species-classification/
├── Flower_Species_Classification.ipynb
├── README.md
└── results/
    ├── scratch_accuracy.png
    ├── scratch_loss.png
    ├── mobilenet_accuracy.png
    ├── mobilenet_loss.png
    ├── scratch_confusion_matrix.png
    └── mobilenet_confusion_matrix.png
```

The raw dataset does not need to be uploaded to the repository.

## How to run

1. Open the notebook in Google Colab.
2. Select a GPU runtime if available.
3. Upload `17flowers.tgz` and `datasplits.mat` when prompted.
4. Run the notebook cells from top to bottom.
5. Train the CNN from scratch.
6. Train MobileNetV2.
7. Compare the validation results.
8. Run the final test evaluation after model development is complete.

## Beginner glossary

**CNN** — A type of neural network designed to learn patterns from images.

**Epoch** — One complete pass through the training data.

**Training accuracy** — How often the model is correct on data used to teach it.

**Validation accuracy** — How often the model is correct on separate data used to check its progress during development.

**Test accuracy** — Final performance on data kept aside from model development.

**Loss** — A number representing how wrong the model is. Lower is generally better.

**Precision** — When the model predicts a particular class, how often that prediction is correct.

**Recall** — Of all the real examples of a class, how many the model finds correctly.

**F1-score** — A combined measure of precision and recall.

**Transfer learning** — Reusing knowledge learned by a model on a previous task instead of starting completely from zero.

**Overfitting** — When a model becomes very good at its training examples but does not improve as much on unseen examples.

**Trainable parameter** — A number inside the model that training is allowed to adjust.
