# Active Learning Labelling

Labelling a 10,000-image digit dataset (28x28 grayscale, 10 classes) with as little manual work as possible. A small hand-labelled seed set is augmented and used to train an RBF SVM, which is then improved over several rounds by labelling its most uncertain images by hand and adding its most confident predictions as pseudo-labels.

Made for ELC 4028 (Artificial Neural Networks), Cairo University. This is Pipeline 2 of the assignment.

## How it works

1. Load all images as 784-dim pixel vectors in [0, 1].
2. Pick 300 random images and label them by hand (weight 100).
3. Augment each one 6 times (rotation +-5 deg, Gaussian noise, 3 shifts of 2 px) at weight 1.
4. Train an RBF SVM (one-vs-one, C = 10) with sample weights, then predict all 10,000 images.
5. Take the 20 images with the smallest margin (top score minus second score), label them by hand, and add them with weight 100.
6. For each class, add up to 50 of the most confident predictions above the 75th percentile margin as pseudo-labels (weight 1).
7. Retrain, check accuracy, and repeat from step 5 until accuracy reaches 99% or stops improving (gain under 0.1% twice in a row).

## Results

| | |
|---|---|
| Features | raw pixels (784) |
| Starting accuracy | 89.2% |
| Final accuracy | 97.43% |
| Iterations | 22 |
| Images labelled by hand | 740 (300 seed + 22 x 20 boundary) |
| Pseudo-labels added | 3,528 |
| Manual time | 7,400 s (about 2.06 h) |
| Fully manual baseline | 100,000 s (about 27.8 h) |

The 99% target was not reached. Manual time is about 93% lower than labelling everything by hand.

Accuracy per iteration:

| Iter | Acc (%) | Iter | Acc (%) |
|---:|---:|---:|---:|
| 1 | 89.22 | 12 | 95.79 |
| 2 | 90.20 | 13 | 95.98 |
| 3 | 91.51 | 14 | 96.18 |
| 4 | 92.54 | 15 | 96.52 |
| 5 | 93.19 | 16 | 96.71 |
| 6 | 93.42 | 17 | 96.86 |
| 7 | 93.78 | 18 | 97.00 |
| 8 | 94.29 | 19 | 97.12 |
| 9 | 94.62 | 20 | 97.28 |
| 10 | 95.03 | 21 | 97.36 |
| 11 | 95.43 | 22 | 97.43 |

These numbers come from a run that used a single global 75th percentile threshold. The current `run_pipeline_2.m` computes the threshold per class and caps at 20 iterations, so a new run may differ.

## Usage

You need MATLAB with the Statistics and Machine Learning Toolbox and the Image Processing Toolbox.

Put these next to `run_pipeline_2.m`:

- `Indian_Digits_Train/` with `1.bmp` to `10000.bmp`
- `check_accuracy.p` (provided with the assignment)
- `seed_labels.csv` with columns `Filename` and `Label`

Then run:

```matlab
run_pipeline_2
```

Each iteration shows the 20 uncertain images one by one. Type the digit you see in the command window. Results are saved to `pipeline2_final.mat`.
