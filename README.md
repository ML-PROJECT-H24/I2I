# High resolution image synthesis with translation

## Project structure overview

- `data\`: Contains datasets
- `logs\`: Contains all logs for experiments
- `requirements.txt`: PIP requirements for this project
- `latent_dataset_creator.ipynb`: Notebook for creating latent datasets
- `gaussian_diffusion.py`: Diffusion algorithms from OpenAI
- `mdtv2.py`: Official MDTv2 architecture
- `latent_dataset.py`: Custom pytorch dataset object for latent datasets
- `img_train.py`: Script for training a model
- `img_sample.py`: Script for generating images from a trained model
- `img_translate.py`: Script for doing image to image translation from a trained model

## Creating a latent dataset

1. Find a dataset to you want to transform
2. Run the latent dataset creator with your dataset

Since the selfie2anime dataset is small enough, we provide the latent dataset `data\latent_selfie2anime`. So there is no need to create a latent dataset to test this repo.

## Training a model

**By default our repo will train the smallest model, however it is still quite large, so make sure you have a good enough GPU**

Example:
```bash
python img_train.py --data-dir="data\latent_selfie2anime"
```

The most recent model will be saved in the logs directly under the directly with the start date. Images generated from the same noise will also be output every epoch, so you can observe the process!

A pretrained model is to large to include in the project so you need to train your own! On a decent GPU the smaller model trained on selfie2anime should not take too long to train, perhaps a day before you see acceptable results.

## Generating images with an existing model

Example:
```bash
python img_sample.py --model-path="data\model.pt" --num-samples=8 --sample-steps=250
```

**sample-steps**: Sample steps are the amount of steps used for the generation, they go from 0 to 1000. Higher leads to better quality but more processing time.

Results will be output in the logs directory.

## Image to image translation

Example:
```bash
python img_translate.py --model-path="model.pt" --src-img-path="my_img.png" --strength=0.4 --cond=0
```

**strength**: Strength is the amount of noise to add to the image from 0-1 where higher means more noise.
**cond**: Is the class/domain identifier, in the case of selfie2anime 0=anime, 1=human

To do image to image translate within the same domain, meaning your making variations of the same image, simply use the same domain as the cond. If your doing translation between domains, use the opposite cond.


