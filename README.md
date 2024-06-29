## Pytorch-Image-Classification

A simple demo of **image classification** using PyTorch. Here, we use a **custom dataset** containing **43956 images** belonging to **11 classes** for training (and validation). Also, we compare three different approaches for training viz. **training from scratch, finetuning the convnet, and convnet as a feature extractor**, with the help of **pretrained** PyTorch models. The models used include: **VGG11, Resnet18, and MobilenetV2**.


### Installation

Make sure to install the required packages listed in the `requirements.txt` file:

```python
pip install -r requirements.txt
```

### How to Run
Make sure to have your data under the example_dataset folder structure.

Run the following scripts for training and/or testing:

#### Training
The train.py script trains the model. You can specify the training mode as finetune, transfer, or scratch.

```python
python train.py --mode=finetune
```

#### Testing
The test.py script will run predictions on sample images from a specified directory. It will save individual prediction images to an output directory and create a JSON file with the predictions.

```python
python test.py <image_directory> <output_directory>
```




