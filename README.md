# Setup:
*Note: run all these from root of the project*

Installation (using python 3.9 virtual env):
```
python3 -m venv venv
```

```
pip install -r requirements.txt
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
```

# Usage

Run the model using
```
python train.py --config <config>
```

Where `config` specifies the configuration file. The script looks for YAML files in the `/configs/` directory.
Configuration files come in the following format:


| Option | Type | Notes|
|------|------|------|
|random_transforms|boolean|Apply random transformations to the training set|
| cosine_annealing|boolean|Use cosine annealing learning rate scheduler|
|class_imbalance_fix|boolean|Use class imbalance fix|
|epochs|integer|Number of training epochs|
|batch_size|integer|SGD batch size|
|freeze_encoder|boolean|Freeze the encoding layers of the network|
|model_type|string|Specify the model type: "unet", "resnet", "fcn", or "new_arch"|
|model_identifier|string|Identifier for save location of model-specific run data|


Please have a look at the report of all of our results.
[CSE_251B_Project_3.pdf](https://github.com/Anirudhaagrawal/PASCAL-VOC-2007-Classification/files/10836679/CSE_251B_Project_3.pdf)
