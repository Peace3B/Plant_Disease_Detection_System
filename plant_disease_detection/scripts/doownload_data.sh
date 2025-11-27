#!/bin/bash

echo "Downloading PlantVillage dataset..."

# Download from Kaggle
kaggle datasets download -d https://www.kaggle.com/datasets/arjuntejaswi/plant-village/data -p data/raw/

# Unzip
cd data/raw/
unzip -q PlantVillage-dataset.zip
cd ../..

# Verify download
echo "Dataset structure:"
ls -lh data/raw/PlantVillage\Diseases\ Dataset\(Augmented\)/

echo "Download complete!"