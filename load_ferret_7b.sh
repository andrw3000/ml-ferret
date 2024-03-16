## Download and install ml-ferret from Apple

# Create conda environment and install dependencies
conda create -n ferret python=3.10 -y
conda activate ferret
pip install --upgrade pip
pip install -e .
pip install pycocotools
pip install protobuf==3.20.0
pip install safetensors

# Download Vicuna weights
mkdir -p ./model
mkdir -p ./model/offload  # For offloading shards during runtime
git lfs install
git clone https://huggingface.co/lmsys/vicuna-7b-v1.3 model/vicuna-7b-v1.3

# Download Ferret delta weights
mkdir -p ./delta
curl -o ./delta/ferret-7b-delta.zip https://docs-assets.developer.apple.com/ml-research/models/ferret/ferret-7b/ferret-7b-delta.zip
unzip ./delta/ferret-7b-delta.zip -d ./delta

# Transform Vicuna model to Ferret
python -m ferret.model.apply_delta \
--base ./model/vicuna-7b-v1.3 \
--target ./model/ferret-7b-v1-3 \
--delta ./delta/ferret-7b-delta
