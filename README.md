
## 🚀 Getting Started
<span id="getting-started"></span>

### Environment Requirement 🌍
<span id="environment-requirement"></span>

- p2p_requirements.txt: for models in `run_editing_p2p.py`
- masactrl_requirements.txt: for models in `run_editing_masactrl.py`
- pnp_requirements.txt: for models in `run_editing_pnp.py`
- videop2p_requirements.txt: for models in `run_videop2p.py`

```shell
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r environment/p2p_requirements.txt

pip install -r environment/masactrl_requirements.txt

pip install -r environment/pnp_requirements.txt

pip install -r environment/videop2p_requirements.txt
```

### Benchmark Download ⬇️
<span id="benchmark-download"></span>

You can download the benchmark PIE-Bench (Prompt-driven Image Editing Benchmark) [here](https://github.com/cure-lab/PnPInversion).

We release our video editing dataset [here](https://drive.google.com/file/d/1D9uX2cY4_l2jmkMKy7f2o5HU790Rc9mH/view?usp=sharing).

## 🏃🏼 Running Scripts
<span id="running-scripts"></span>

### Inference 📜
<span id="inference"></span>

For example, if you want to run Prompt-to-Prompt:

```
python run_editing_p2p.py --image_path scripts/car.jpg --original_prompt "a colorful car is parked on the street" --editing_prompt "a colorful motorcycle is parked on the street" --blended_word "car motorcycle"
```

if you want to run MasaCtrl:

```
python run_editing_masactrl.py --image_path scripts/walking_woman.jpg --original_prompt "a woman in a hat and dress walking down a path at sunset" --editing_prompt "a woman in a hat and dress running down a path at sunset"
```

if you want to run PNP:

```
python run_editing_pnp.py --image_path scripts/oil_painting.jpg --original_prompt "white flowers on a tree branch with blue sky background" --editing_prompt "an oil painting ofwhite flowers on a tree branch with blue sky background"
```

if you want to run Video-P2P:

To download the pre-trained model, please refer to [diffusers](https://github.com/huggingface/diffusers).

Please download [sd1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5) and fill pretrained_model_path at config.

```
python run_tuning.py  --config="configs/rabbit-jump-tune.yaml"
python run_videop2p.py --config="configs/rabbit-jump-p2p.yaml" --fast
```



