# Depth Anything V2 Model Files

This directory contains the ONNX model file for Depth Anything V2 depth prediction.

## Required File

**File:** `depth-anything-v2-small-518.onnx`

**Size:** ~97 MB (ViT-Small backbone)

## Important Notes for GitHub Pages Hosting

- **Included in Repository:** The model file IS included in this repository for user convenience.
- **Browser Caching:** Once downloaded by a user's browser, the 97MB file is cached. Subsequent page loads will use the cached version (no re-download).
- **In-Memory Caching:** The model is loaded once into memory and reused for all depth predictions. Changing depth parameters (gamma, invert, etc.) does NOT reload the model.
- **Preloading:** The app automatically starts loading the model 1 second after page load, so it's ready when users add their first image.

## How to Obtain the Model (For Reference)

The model is already included in this repository. These instructions are provided for reference in case you need to update or obtain the model separately.

### Option 1: Download Pre-Converted ONNX Model (Recommended)

Download the ONNX model from Hugging Face ONNX Community:

1. Visit: https://huggingface.co/onnx-community/depth-anything-v2-small
2. Navigate to the `onnx/` directory
3. Download `model.onnx` from: https://huggingface.co/onnx-community/depth-anything-v2-small/blob/main/onnx/model.onnx
4. Rename the downloaded file to `depth-anything-v2-small-518.onnx`
5. Place it in this directory

**Direct Download Link:**
```bash
curl -L "https://huggingface.co/onnx-community/depth-anything-v2-small/resolve/main/onnx/model.onnx" -o depth-anything-v2-small-518.onnx
```

**Attribution:** ONNX conversion provided by Hugging Face ONNX Community

### Option 2: Convert PyTorch Model to ONNX

If pre-converted ONNX models aren't available:

1. Clone the Depth Anything V2 repository:
   ```bash
   git clone https://github.com/DepthAnything/Depth-Anything-V2.git
   cd Depth-Anything-V2
   ```

2. Install dependencies:
   ```bash
   pip install torch torchvision onnx
   ```

3. Download the Small checkpoint:
   - https://huggingface.co/depth-anything/Depth-Anything-V2-Small

4. Convert to ONNX (518x518 input):
   ```python
   import torch
   from depth_anything_v2.dpt import DepthAnythingV2

   model = DepthAnythingV2(encoder='vits', features=64, out_channels=[48, 96, 192, 384])
   model.load_state_dict(torch.load('depth_anything_v2_vits.pth'))
   model.eval()

   dummy_input = torch.randn(1, 3, 518, 518)
   torch.onnx.export(
       model,
       dummy_input,
       'depth-anything-v2-small-518.onnx',
       input_names=['image'],
       output_names=['depth'],
       dynamic_axes={'image': {0: 'batch'}, 'depth': {0: 'batch'}},
       opset_version=14
   )
   ```

5. Copy the generated ONNX file to this directory

## Model Information

- **Model:** Depth Anything V2 (Small)
- **Input:** RGB image, 518x518, NCHW format, ImageNet normalized
- **Output:** Depth map, 518x518, float32
- **License:** Apache 2.0
- **Repository:** https://github.com/DepthAnything/Depth-Anything-V2
- **Paper:** "Depth Anything V2" (https://arxiv.org/abs/2406.09414)

## Fallback Behavior

If the model file is not present, Apoculus will:
- Display an error in the browser console
- Fall back to grayscale-based depth mapping
- Continue to function with other depth algorithms

## License Compliance

Depth Anything V2 is licensed under Apache License 2.0, which is compatible with this MIT-licensed project. See `LICENSES-THIRD-PARTY.md` in the parent directory for full attribution.
