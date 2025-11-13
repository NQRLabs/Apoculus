<img alt="lucyra-logo" src="./assets/images/logo.png" style="margin-left:auto; margin-right:auto; display:block; width:200px;"/>

# Apoculus - 3D Image Generator

Create mesmerizing autostereograms (3D images) from text and images. View hidden 3D content by crossing or diverging your eyes - no special glasses needed!

**Live App:** [nqrlabs.com/Apoculus/](https://nqrlabs.com/Apoculus/)

## What is an Autostereogram?

An autostereogram is a single-image stereogram (SIS) that creates the illusion of 3D depth from a 2D pattern. Popular in the 1990s, these images encode depth information in a repeating random-dot pattern. When viewed correctly, hidden 3D shapes pop out of the flat image.

## Setup (Optional AI Features)

Apoculus works out-of-the-box with direct grayscale and shadow-aware depth algorithms. For AI-powered depth estimation using **Depth Anything V2**:

1. Download the ONNX model file (~97MB) following instructions in `models/README.md`
2. Place the model in the `Apoculus/models/` directory
3. The AI algorithm will then be available in the Depth Algorithm dropdown

**Note:** The AI algorithm requires WebGPU (Chrome/Edge 113+) or falls back to WebAssembly. Direct and Shadow-Aware algorithms work on all browsers without any model download.

## Features

### Layer Management
- **Add Text Layers** - Type any text and see it float in 3D space
- **Add Image Layers** - Upload images to embed at different depth levels
- **Multiple Layers** - Combine multiple text and image elements at various depths
- **Full Control** - Adjust size, rotation, position, and depth for each layer

### Text Options
- **Font Selection** - Choose from bold, high-visibility fonts
- **Custom Fonts** - Upload your own TTF, OTF, WOFF, or WOFF2 fonts
- **Adjustable Size** - Scale text from small to massive (20-400px)
- **Text Rotation** - Rotate text to any angle (-180° to +180°)

### Image Support
- **Upload Images** - PNG, JPG, GIF, and other web formats
- **Transparency Support** - Transparent pixels always map to background depth, regardless of other settings
- **Grayscale-to-Depth Mapping** - Image brightness directly controls depth (white=near, black=far)
- **Auto-Level Correction** - Automatically stretches contrast for washed-out scans
- **Depth Algorithms** - Choose between AI-powered, direct mapping, or shadow-aware processing:
  - **Depth Anything V2 (AI)** - State-of-the-art AI depth prediction (requires model download, see `models/README.md`)
  - **Direct Grayscale Mapping** - Straightforward brightness-to-depth conversion
  - **Shadow-Aware (Retinex-inspired)** - Separates lighting from surface brightness to prevent shadows from creating false depth holes
- **Depth Gamma Control** - Adjust midtone depth distribution (0.6-1.4) for flatter or steeper curves
- **Invert Depth** - Reverse depth mapping (black=near, white=far); transparent pixels always stay background
- **Size Control** - Scale images to fit your composition (4x multiplier for better visibility)
- **Depth Positioning** - Place images at any depth level

### 3D Depth Control
- **Depth Levels** - Position each layer from far background (0) to close foreground (100)
- **3D Effect Strength** - Adjust the overall intensity of the 3D effect
- **Layer Ordering** - Closer layers automatically occlude farther ones

### Pattern Customization
- **Pattern Density** - Control the number of dots in the random pattern (2-8 dots per tile)
- **Pattern Width** - Adjust the repeating tile size (47-97px, prime numbers recommended for minimal artifacts)
- **3D Effect Strength** - Adjust how dramatic the depth appears (0.3-1.5x)
- **Output Size** - Generate images from 800px to 2000px wide
- **Color Scheme** - Choose from multiple color options:
  - **Grayscale** - Classic black and white random dots
  - **Random Colors** - Full RGB random colors for each dot
  - **Cyan Dots** - Cyan/aqua colored pattern
  - **Magenta Dots** - Magenta/pink colored pattern
  - **Green Dots** - Green colored pattern
  - **Amber Dots** - Orange/amber colored pattern
  - **Rainbow** - Multi-colored rainbow dots
  - **Neon Mix** - Bright neon colors (magenta, cyan, yellow, green, hot pink)
  - **Fire** - Reds, oranges, and yellows
  - **Ice** - Cool blues and cyans
  - **Forest** - Earth greens and browns
  - **Sunset** - Purples, oranges, and pinks
  - **Ocean** - Deep blues to bright cyan
  - **Matrix Green** - Various shades of green with slight blue tint
  - **Vaporwave** - Pink and cyan aesthetic
  - **Synthwave** - Purple, pink, and cyan retro aesthetic

### Export
- **Download PNG** - Save your autostereogram as a high-quality PNG image
- **Real-time Preview** - See changes instantly as you adjust settings
- **Regenerate** - Create new random patterns while keeping your layers

## How to Use

### Basic Workflow

1. **Add a Layer**
   - Click "+ Text" to add text, or "+ Image" to upload an image
   - Your first layer will appear in the preview

2. **Edit Layer Content**
   - For text: Type in the text box
   - For images: Upload a new image file
   - Select a layer from the list to edit it

3. **Adjust Position & Depth**
   - **Size** - How large the element appears
   - **Depth Level** - How close it floats (higher = closer to viewer)
   - **Rotation** - Angle in degrees
   - **Horizontal/Vertical Position** - Where it appears in the frame

4. **Customize the Pattern**
   - **Pattern Density** - More dots = finer detail
   - **Pattern Width** - Larger tiles = easier to view
   - **3D Effect Strength** - How dramatic the depth appears

5. **Download**
   - Click "⬇ Download Image" to save your creation

### How to View the 3D Effect

There are two viewing methods:

**Cross-Eye Method (Easier for Beginners):**
1. Hold the image at arm's length
2. Cross your eyes slightly until you see three images
3. Focus on the middle image - the 3D shape will appear
4. Relax and enjoy the depth

**Parallel Method (Wall-Eye):**
1. Hold the image close to your face
2. Slowly move it away while looking "through" it
3. Keep your eyes parallel (like looking at something far away)
4. The 3D image will snap into focus

**Tips:**
- Start with high-contrast, bold text for easier viewing
- Ensure good lighting
- Take breaks if your eyes get tired
- Some people find one method easier than the other - try both!

## Technical Details

### Algorithm Overview

Apoculus uses a depth-map-based autostereogram generation algorithm:

1. **Image Processing**
   - Convert RGB to grayscale using Rec. 709 coefficients (0.2126R + 0.7152G + 0.0722B)
   - Preserve alpha channel from original image (transparency)
   - Apply auto-level correction to stretch histogram for optimal contrast (ignoring transparent pixels)
   - Three depth mapping algorithms available:
     - **Depth Anything V2 (AI)**: Deep learning-based monocular depth estimation
     - **Direct Mapping**: Grayscale value directly determines depth
     - **Shadow-Aware (Retinex-inspired)**: Separates illumination from reflectance

2. **Depth Anything V2 (AI) Algorithm**
   - State-of-the-art monocular depth estimation using deep learning
   - Based on Vision Transformer (ViT-Small) architecture trained on diverse datasets
   - Runs entirely in-browser using ONNX Runtime Web (WebGPU/WASM)
   - Processes images at 518×518 resolution with letterboxing for aspect preservation
   - Applies ImageNet normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
   - Outputs are normalized using 2nd-98th percentile for robustness to outliers
   - Inverted so white=near, black=far (consistent with other algorithms)
   - Best for photos, complex scenes, and realistic depth from natural images
   - Optional model download (~97MB), see `models/README.md` for setup
   - ONNX model provided by Hugging Face ONNX Community: https://huggingface.co/onnx-community/depth-anything-v2-small
   - Reference: Yang et al. (2024). "Depth Anything V2" https://arxiv.org/abs/2406.09414
   - License: Apache 2.0 (see `LICENSES-THIRD-PARTY.md`)

3. **Shadow-Aware Depth Algorithm**
   - Based on classic Retinex theory (Land & McCann, 1971)
   - Estimates illumination field via large-scale Gaussian blur
   - Divides image by illumination to extract reflectance (albedo)
   - Maps albedo to depth, preventing shadows from creating false depth holes
   - Uses only unpatented, fundamental signal processing techniques
   - Reference: Land, E. H., & McCann, J. J. (1971). "Lightness and retinex theory." *Journal of the Optical Society of America, 61*(1), 1-11.

4. **Depth Map Generation**
   - Each layer is rendered to a grayscale depth map
   - Grayscale values mapped to depth via configurable LUT (with gamma correction)
   - Brighter pixels = closer to viewer (or inverted if "Invert Depth" enabled)
   - **Transparent pixels always map to depth 0 (background)**, regardless of invert setting
   - Layers are composited by depth priority (nearer layers occlude farther ones)
   - Soft blur reduces edge artifacts

5. **Random Pattern Strip**
   - Blue-noise or hash-based random dot pattern
   - Prime-number strip widths (47-97px) minimize visible stratification artifacts
   - Pattern density affects detail resolution
   - Color schemes apply different color palettes to the random pattern

6. **Scanline Processing with Union-Find**
   - For each horizontal scanline:
     - Pixels are linked based on depth disparity
     - Disparity = depth × scaling factor
     - Union-find algorithm groups linked pixels
     - Occlusion handling: nearer depths take priority

7. **Pixel Assignment**
   - Each pixel group (union-find set) receives a single color from the pattern strip
   - Phase offset computed from group center for visual symmetry
   - Row-wise phase shifts break horizontal stratification
   - Creates the repeating pattern with depth-based offsets
   - Final image encodes 3D information in 2D pattern

### Performance Optimizations

- **Typed Arrays** - Efficient pixel data processing
- **Incremental Rendering** - Updates only when layers change
- **Canvas Reuse** - Minimal memory allocation
- **Real-time Generation** - Fast enough for interactive editing on mobile devices

### Browser Compatibility

- **Modern Browsers** - Chrome, Firefox, Safari, Edge (latest versions)
- **Mobile Support** - Works on iOS Safari and Android Chrome
- **WebGPU Support** - Depth Anything V2 requires WebGPU (Chrome/Edge 113+) or falls back to WASM
- **Font Upload** - Uses FontFace API (widely supported)
- **No Dependencies** - Pure vanilla JavaScript, no frameworks (ONNX Runtime loaded via CDN for AI features)

## Best Practices

### For Best Results

**Text:**
- Use bold, thick fonts (Arial Black, Impact)
- Keep text short and clear
- Moderate depth values (10-30) are easier to view comfortably
- Higher depth values create more dramatic pop-out effect but may be harder to focus
- Avoid thin, delicate fonts

**Images:**
- Grayscale images with good contrast work best
- Color images are automatically converted to grayscale with auto-leveling
- **PNG images with transparency** - Transparent backgrounds automatically stay at background depth
- **Algorithm Selection:**
  - Use **Depth Anything V2 (AI)** for photos, portraits, landscapes, and complex natural scenes
  - Use **Direct Mapping** for simple, evenly-lit images, logos, and graphics
  - Use **Shadow-Aware** for images with strong shadows or uneven lighting
- Adjust **Depth Gamma** (0.6-1.4) to control how midtones map to depth
- Use "Invert Depth" to reverse depth mapping (useful for inverted subjects like white-on-black logos)
- Transparent pixels always remain at background depth, even when inverted
- Photographs, logos, icons, and text all work well
- Auto-level correction helps with washed-out or low-contrast images
- Simple shapes are easier to see in 3D than complex patterns

**Composition:**
- Start with one or two layers
- Use moderate depth values (10-30) for foreground elements
- Background depth stays at 0 automatically
- Space out depth values between layers for clear separation
- Center important elements

**Pattern Settings:**
- Default settings (Density: 4, Width: 73px) work for most cases
- Use prime-number widths (53, 59, 61, 67, 71, 73, 79, 83, 89, 97) to minimize visible artifacts
- Larger pattern width = easier to view but less detail
- Higher density = more detail but harder to view
- 3D Effect Strength of 0.8 is a good starting point

## Limitations

- **Viewing Skill Required** - Not everyone can see autostereograms (about 5-10% of people cannot)
- **Eye Strain** - Extended viewing can cause eye fatigue
- **Depth Range** - Limited to 2D layers at different depths (not true volumetric 3D)
- **Pattern Artifacts** - Very sharp edges may create "sparkle" or noise
- **File Size** - High-resolution outputs can be several MB

## Use Cases

- **Artistic Images** - Create unique 3D art
- **Hidden Messages** - Embed secret text visible only to those who know how to view it
- **Puzzle Games** - Hide clues in autostereograms
- **Educational** - Demonstrate stereoscopic vision principles
- **Nostalgic Fun** - Recreate the autostereogram experience from the 90s

## Privacy

Apoculus runs entirely in your browser. No data is uploaded to any server:
- ✅ All images and text stay on your device
- ✅ No tracking or analytics
- ✅ No external dependencies or CDNs
- ✅ Works offline after initial load

## License

MIT License - See [LICENSE](LICENSE) file for details.

Created by **NQR** for the NQR Labs ARG toolkit.

## Support & Feedback

- **Issues & Suggestions:** https://github.com/NQRLabs/nqrlabs.github.io/issues
- **Website:** https://nqrlabs.com
- **Discord:** https://discord.gg/HT9YE8rvuN

---

© 2025 NQR | Part of the [NQR Labs](https://nqrlabs.com) ARG Toolkit
