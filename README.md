<img alt="lucyra-logo" src="./assets/images/logo.png" style="margin-left:auto; margin-right:auto; display:block; width:200px;"/>

# Apoculus - 3D Image Generator

Create mesmerizing autostereograms (3D images) from text and images. View hidden 3D content by crossing or diverging your eyes - no special glasses needed!

**Live App:** [nqrlabs.com/Apoculus/](https://nqrlabs.com/Apoculus/)

## What is an Autostereogram?

An autostereogram is a single-image stereogram (SIS) that creates the illusion of 3D depth from a 2D pattern. Popular in the 1990s, these images encode depth information in a repeating random-dot pattern. When viewed correctly, hidden 3D shapes pop out of the flat image.

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
- **Automatic Processing** - Color images are converted to grayscale, dithered, and made transparent
- **Floyd-Steinberg Dithering** - High-quality black & white conversion
- **Invert Alpha** - Choose whether white or black becomes transparent
- **Size Control** - Scale images to fit your composition (auto-reprocesses on resize, 4x multiplier for better visibility)
- **Depth Positioning** - Place images at any depth level
- **Alpha Channel Detection** - Images with existing transparency are preserved as-is

### 3D Depth Control
- **Depth Levels** - Position each layer from far background (0) to close foreground (100)
- **3D Effect Strength** - Adjust the overall intensity of the 3D effect
- **Layer Ordering** - Closer layers automatically occlude farther ones

### Pattern Customization
- **Pattern Density** - Control the number of dots in the random pattern (2-8 dots per tile)
- **Pattern Width** - Adjust the repeating tile size (48-96px)
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

1. **Image Processing** (for uploaded images without alpha)
   - Convert RGB to grayscale using luminance formula (0.299R + 0.587G + 0.114B)
   - Apply Floyd-Steinberg dithering for high-quality binary conversion
   - Convert white (or black if inverted) to transparent alpha channel
   - Preserve original image for re-processing when scaled

2. **Depth Map Generation**
   - Each layer is rendered to a grayscale depth map
   - Brighter pixels = closer to viewer
   - Layers are composited by depth priority (nearer layers occlude farther ones)
   - Soft Gaussian blur reduces edge artifacts

3. **Random Pattern Strip**
   - A narrow vertical strip of random dots is generated using hash-based randomization
   - Hash function ensures uniform distribution across the entire image
   - Strip width (typically 48-96px) determines the viewing comfort
   - Pattern density affects detail resolution
   - Color schemes apply different color palettes to the random pattern

4. **Scanline Processing**
   - For each horizontal scanline:
     - Pixels are linked based on depth disparity
     - Disparity = depth × scaling factor
     - Occlusion handling: nearer depths take priority

5. **Pixel Assignment**
   - Linked pixels receive the same color from the pattern strip
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
- **Font Upload** - Uses FontFace API (widely supported)
- **No Dependencies** - Pure vanilla JavaScript, no frameworks

## Best Practices

### For Best Results

**Text:**
- Use bold, thick fonts (Arial Black, Impact)
- Keep text short and clear
- Moderate depth values (10-30) are easier to view comfortably
- Higher depth values create more dramatic pop-out effect but may be harder to focus
- Avoid thin, delicate fonts

**Images:**
- High-contrast images work best
- Silhouettes and simple shapes are easier to see
- Color images are automatically converted using dithering
- Use "Invert Alpha" if your subject should be transparent instead of the background
- Images with existing alpha transparency are preserved as-is
- Simple logos, icons, and text work great
- Avoid very detailed or noisy photographs (dithering may lose detail)

**Composition:**
- Start with one or two layers
- Use moderate depth values (10-30) for foreground elements
- Background depth stays at 0 automatically
- Space out depth values between layers for clear separation
- Center important elements

**Pattern Settings:**
- Default settings (Density: 4, Width: 72px) work for most cases
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
