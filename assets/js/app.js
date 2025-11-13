/**
 * Apoculus - 3D Autostereogram Generator
 *
 * Architecture:
 * - State-based layer management (text, image, video)
 * - Three depth algorithms: AI (Depth Anything V2), Direct mapping, Shadow-aware (Retinex)
 * - Scanline autostereogram generation using union-find algorithm
 * - Video support: frame-by-frame processing with consistent normalization
 *
 * Main flow: Layers → Depth Maps → Pattern Strip → Scanline Processing → Autostereogram
 */

// ============================================================================
// STATE MANAGEMENT
// ============================================================================

    // App State
    const state = {
      layers: [],
      activeLayerId: null,
      customFonts: new Map(),
      settings: {
        patternDensity: 4,
        stripWidth: 73,
        depthScale: 0.8,
        outputWidth: 1200,
        outputHeight: 800,
        colorScheme: 'grayscale'
      }
    };

    let nextLayerId = 1;

    // Cache for AI depth predictions (avoid re-running model on parameter changes)
    const depthCache = new Map(); // layerId -> { rawDepth: ImageData, algorithm: string }

// ============================================================================
// DEPTH ANYTHING V2 - AI DEPTH PREDICTION
// ============================================================================

    // --- Depth Anything V2 (AI depth prediction) ---
    // Based on "Depth Anything V2" by DepthAnything team
    // Model: https://github.com/DepthAnything/Depth-Anything-V2
    // ONNX Model: https://huggingface.co/onnx-community/depth-anything-v2-small
    // ONNX conversion by Hugging Face ONNX Community
    // License: Apache 2.0 (see LICENSES-THIRD-PARTY.md)
    let depthModel = null;
    let depthModelLoading = false;

    // ImageNet normalization constants
    const IMAGENET_MEAN = [0.485, 0.456, 0.406];
    const IMAGENET_STD = [0.229, 0.224, 0.225];

    // Load the Depth Anything V2 model
    async function loadDepthModel() {
      if (depthModel) return depthModel;
      if (depthModelLoading) {
        // Wait for ongoing load
        while (depthModelLoading) {
          await new Promise(resolve => setTimeout(resolve, 100));
        }
        return depthModel;
      }

      depthModelLoading = true;
      try {
        // Configure ONNX Runtime to use WebGPU with WASM fallback
        const session = await ort.InferenceSession.create(
          'models/depth-anything-v2-small-518.onnx',
          { executionProviders: ['webgpu', 'wasm'] }
        );
        depthModel = session;
        console.log('Depth Anything V2 model loaded successfully');
        return depthModel;
      } catch (error) {
        console.error('Failed to load Depth Anything V2 model:', error);
        throw error;
      } finally {
        depthModelLoading = false;
      }
    }

    // Resize and letterbox image to 518x518 preserving aspect ratio
    function drawToSquare518(img) {
      const targetSize = 518;
      const canvas = document.createElement('canvas');
      canvas.width = targetSize;
      canvas.height = targetSize;
      const ctx = canvas.getContext('2d');

      // Fill with black (will be letterbox bars)
      ctx.fillStyle = '#000000';
      ctx.fillRect(0, 0, targetSize, targetSize);

      // Calculate scaling to fit within 518x518 while preserving aspect ratio
      const scale = Math.min(targetSize / img.width, targetSize / img.height);
      const scaledWidth = Math.round(img.width * scale);
      const scaledHeight = Math.round(img.height * scale);

      // Center the image
      const x = Math.floor((targetSize - scaledWidth) / 2);
      const y = Math.floor((targetSize - scaledHeight) / 2);

      ctx.drawImage(img, x, y, scaledWidth, scaledHeight);

      return { canvas, scaledWidth, scaledHeight, offsetX: x, offsetY: y };
    }

    // Build NCHW float32 tensor with ImageNet normalization (optimized)
    function makeInputTensor(canvas) {
      const ctx = canvas.getContext('2d');
      const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
      const { data, width, height } = imageData;

      // Create NCHW tensor (1, 3, 518, 518)
      const tensorData = new Float32Array(1 * 3 * height * width);
      const pixelCount = height * width;

      // Pre-calculate normalization constants
      const r_scale = 1.0 / (255.0 * IMAGENET_STD[0]);
      const g_scale = 1.0 / (255.0 * IMAGENET_STD[1]);
      const b_scale = 1.0 / (255.0 * IMAGENET_STD[2]);
      const r_mean_norm = IMAGENET_MEAN[0] / IMAGENET_STD[0];
      const g_mean_norm = IMAGENET_MEAN[1] / IMAGENET_STD[1];
      const b_mean_norm = IMAGENET_MEAN[2] / IMAGENET_STD[2];

      // Convert RGBA to normalized RGB channels (NCHW format)
      // Optimized: single pass through pixel data, vectorized operations
      const rOffset = 0;
      const gOffset = pixelCount;
      const bOffset = pixelCount * 2;

      for (let i = 0; i < pixelCount; i++) {
        const pixelOffset = i * 4;
        tensorData[rOffset + i] = data[pixelOffset] * r_scale - r_mean_norm;
        tensorData[gOffset + i] = data[pixelOffset + 1] * g_scale - g_mean_norm;
        tensorData[bOffset + i] = data[pixelOffset + 2] * b_scale - b_mean_norm;
      }

      return new ort.Tensor('float32', tensorData, [1, 3, height, width]);
    }

    // Run the model and return Float32 depth array (HxW)
    async function predictDepthFloat(img) {
      const model = await loadDepthModel();

      // Prepare input
      const { canvas, scaledWidth, scaledHeight, offsetX, offsetY } = drawToSquare518(img);
      const inputTensor = makeInputTensor(canvas);

      // Run inference
      const feeds = { pixel_values: inputTensor };
      const results = await model.run(feeds);

      // Get depth output (should be [1, 518, 518])
      const depthTensor = results[Object.keys(results)[0]];
      const depthData = depthTensor.data;
      const tensorHeight = depthTensor.dims[1];
      const tensorWidth = depthTensor.dims[2];

      // Extract the region corresponding to the actual image (remove letterbox)
      // Optimized: use row-based copying instead of pixel-by-pixel
      const depthArray = new Float32Array(scaledHeight * scaledWidth);

      for (let y = 0; y < scaledHeight; y++) {
        const srcRowStart = (offsetY + y) * tensorWidth + offsetX;
        const dstRowStart = y * scaledWidth;
        // Copy entire row at once (much faster than pixel-by-pixel)
        for (let x = 0; x < scaledWidth; x++) {
          depthArray[dstRowStart + x] = depthData[srcRowStart + x];
        }
      }

      return {
        data: depthArray,
        width: scaledWidth,
        height: scaledHeight
      };
    }

    // Fast histogram-based percentile calculation (O(n) instead of O(n log n) sort)
    function findPercentilesHistogram(data, p2, p98) {
      const len = data.length;

      // Find min/max for histogram range
      let min = Infinity, max = -Infinity;
      for (let i = 0; i < len; i++) {
        const v = data[i];
        if (v < min) min = v;
        if (v > max) max = v;
      }

      // Create histogram with 1024 buckets (good balance of accuracy vs speed)
      const numBuckets = 1024;
      const histogram = new Uint32Array(numBuckets);
      const range = max - min || 1;
      const bucketSize = range / numBuckets;

      // Build histogram
      for (let i = 0; i < len; i++) {
        const bucketIdx = Math.min(numBuckets - 1, Math.floor((data[i] - min) / bucketSize));
        histogram[bucketIdx]++;
      }

      // Find percentile values from histogram
      const p2_target = Math.floor(len * p2);
      const p98_target = Math.floor(len * p98);

      let count = 0;
      let p2_val = min, p98_val = max;
      let foundP2 = false, foundP98 = false;

      for (let i = 0; i < numBuckets; i++) {
        count += histogram[i];

        if (!foundP2 && count >= p2_target) {
          p2_val = min + (i + 0.5) * bucketSize;
          foundP2 = true;
        }

        if (!foundP98 && count >= p98_target) {
          p98_val = min + (i + 0.5) * bucketSize;
          foundP98 = true;
          break;
        }
      }

      return { min: p2_val, max: p98_val };
    }

    // Convert depth to U8 using robust min/max (2-98 percentiles), invert so white=near
    function depthToU8(depthFloat, width, height) {
      const data = depthFloat.data;
      const len = data.length;

      // Fast histogram-based percentile calculation (O(n) instead of O(n log n))
      const { min: minVal, max: maxVal } = findPercentilesHistogram(data, 0.02, 0.98);
      const range = maxVal - minVal || 1;
      const invRange = 1.0 / range;

      // Convert to U8, inverting so white=near (closer), black=far
      // Optimized: pre-calculate constants, minimize operations per pixel
      const u8Data = new Uint8ClampedArray(width * height * 4);

      for (let i = 0; i < len; i++) {
        // Normalize and clamp in one operation
        let normalized = (data[i] - minVal) * invRange;
        normalized = normalized < 0 ? 0 : normalized > 1 ? 1 : normalized;

        // Invert and convert: (1 - normalized) * 255
        const value = ((1 - normalized) * 255) | 0; // Bitwise OR for fast floor

        const idx = i * 4;
        u8Data[idx] = value;
        u8Data[idx + 1] = value;
        u8Data[idx + 2] = value;
        u8Data[idx + 3] = 255;
      }

      return u8Data;
    }

    // Apply Depth Anything V2 to get RAW depth map (normalized grayscale 0-255)
    // Returns normalized depth without LUT application - LUT applied later for fast parameter changes
    async function applyDepthAnythingV2Raw(img) {
      const startTime = performance.now();

      // Get depth prediction
      const depthFloat = await predictDepthFloat(img);
      const inferenceTime = performance.now();
      console.log(`Depth inference: ${(inferenceTime - startTime).toFixed(1)}ms`);

      // Convert to U8 grayscale (normalized, inverted depth)
      const u8Data = depthToU8(depthFloat, depthFloat.width, depthFloat.height);
      const convertTime = performance.now();
      console.log(`Depth normalization: ${(convertTime - inferenceTime).toFixed(1)}ms`);

      // Create ImageData and resize back to original image dimensions
      const tempCanvas = document.createElement('canvas');
      tempCanvas.width = depthFloat.width;
      tempCanvas.height = depthFloat.height;
      const tempCtx = tempCanvas.getContext('2d', { willReadFrequently: false });
      const tempImageData = new ImageData(u8Data, depthFloat.width, depthFloat.height);
      tempCtx.putImageData(tempImageData, 0, 0);

      // Resize to original image size
      const outputCanvas = document.createElement('canvas');
      outputCanvas.width = img.width;
      outputCanvas.height = img.height;
      const outputCtx = outputCanvas.getContext('2d', { willReadFrequently: false });
      const useFastResize = Math.abs(img.width - 518) < 50 && Math.abs(img.height - 518) < 50;
      if (useFastResize) {
        outputCtx.imageSmoothingEnabled = false;
      }
      outputCtx.drawImage(tempCanvas, 0, 0, img.width, img.height);

      const result = outputCtx.getImageData(0, 0, img.width, img.height);
      const totalTime = performance.now();
      console.log(`Total AI depth processing: ${(totalTime - startTime).toFixed(1)}ms`);

      return result;
    }

    // Apply LUT transformation to cached raw depth (fast, for parameter changes)
    // This allows real-time adjustment of depth gamma, invert, and depth level without re-running AI
    function applyDepthLUT(rawDepthImageData, nearDepth, farDepth, invertDepth, depthGamma) {
      const width = rawDepthImageData.width;
      const height = rawDepthImageData.height;
      const rawData = rawDepthImageData.data;

      // Create depth LUT
      const depthLUT = makeDepthLUT(nearDepth, farDepth, invertDepth, depthGamma);

      // Apply LUT to grayscale depth
      const result = new ImageData(width, height);
      const resultData = result.data;

      for (let i = 0; i < width * height; i++) {
        const idx = i * 4;
        const alpha = rawData[idx + 3];

        let depth;
        if (alpha < 8) {
          // Transparent pixels always go to background depth
          depth = 0;
        } else {
          // Apply LUT to grayscale value
          const gray = rawData[idx]; // R channel (grayscale)
          depth = depthLUT[gray];
        }

        resultData[idx] = depth;
        resultData[idx + 1] = depth;
        resultData[idx + 2] = depth;
        resultData[idx + 3] = 255; // Full opacity in depth map
      }

      return result;
    }

// ============================================================================
// IMAGE PROCESSING UTILITIES
// ============================================================================

    // --- Blue-noise loader (tileable PNG), with safe fallback to hash if not ready ---
    const BlueNoise = { img: new Image(), ready: false, data: null, w: 0, h: 0 };
    BlueNoise.img.crossOrigin = 'anonymous';
    BlueNoise.img.src = 'assets/images/blue_noise_256.png';  // put a tileable PNG here (128–256 px square)
    
    BlueNoise.img.onload = () => {
      const c = document.createElement('canvas');
      c.width = BlueNoise.img.naturalWidth;
      c.height = BlueNoise.img.naturalHeight;
      const cx = c.getContext('2d');
      cx.drawImage(BlueNoise.img, 0, 0);
      const g = cx.getImageData(0, 0, c.width, c.height);
      BlueNoise.data = g.data;
      BlueNoise.w = c.width;
      BlueNoise.h = c.height;
      BlueNoise.ready = true;
    };
    BlueNoise.img.onerror = () => { BlueNoise.ready = false; }; // fallback to hash

    // Deterministic 32-bit hash -> [0..2^32-1]
    function hash32(n) {
      n = (n ^ 61) ^ (n >>> 16);
      n = n + (n << 3);
      n = n ^ (n >>> 4);
      n = n * 0x27d4eb2d;
      n = n ^ (n >>> 15);
      return n >>> 0;
    }

    // Map 0..255 grayscale -> 0..255 depth using a precomputed LUT
    function makeDepthLUT(nearDepth /*white*/, farDepth /*black*/, invert=false, gamma=1.0) {
      // clamp inputs to 0..255
      nearDepth = Math.max(0, Math.min(255, nearDepth));
      farDepth  = Math.max(0, Math.min(255,  farDepth));
      const lut = new Uint8ClampedArray(256);
    
      for (let g=0; g<256; g++) {
        // normalized grayscale (0=black .. 1=white)
        let t = g / 255;
        // gamma for perceptual control (1.0 = linear)
        if (gamma !== 1) t = Math.pow(t, gamma);
        // optional invert (t -> 1-t)
        if (invert) t = 1 - t;
    
        // linear interpolate between far (black) and near (white)
        const d = farDepth + (nearDepth - farDepth) * t;
        lut[g] = d < 0 ? 0 : d > 255 ? 255 : d|0;
      }
      return lut;
    }

    function makeAutoLevelsLUT(id /*Uint8ClampedArray RGBA*/) {
      let lo = 255, hi = 0;
      for (let i=0; i<id.length; i+=4) {
        const r=id[i], g=id[i+1], b=id[i+2], a=id[i+3];
        if (a < 8) continue;
        const y = (0.2126*r + 0.7152*g + 0.0722*b) | 0;
        if (y < lo) lo = y;
        if (y > hi) hi = y;
      }
      const lut = new Uint8ClampedArray(256);
      const span = Math.max(1, hi - lo);
      for (let v=0; v<256; v++) {
        const t = (v - lo) / span;
        lut[v] = Math.max(0, Math.min(255, (t * 255)|0));
      }
      return lut;
    }

    // Create LUT from explicit min/max values (for consistent video normalization)
    function makeLUTFromRange(lo, hi) {
      const lut = new Uint8ClampedArray(256);
      const span = Math.max(1, hi - lo);
      for (let v=0; v<256; v++) {
        const t = (v - lo) / span;
        lut[v] = Math.max(0, Math.min(255, (t * 255)|0));
      }
      return lut;
    }

    // Analyze image to get min/max for normalization (with optional scaling)
    function getImageRange(id /*Uint8ClampedArray RGBA*/, scaleFactor = 1.0) {
      let lo = 255, hi = 0;
      for (let i=0; i<id.length; i+=4) {
        const r=id[i], g=id[i+1], b=id[i+2], a=id[i+3];
        if (a < 8) continue;
        const y = (0.2126*r + 0.7152*g + 0.0722*b) | 0;
        if (y < lo) lo = y;
        if (y > hi) hi = y;
      }
      // Apply scale factor (e.g., 0.8 to set peak at 80%)
      const span = hi - lo;
      const scaledSpan = span / scaleFactor;
      const newHi = lo + scaledSpan;
      return { lo, hi: Math.min(255, newHi) };
    }

    // DOM Elements
// ============================================================================
// DOM ELEMENT REFERENCES
// ============================================================================

    const previewCanvas = document.getElementById('previewCanvas');
    const ctx = previewCanvas.getContext('2d', { willReadFrequently: true });
    const layerList = document.getElementById('layerList');
    const layerSettings = document.getElementById('layerSettings');

    // Layer management
    const addTextBtn = document.getElementById('addTextBtn');
    const addImageBtn = document.getElementById('addImageBtn');
    const addVideoBtn = document.getElementById('addVideoBtn');
    const imageFileInput = document.getElementById('imageFileInput');
    const videoFileInput = document.getElementById('videoFileInput');

    // Layer controls
    const textContentGroup = document.getElementById('textContentGroup');
    const fontGroup = document.getElementById('fontGroup');
    const layerText = document.getElementById('layerText');
    const layerFont = document.getElementById('layerFont');
    const customFontInput = document.getElementById('customFontInput');
    const layerSize = document.getElementById('layerSize');
    const layerDepth = document.getElementById('layerDepth');
    const layerRotation = document.getElementById('layerRotation');
    const layerX = document.getElementById('layerX');
    const layerY = document.getElementById('layerY');
    const depthAlgorithmGroup = document.getElementById('depthAlgorithmGroup');
    const layerDepthAlgorithm = document.getElementById('layerDepthAlgorithm');
    const retinexRadiusGroup = document.getElementById('retinexRadiusGroup');
    const layerRetinexRadius = document.getElementById('layerRetinexRadius');
    const depthGammaGroup = document.getElementById('depthGammaGroup');
    const layerDepthGamma = document.getElementById('layerDepthGamma');
    const invertDepthGroup = document.getElementById('invertDepthGroup');
    const invertDepth = document.getElementById('invertDepth');
    const downloadDepthmapGroup = document.getElementById('downloadDepthmapGroup');
    const downloadDepthmapBtn = document.getElementById('downloadDepthmapBtn');
    const videoFrameGroup = document.getElementById('videoFrameGroup');
    const videoFrameSlider = document.getElementById('videoFrameSlider');
    const videoFrameInput = document.getElementById('videoFrameInput');
    const videoFrameInfo = document.getElementById('videoFrameInfo');

    // Global controls
    const patternDensity = document.getElementById('patternDensity');
    const stripWidth = document.getElementById('stripWidth');
    const depthScale = document.getElementById('depthScale');
    const outputWidth = document.getElementById('outputWidth');
    const colorScheme = document.getElementById('colorScheme');

    // Actions
    const regenerateBtn = document.getElementById('regenerateBtn');
    const downloadBtn = document.getElementById('downloadBtn');
    const downloadVideoBtn = document.getElementById('downloadVideoBtn');
    const videoProgress = document.getElementById('videoProgress');
    const videoProgressBar = document.getElementById('videoProgressBar');
    const videoProgressText = document.getElementById('videoProgressText');
    const stopRenderingBtn = document.getElementById('stopRenderingBtn');

    // Flag to track if rendering should stop
    let shouldStopRendering = false;

// ============================================================================
// LAYER MANAGEMENT - ADD/EDIT LAYERS
// ============================================================================

    // Add Text Layer
    addTextBtn.addEventListener('click', () => {
      const layer = {
        id: nextLayerId++,
        type: 'text',
        text: 'YOUR\nTEXT\nHERE',
        font: "'Arial Black', sans-serif",
        size: 200,
        depth: 15,
        rotation: 0,
        x: state.settings.outputWidth / 2,
        y: state.settings.outputHeight / 4
      };
      state.layers.push(layer);
      updateLayerList();
      selectLayer(layer.id);
      generateAutostereogram();
    });

    // Apply fast box blur approximation of Gaussian blur
    // This is a standard technique: multiple box blur passes approximate Gaussian
    function fastBoxBlur(imageData, radius) {
      const w = imageData.width;
      const h = imageData.height;
      const data = imageData.data;

      // Create temporary buffer for grayscale values
      const gray = new Float32Array(w * h);
      for (let i = 0; i < gray.length; i++) {
        const idx = i * 4;
        gray[i] = 0.2126 * data[idx] + 0.7152 * data[idx + 1] + 0.0722 * data[idx + 2];
      }

      // Apply horizontal box blur
      const temp = new Float32Array(w * h);
      for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
          let sum = 0;
          let count = 0;
          for (let dx = -radius; dx <= radius; dx++) {
            const nx = x + dx;
            if (nx >= 0 && nx < w) {
              sum += gray[y * w + nx];
              count++;
            }
          }
          temp[y * w + x] = sum / count;
        }
      }

      // Apply vertical box blur
      const blurred = new Float32Array(w * h);
      for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
          let sum = 0;
          let count = 0;
          for (let dy = -radius; dy <= radius; dy++) {
            const ny = y + dy;
            if (ny >= 0 && ny < h) {
              sum += temp[ny * w + x];
              count++;
            }
          }
          blurred[y * w + x] = sum / count;
        }
      }

      return blurred;
    }

    // Process image: convert to grayscale with auto-levels, preserve alpha
    // If providedLUT is given, use it instead of computing a new one (for consistent video normalization)
    function processImageToGrayscale(img, providedLUT = null) {
      const canvas = document.createElement('canvas');
      canvas.width = img.width;
      canvas.height = img.height;
      const ctx = canvas.getContext('2d');

      // Draw original image
      ctx.drawImage(img, 0, 0);
      const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
      const data = imageData.data;

      // Use provided LUT or create auto-levels LUT for this image
      const autoLUT = providedLUT || makeAutoLevelsLUT(data);

      // Convert to grayscale with auto-levels applied, preserve alpha channel
      for (let i = 0; i < data.length; i += 4) {
        const alpha = data[i + 3];

        // Convert to grayscale using Rec. 709 coefficients
        const gray = Math.floor(0.2126 * data[i] + 0.7152 * data[i + 1] + 0.0722 * data[i + 2]);

        // Apply auto-levels
        const stretched = autoLUT[gray];

        // Store as grayscale, preserving original alpha
        data[i] = stretched;
        data[i + 1] = stretched;
        data[i + 2] = stretched;
        data[i + 3] = alpha; // Preserve original alpha
      }

      ctx.putImageData(imageData, 0, 0);

      // Convert canvas to image
      const processedImg = new Image();
      processedImg.src = canvas.toDataURL();
      return processedImg;
    }

    // Apply shadow-aware processing using Retinex-inspired approach
    //
    // This implementation is based on classic, long-standing academic signal processing techniques:
    // 1. Illumination estimation via large-scale Gaussian blur (pre-1970s technique)
    // 2. Division to extract reflectance/albedo (fundamental to Retinex theory, Land & McCann 1971)
    // 3. Auto-level normalization for optimal contrast (standard histogram stretching)
    //
    // The goal is to separate surface brightness (which should affect depth) from lighting conditions
    // (which should not). This prevents dark shadows from creating false depth holes while preserving
    // genuinely dark objects as distant.
    //
    // This implementation uses only fundamental, unpatented techniques from academic literature.
    // It does NOT implement any specific patented Retinex variants (e.g., MSRCR, MSRCP).
    // The approach follows the original Retinex concept paper:
    // Land, E. H., & McCann, J. J. (1971). "Lightness and retinex theory."
    // Journal of the Optical Society of America, 61(1), 1-11.
    //
    function applyShadowAwareDepth(imageData, radius) {
      const w = imageData.width;
      const h = imageData.height;
      const data = imageData.data;

      // Step 1: Estimate illumination using large-scale blur
      // This approximates the lighting field without capturing fine detail
      const illumination = fastBoxBlur(imageData, radius);

      // Step 2: Calculate reflectance (albedo) by dividing by illumination
      // Add small epsilon to avoid division by zero
      const epsilon = 1.0;
      const reflectance = new Uint8ClampedArray(w * h);

      for (let i = 0; i < w * h; i++) {
        const idx = i * 4;
        const gray = 0.2126 * data[idx] + 0.7152 * data[idx + 1] + 0.0722 * data[idx + 2];
        const illum = illumination[i] + epsilon;

        // Normalize reflectance to 0-255 range
        // This separates surface brightness from lighting
        const refl = Math.min(255, (gray / illum) * 128);
        reflectance[i] = refl;
      }

      // Step 3: Apply auto-levels to reflectance for better depth range
      // Create temporary RGBA data for autolevels function, preserving alpha
      const tempData = new Uint8ClampedArray(w * h * 4);
      for (let i = 0; i < w * h; i++) {
        const idx = i * 4;
        tempData[idx] = reflectance[i];
        tempData[idx + 1] = reflectance[i];
        tempData[idx + 2] = reflectance[i];
        tempData[idx + 3] = data[idx + 3]; // Preserve original alpha
      }
      const autoLUT = makeAutoLevelsLUT(tempData);

      // Step 4: Apply auto-levels and store result, preserving alpha
      for (let i = 0; i < w * h; i++) {
        const stretched = autoLUT[reflectance[i]];
        const idx = i * 4;
        const alpha = data[idx + 3];
        data[idx] = stretched;
        data[idx + 1] = stretched;
        data[idx + 2] = stretched;
        data[idx + 3] = alpha; // Preserve original alpha
      }

      return imageData;
    }

    // Add Image Layer
    addImageBtn.addEventListener('click', () => {
      imageFileInput.click();
    });

    imageFileInput.addEventListener('change', async (e) => {
      const file = e.target.files[0];
      if (!file) return;

      const img = new Image();
      img.onload = () => {
        const layer = {
          id: nextLayerId++,
          type: 'image',
          originalImage: img,
          image: img, // Will be processed
          size: 50, // Multiplied by 4 internally = 200px actual
          depth: 20,
          rotation: 0,
          x: state.settings.outputWidth / 2,
          y: state.settings.outputHeight / 2,
          depthAlgorithm: 'depth-anything-v2',
          retinexRadius: 32,
          invertDepth: false,
          depthGamma: 1.0
        };

        // Process the image to grayscale with auto-levels
        layer.image = processImageToGrayscale(img);

        state.layers.push(layer);
        updateLayerList();
        selectLayer(layer.id);

        // Wait for processed image to load before generating
        layer.image.onload = () => generateAutostereogram();
      };
      img.src = URL.createObjectURL(file);
      imageFileInput.value = '';
    });

// ============================================================================
// VIDEO LAYER FUNCTIONS
// ============================================================================

    // Helper function to extract frame from video
    async function extractVideoFrame(video, time = 0) {
      return new Promise((resolve) => {
        video.currentTime = time;
        video.onseeked = () => {
          const canvas = document.createElement('canvas');
          canvas.width = video.videoWidth;
          canvas.height = video.videoHeight;
          const ctx = canvas.getContext('2d');
          ctx.drawImage(video, 0, 0);

          const img = new Image();
          img.onload = () => resolve(img);
          img.src = canvas.toDataURL();
        };
      });
    }

    // Update active video layer to show a specific frame
    async function updateVideoFrame(frameIndex) {
      const activeLayer = getActiveLayer();
      if (!activeLayer || activeLayer.type !== 'video') return;

      const time = frameIndex / activeLayer.fps;
      const frame = await extractVideoFrame(activeLayer.videoElement, time);

      // Update the layer's current frame
      activeLayer.currentFrame = frameIndex;
      activeLayer.originalImage = frame;

      // Clear depth cache to force recomputation
      depthCache.delete(activeLayer.id);

      // Process the frame
      activeLayer.image = processImageToGrayscale(frame);

      // Wait for image to load
      await new Promise(resolve => {
        if (activeLayer.image.complete) {
          resolve();
        } else {
          activeLayer.image.onload = resolve;
        }
      });

      // Update the info text
      videoFrameInfo.textContent = `Frame ${frameIndex} of ${Math.ceil(activeLayer.duration * activeLayer.fps) - 1} (${time.toFixed(2)}s)`;

      // Regenerate the preview
      generateAutostereogram();
    }

    // Add Video Layer
    addVideoBtn.addEventListener('click', () => {
      videoFileInput.click();
    });

    videoFileInput.addEventListener('change', async (e) => {
      const file = e.target.files[0];
      if (!file) return;

      const video = document.createElement('video');
      video.preload = 'auto'; // Changed to 'auto' for better seeking
      video.muted = true; // Mute to avoid audio playback
      video.playsInline = true; // Better mobile compatibility

      video.onloadedmetadata = async () => {
        try {
          // Wait for video to be fully seekable
          await new Promise(resolve => {
            if (video.readyState >= 2) {
              resolve();
            } else {
              video.onloadeddata = resolve;
            }
          });

          console.log('Video loaded, duration:', video.duration);

          // Extract first frame
          const firstFrame = await extractVideoFrame(video, 0);
          console.log('First frame extracted:', firstFrame.width, 'x', firstFrame.height);

          // Process the first frame to grayscale with auto-levels
          const processedImage = processImageToGrayscale(firstFrame);

          // Wait for processed image to load
          await new Promise(resolve => {
            if (processedImage.complete) {
              resolve();
            } else {
              processedImage.onload = resolve;
            }
          });

          console.log('Processed image loaded');

          // Try to detect framerate (defaults to 30fps if not available)
          // Note: HTML5 video API doesn't expose framerate directly, so we use a common default
          // Advanced: could parse file metadata or estimate from duration/frames
          const fps = 30; // Default to 30fps (common for web video)

          const layer = {
            id: nextLayerId++,
            type: 'video',
            videoElement: video,
            videoURL: URL.createObjectURL(file),
            duration: video.duration,
            fps: fps, // Store framerate for consistent playback
            currentFrame: 0, // Track which frame is currently displayed
            originalImage: firstFrame,
            image: processedImage,
            size: 50,
            depth: 20,
            rotation: 0,
            x: state.settings.outputWidth / 2,
            y: state.settings.outputHeight / 2,
            depthAlgorithm: 'depth-anything-v2',
            retinexRadius: 32,
            invertDepth: false,
            depthGamma: 1.0
          };

          console.log(`Video layer ${layer.id}: ${fps}fps, ${video.duration.toFixed(2)}s`);

          state.layers.push(layer);
          updateLayerList();
          selectLayer(layer.id);

          // Generate stereogram
          console.log('Generating stereogram for video layer');
          await generateAutostereogram();
        } catch (error) {
          console.error('Error loading video layer:', error);
          alert('Error loading video: ' + error.message);
        }
      };

      video.src = URL.createObjectURL(file);
      videoFileInput.value = '';
    });

    // Custom Font Upload
    customFontInput.addEventListener('change', async (e) => {
      const file = e.target.files[0];
      if (!file) return;

      const fontName = 'CustomFont_' + Date.now();
      const arrayBuffer = await file.arrayBuffer();
      const fontFace = new FontFace(fontName, arrayBuffer);
      await fontFace.load();
      document.fonts.add(fontFace);

      state.customFonts.set(fontName, fontName);

      const activeLayer = getActiveLayer();
      if (activeLayer && activeLayer.type === 'text') {
        activeLayer.font = fontName;
        generateAutostereogram();
      }

      customFontInput.value = '';
    });

    layerFont.addEventListener('change', (e) => {
      if (e.target.value === 'custom') {
        customFontInput.click();
      } else {
        const activeLayer = getActiveLayer();
        if (activeLayer && activeLayer.type === 'text') {
          activeLayer.font = e.target.value;
          generateAutostereogram();
        }
      }
    });

// ============================================================================
// EVENT LISTENERS - LAYER SETTINGS
// ============================================================================

    // Layer Settings Event Listeners
    layerText.addEventListener('input', () => {
      const activeLayer = getActiveLayer();
      if (activeLayer && activeLayer.type === 'text') {
        activeLayer.text = layerText.value;
        updateLayerList();
        generateAutostereogram();
      }
    });

    layerSize.addEventListener('input', (e) => {
      const activeLayer = getActiveLayer();
      if (activeLayer) {
        activeLayer.size = parseInt(e.target.value);
        document.getElementById('layerSizeValue').textContent = e.target.value;
        generateAutostereogram();
      }
    });

    layerDepthAlgorithm.addEventListener('change', (e) => {
      const activeLayer = getActiveLayer();
      if (activeLayer && (activeLayer.type === 'image' || activeLayer.type === 'video')) {
        activeLayer.depthAlgorithm = e.target.value;
        // Show/hide retinex radius based on algorithm
        retinexRadiusGroup.style.display = e.target.value === 'retinex' ? 'block' : 'none';
        // Invalidate cache when algorithm changes (will re-run on next generation)
        // Note: cache check in generateDepthMap will handle this, but explicit delete is cleaner
        depthCache.delete(activeLayer.id);

        // Show/hide download depthmap button based on algorithm
        if (e.target.value === 'depth-anything-v2') {
          downloadDepthmapGroup.style.display = 'block';
          downloadDepthmapBtn.disabled = true; // Disabled until depth is computed
        } else {
          downloadDepthmapGroup.style.display = 'none';
        }

        generateAutostereogram();
      }
    });

    layerRetinexRadius.addEventListener('input', (e) => {
      const activeLayer = getActiveLayer();
      if (activeLayer && activeLayer.type === 'image') {
        activeLayer.retinexRadius = parseInt(e.target.value);
        document.getElementById('layerRetinexRadiusValue').textContent = e.target.value + 'px';
        generateAutostereogram();
      }
    });

    layerDepthGamma.addEventListener('input', (e) => {
      const activeLayer = getActiveLayer();
      if (activeLayer && (activeLayer.type === 'image' || activeLayer.type === 'video')) {
        activeLayer.depthGamma = parseFloat(e.target.value);
        document.getElementById('layerDepthGammaValue').textContent = e.target.value;
        generateAutostereogram();
      }
    });

    invertDepth.addEventListener('change', (e) => {
      const activeLayer = getActiveLayer();
      if (activeLayer && (activeLayer.type === 'image' || activeLayer.type === 'video')) {
        activeLayer.invertDepth = e.target.checked;
        generateAutostereogram();
      }
    });

    layerDepth.addEventListener('input', (e) => {
      const activeLayer = getActiveLayer();
      if (activeLayer) {
        activeLayer.depth = parseInt(e.target.value);
        document.getElementById('layerDepthValue').textContent = e.target.value;
        generateAutostereogram();
      }
    });

    layerRotation.addEventListener('input', (e) => {
      const activeLayer = getActiveLayer();
      if (activeLayer) {
        activeLayer.rotation = parseInt(e.target.value);
        document.getElementById('layerRotationValue').textContent = e.target.value + '°';
        generateAutostereogram();
      }
    });

    layerX.addEventListener('input', (e) => {
      const activeLayer = getActiveLayer();
      if (activeLayer) {
        activeLayer.x = parseInt(e.target.value);
        document.getElementById('layerXValue').textContent = e.target.value;
        generateAutostereogram();
      }
    });

    layerY.addEventListener('input', (e) => {
      const activeLayer = getActiveLayer();
      if (activeLayer) {
        activeLayer.y = parseInt(e.target.value);
        document.getElementById('layerYValue').textContent = e.target.value;
        generateAutostereogram();
      }
    });

    // Video Frame Scrubber Event Listeners
    videoFrameSlider.addEventListener('input', async (e) => {
      const frameIndex = parseInt(e.target.value);
      videoFrameInput.value = frameIndex;
      await updateVideoFrame(frameIndex);
    });

    videoFrameInput.addEventListener('input', async (e) => {
      const frameIndex = parseInt(e.target.value) || 0;
      const activeLayer = getActiveLayer();
      if (activeLayer && activeLayer.type === 'video') {
        const maxFrame = Math.ceil(activeLayer.duration * activeLayer.fps) - 1;
        const clampedFrame = Math.max(0, Math.min(frameIndex, maxFrame));
        videoFrameInput.value = clampedFrame;
        videoFrameSlider.value = clampedFrame;
        await updateVideoFrame(clampedFrame);
      }
    });

// ============================================================================
// EVENT LISTENERS - GLOBAL SETTINGS
// ============================================================================

    // Global Settings Event Listeners
    patternDensity.addEventListener('input', (e) => {
      state.settings.patternDensity = parseInt(e.target.value);
      document.getElementById('patternDensityValue').textContent = e.target.value;
      generateAutostereogram();
    });

    stripWidth.addEventListener('input', (e) => {
      state.settings.stripWidth = parseInt(e.target.value);
      document.getElementById('stripWidthValue').textContent = e.target.value + 'px';
      generateAutostereogram();
    });

    depthScale.addEventListener('input', (e) => {
      state.settings.depthScale = parseFloat(e.target.value);
      document.getElementById('depthScaleValue').textContent = e.target.value;
      generateAutostereogram();
    });

    outputWidth.addEventListener('input', (e) => {
      state.settings.outputWidth = parseInt(e.target.value);
      state.settings.outputHeight = Math.round(state.settings.outputWidth * 2 / 3);
      document.getElementById('outputWidthValue').textContent = e.target.value + 'px';

      previewCanvas.width = state.settings.outputWidth;
      previewCanvas.height = state.settings.outputHeight;

      // Update position sliders max values
      layerX.max = state.settings.outputWidth;
      layerY.max = state.settings.outputHeight;

      generateAutostereogram();
    });

    colorScheme.addEventListener('change', (e) => {
      state.settings.colorScheme = e.target.value;
      generateAutostereogram();
    });

    regenerateBtn.addEventListener('click', () => {
      generateAutostereogram();
    });

    downloadBtn.addEventListener('click', () => {
      const link = document.createElement('a');
      link.download = 'apoculus-3d-image.png';
      link.href = previewCanvas.toDataURL();
      link.click();
    });

// ============================================================================
// VIDEO RENDERING - EXPORT FUNCTIONS
// ============================================================================

    // Helper function to invert depth map (so white = foreground, black = background)
    function invertDepthMap(imageData) {
      const inverted = new ImageData(imageData.width, imageData.height);
      for (let i = 0; i < imageData.data.length; i += 4) {
        inverted.data[i] = 255 - imageData.data[i];     // Invert R
        inverted.data[i + 1] = 255 - imageData.data[i + 1]; // Invert G
        inverted.data[i + 2] = 255 - imageData.data[i + 2]; // Invert B
        inverted.data[i + 3] = imageData.data[i + 3];   // Keep alpha
      }
      return inverted;
    }

    // Download Depthmap (still image or video)
    downloadDepthmapBtn.addEventListener('click', async () => {
      const activeLayer = getActiveLayer();
      if (!activeLayer || (activeLayer.type !== 'image' && activeLayer.type !== 'video')) {
        alert('Please select an image or video layer');
        return;
      }

      // Check if we have a cached depth map for Depth Anything V2
      const cacheEntry = depthCache.get(activeLayer.id);
      if (!cacheEntry || cacheEntry.algorithm !== 'depth-anything-v2') {
        alert('No Depth Anything V2 depth map available. Please ensure the layer uses Depth Anything V2 and has been rendered at least once.');
        return;
      }

      if (activeLayer.type === 'image') {
        // Download still image depthmap
        const rawDepth = cacheEntry.rawDepth;

        // Invert the depth map
        const invertedDepth = invertDepthMap(rawDepth);

        // Create a canvas to convert the ImageData to a downloadable image
        const canvas = document.createElement('canvas');
        canvas.width = invertedDepth.width;
        canvas.height = invertedDepth.height;
        const ctx = canvas.getContext('2d');
        ctx.putImageData(invertedDepth, 0, 0);

        // Download as PNG
        const link = document.createElement('a');
        link.download = `depthmap-layer${activeLayer.id}.png`;
        link.href = canvas.toDataURL('image/png');
        link.click();

        console.log('Downloaded depth map:', invertedDepth.width, 'x', invertedDepth.height);
      } else if (activeLayer.type === 'video') {
        // Download video depthmap
        await renderDepthmapVideo(activeLayer);
      }
    });

    // Render depthmap video for a video layer
    async function renderDepthmapVideo(layer) {
      const fps = layer.fps || 30; // Use layer's native framerate
      const totalFrames = Math.ceil(layer.duration * fps);

      console.log(`Rendering depthmap video at ${fps}fps (${totalFrames} frames)`);

      // Reset stop flag
      shouldStopRendering = false;

      // Show progress
      videoProgress.style.display = 'block';
      downloadDepthmapBtn.disabled = true;

      try {
        // PHASE 1: Pre-render all frames (slow - depth prediction)
        videoProgressText.textContent = 'Phase 1: Computing depth maps...';
        const renderedFrames = [];
        let canvasWidth, canvasHeight;

        for (let frameIndex = 0; frameIndex < totalFrames; frameIndex++) {
          // Check if user requested stop
          if (shouldStopRendering) {
            console.log('Depthmap rendering stopped by user at frame', frameIndex);
            break;
          }

          const time = frameIndex / fps;

          // Update progress
          const progress = ((frameIndex + 1) / totalFrames) * 50; // First 50%
          videoProgressBar.style.width = progress + '%';
          videoProgressText.textContent = `Phase 1: Computing depth ${frameIndex + 1}/${totalFrames}`;

          // Extract frame at the current time
          const frame = await extractVideoFrame(layer.videoElement, time);

          // Run depth prediction on this frame
          let rawDepth;
          try {
            rawDepth = await applyDepthAnythingV2Raw(frame);
          } catch (error) {
            console.error('Depth prediction failed for frame', frameIndex, error);
            continue;
          }

          // Store dimensions from first frame
          if (frameIndex === 0) {
            canvasWidth = rawDepth.width;
            canvasHeight = rawDepth.height;
          }

          // Invert the depth map and store
          const invertedDepth = invertDepthMap(rawDepth);
          renderedFrames.push(invertedDepth);
        }

        if (renderedFrames.length === 0) {
          throw new Error('No frames were rendered');
        }

        // PHASE 2: Encode frames at correct framerate (fast - just drawing)
        videoProgressText.textContent = 'Phase 2: Encoding video...';

        const depthmapCanvas = document.createElement('canvas');
        depthmapCanvas.width = canvasWidth;
        depthmapCanvas.height = canvasHeight;
        const stream = depthmapCanvas.captureStream(fps);

        // Try different codecs - prefer WebM VP9 for better quality/compression
        let options, fileExtension, mimeType;
        const highBitrate = 20000000; // 20 Mbps - high quality for depthmap details

        if (MediaRecorder.isTypeSupported('video/webm;codecs=vp9')) {
          options = { mimeType: 'video/webm;codecs=vp9', videoBitsPerSecond: highBitrate };
          fileExtension = 'webm';
          mimeType = 'video/webm';
        } else if (MediaRecorder.isTypeSupported('video/mp4')) {
          options = { mimeType: 'video/mp4', videoBitsPerSecond: highBitrate };
          fileExtension = 'mp4';
          mimeType = 'video/mp4';
        } else if (MediaRecorder.isTypeSupported('video/webm;codecs=vp8')) {
          options = { mimeType: 'video/webm;codecs=vp8', videoBitsPerSecond: highBitrate };
          fileExtension = 'webm';
          mimeType = 'video/webm';
        } else if (MediaRecorder.isTypeSupported('video/webm')) {
          options = { mimeType: 'video/webm', videoBitsPerSecond: highBitrate };
          fileExtension = 'webm';
          mimeType = 'video/webm';
        } else {
          options = { videoBitsPerSecond: highBitrate };
          fileExtension = 'webm';
          mimeType = 'video/webm';
        }

        console.log(`Depthmap video: using ${fileExtension.toUpperCase()} format at ${(highBitrate/1000000).toFixed(0)} Mbps`);

        const chunks = [];
        const mediaRecorder = new MediaRecorder(stream, options);

        mediaRecorder.ondataavailable = (e) => {
          if (e.data.size > 0) {
            chunks.push(e.data);
          }
        };

        mediaRecorder.onstop = () => {
          const blob = new Blob(chunks, { type: mimeType });
          const url = URL.createObjectURL(blob);
          const link = document.createElement('a');
          link.download = `depthmap-video-layer${layer.id}.${fileExtension}`;
          link.href = url;
          link.click();
          URL.revokeObjectURL(url);

          // Hide progress and reset
          videoProgress.style.display = 'none';
          downloadDepthmapBtn.disabled = false;
          shouldStopRendering = false;
        };

        mediaRecorder.start();

        // Draw pre-rendered frames at correct framerate
        const ctx = depthmapCanvas.getContext('2d');
        const frameDuration = 1000 / fps;

        for (let i = 0; i < renderedFrames.length; i++) {
          const progress = 50 + ((i + 1) / renderedFrames.length) * 50; // Second 50%
          videoProgressBar.style.width = progress + '%';
          videoProgressText.textContent = `Phase 2: Encoding frame ${i + 1}/${renderedFrames.length}`;

          ctx.putImageData(renderedFrames[i], 0, 0);
          await new Promise(resolve => setTimeout(resolve, frameDuration));
        }

        // Stop recording
        mediaRecorder.stop();
      } catch (error) {
        console.error('Error rendering depthmap video:', error);
        alert('Error rendering depthmap video: ' + error.message);
        videoProgress.style.display = 'none';
        downloadDepthmapBtn.disabled = false;
        shouldStopRendering = false;
      }
    }

    // Video rendering function
    async function renderVideo() {
      // Find all video layers
      const videoLayers = state.layers.filter(l => l.type === 'video');

      if (videoLayers.length === 0) {
        alert('No video layers found. Add at least one video layer to render a video.');
        return;
      }

      // Find the longest video and use its framerate
      const longestVideo = videoLayers.reduce((longest, current) =>
        current.duration > longest.duration ? current : longest
      );
      const fps = longestVideo.fps || 30; // Use longest video's framerate
      const maxDuration = longestVideo.duration;
      const totalFrames = Math.ceil(maxDuration * fps);

      console.log(`Rendering stereogram video at ${fps}fps from longest video (${maxDuration.toFixed(2)}s, ${totalFrames} frames)`);

      // Reset stop flag
      shouldStopRendering = false;

      // Show progress
      videoProgress.style.display = 'block';
      downloadVideoBtn.disabled = true;

      try {

        // PHASE 1: Pre-render all stereogram frames (slow - depth + stereogram generation)
        videoProgressText.textContent = 'Phase 1: Rendering stereograms...';

        // Compute consistent normalization from first frame of each video layer (80% scaling for headroom)
        const videoNormalizationLUTs = new Map();
        for (const layer of videoLayers) {
          const firstFrame = await extractVideoFrame(layer.videoElement, 0);

          // Analyze first frame to get its range
          const tempCanvas = document.createElement('canvas');
          tempCanvas.width = firstFrame.width;
          tempCanvas.height = firstFrame.height;
          const tempCtx = tempCanvas.getContext('2d');
          tempCtx.drawImage(firstFrame, 0, 0);
          const imageData = tempCtx.getImageData(0, 0, firstFrame.width, firstFrame.height);

          // Get range with 80% scaling (leaves 20% headroom for brighter/closer objects in later frames)
          const range = getImageRange(imageData.data, 0.8);
          const lut = makeLUTFromRange(range.lo, range.hi);
          videoNormalizationLUTs.set(layer.id, lut);

          console.log(`Video layer ${layer.id}: Consistent normalization computed (lo=${range.lo}, hi=${range.hi.toFixed(1)})`);
        }

        // Store pre-rendered frames
        const renderedFrames = [];

        // Render each stereogram frame
        for (let frameIndex = 0; frameIndex < totalFrames; frameIndex++) {
          // Check if user requested stop
          if (shouldStopRendering) {
            console.log('Video rendering stopped by user at frame', frameIndex);
            break;
          }

          const time = frameIndex / fps;

          // Update progress
          const progress = ((frameIndex + 1) / totalFrames) * 50; // First 50%
          videoProgressBar.style.width = progress + '%';
          videoProgressText.textContent = `Phase 1: Rendering ${frameIndex + 1}/${totalFrames}`;

          // Update all video layers to the current time
          for (const layer of videoLayers) {
            // Loop shorter videos
            const layerTime = time % layer.duration;

            // Extract frame at the current time
            const frame = await extractVideoFrame(layer.videoElement, layerTime);

            // Update originalImage so Depth Anything V2 processes the current frame
            layer.originalImage = frame;

            // CRITICAL: Clear depth cache so each frame gets fresh depth computation
            depthCache.delete(layer.id);

            // Use consistent normalization LUT for this video layer
            const normalizationLUT = videoNormalizationLUTs.get(layer.id);
            layer.image = processImageToGrayscale(frame, normalizationLUT);

            // Wait for image to load
            await new Promise(resolve => {
              if (layer.image.complete) {
                resolve();
              } else {
                layer.image.onload = resolve;
              }
            });
          }

          // Generate the autostereogram for this frame
          await generateAutostereogram();

          // Store the rendered frame
          const frameData = ctx.getImageData(0, 0, previewCanvas.width, previewCanvas.height);
          renderedFrames.push(frameData);
        }

        if (renderedFrames.length === 0) {
          throw new Error('No frames were rendered');
        }

        // PHASE 2: Export frames as PNG files in ZIP (fast - just encoding)
        videoProgressText.textContent = 'Phase 2: Creating PNG frames...';

        const zip = new JSZip();
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = previewCanvas.width;
        tempCanvas.height = previewCanvas.height;
        const tempCtx = tempCanvas.getContext('2d');

        for (let i = 0; i < renderedFrames.length; i++) {
          const progress = 50 + ((i + 1) / renderedFrames.length) * 50; // Second 50%
          videoProgressBar.style.width = progress + '%';
          videoProgressText.textContent = `Phase 2: Exporting frame ${i + 1}/${renderedFrames.length}`;

          // Draw frame to temp canvas
          tempCtx.putImageData(renderedFrames[i], 0, 0);

          // Convert to PNG blob
          const blob = await new Promise(resolve => tempCanvas.toBlob(resolve, 'image/png'));

          // Add to ZIP with zero-padded filename
          const frameNumber = String(i).padStart(5, '0');
          zip.file(`frame_${frameNumber}.png`, blob);
        }

        // Generate ZIP file
        videoProgressText.textContent = 'Generating ZIP file...';
        const zipBlob = await zip.generateAsync({ type: 'blob' });

        // Download ZIP
        const a = document.createElement('a');
        a.href = URL.createObjectURL(zipBlob);
        a.download = 'apoculus-stereogram-frames.zip';
        a.click();
        URL.revokeObjectURL(a.href);

        // Reset video layers to first frame
        for (const layer of videoLayers) {
          const firstFrame = await extractVideoFrame(layer.videoElement, 0);
          const normalizationLUT = videoNormalizationLUTs.get(layer.id);
          layer.image = processImageToGrayscale(firstFrame, normalizationLUT);
          await new Promise(resolve => {
            if (layer.image.complete) {
              resolve();
            } else {
              layer.image.onload = resolve;
            }
          });
        }

        // Regenerate preview at first frame
        await generateAutostereogram();

        // Hide progress and re-enable button
        videoProgress.style.display = 'none';
        downloadVideoBtn.disabled = false;
        shouldStopRendering = false;
      } catch (error) {
        console.error('Error rendering video:', error);
        alert('Error rendering video: ' + error.message);
        videoProgress.style.display = 'none';
        downloadVideoBtn.disabled = false;
        shouldStopRendering = false;
      }
    }

    downloadVideoBtn.addEventListener('click', () => {
      renderVideo();
    });

    stopRenderingBtn.addEventListener('click', () => {
      shouldStopRendering = true;
      videoProgressText.textContent = 'Stopping...';
    });

// ============================================================================
// MODAL HANDLERS
// ============================================================================

    // License modal
    document.getElementById('licenseFooter').addEventListener('click', () => {
      document.getElementById('licenseModal').classList.add('show');
      document.getElementById('licenseOverlay').classList.add('show');
    });

    const closeLicense = () => {
      document.getElementById('licenseModal').classList.remove('show');
      document.getElementById('licenseOverlay').classList.remove('show');
    };

    document.getElementById('licenseClose').addEventListener('click', closeLicense);
    document.getElementById('licenseOverlay').addEventListener('click', closeLicense);

    // Help modal
    document.getElementById('imageDepthHelpLink').addEventListener('click', () => {
      document.getElementById('depthHelpModal').classList.add('show');
      document.getElementById('depthHelpOverlay').classList.add('show');
    });

    const closeDepthHelp = () => {
      document.getElementById('depthHelpModal').classList.remove('show');
      document.getElementById('depthHelpOverlay').classList.remove('show');
    };

    document.getElementById('depthHelpClose').addEventListener('click', closeDepthHelp);
    document.getElementById('depthHelpOverlay').addEventListener('click', closeDepthHelp);

    // General Help modal
    document.getElementById('generalHelpLink').addEventListener('click', () => {
      document.getElementById('generalHelpModal').classList.add('show');
      document.getElementById('generalHelpOverlay').classList.add('show');
    });

    const closeGeneralHelp = () => {
      document.getElementById('generalHelpModal').classList.remove('show');
      document.getElementById('generalHelpOverlay').classList.remove('show');
    };

    document.getElementById('generalHelpClose').addEventListener('click', closeGeneralHelp);
    document.getElementById('generalHelpOverlay').addEventListener('click', closeGeneralHelp);

// ============================================================================
// LAYER MANAGEMENT - CORE FUNCTIONS
// ============================================================================

    // Helper Functions
    function getActiveLayer() {
      return state.layers.find(l => l.id === state.activeLayerId);
    }

    function selectLayer(id) {
      state.activeLayerId = id;
      const layer = getActiveLayer();

      if (!layer) {
        layerSettings.style.display = 'none';
        return;
      }

      layerSettings.style.display = 'block';

      // Show/hide appropriate controls
      if (layer.type === 'text') {
        textContentGroup.style.display = 'block';
        fontGroup.style.display = 'block';
        depthAlgorithmGroup.style.display = 'none';
        retinexRadiusGroup.style.display = 'none';
        depthGammaGroup.style.display = 'none';
        invertDepthGroup.style.display = 'none';
        downloadDepthmapGroup.style.display = 'none';
        videoFrameGroup.style.display = 'none';
        layerText.value = layer.text;
        layerFont.value = layer.font;
      } else if (layer.type === 'video') {
        textContentGroup.style.display = 'none';
        fontGroup.style.display = 'none';
        depthAlgorithmGroup.style.display = 'block';
        depthGammaGroup.style.display = 'block';
        invertDepthGroup.style.display = 'block';
        downloadDepthmapGroup.style.display = 'block';
        videoFrameGroup.style.display = 'block';

        // Set up video frame controls
        const maxFrame = Math.ceil(layer.duration * layer.fps) - 1;
        videoFrameSlider.max = maxFrame;
        videoFrameInput.max = maxFrame;
        videoFrameSlider.value = layer.currentFrame || 0;
        videoFrameInput.value = layer.currentFrame || 0;
        const time = (layer.currentFrame || 0) / layer.fps;
        videoFrameInfo.textContent = `Frame ${layer.currentFrame || 0} of ${maxFrame} (${time.toFixed(2)}s)`;

        layerDepthAlgorithm.value = layer.depthAlgorithm || 'direct';
        layerRetinexRadius.value = layer.retinexRadius || 32;
        document.getElementById('layerRetinexRadiusValue').textContent = (layer.retinexRadius || 32) + 'px';

        // Show/hide retinex radius based on algorithm
        retinexRadiusGroup.style.display = (layer.depthAlgorithm === 'retinex') ? 'block' : 'none';

        layerDepthGamma.value = layer.depthGamma || 1.0;
        document.getElementById('layerDepthGammaValue').textContent = (layer.depthGamma || 1.0).toFixed(2);
        invertDepth.checked = layer.invertDepth || false;

        // Show/hide download depthmap button based on algorithm
        if (layer.depthAlgorithm === 'depth-anything-v2') {
          downloadDepthmapGroup.style.display = 'block';
          const hasCachedDepth = depthCache.has(layer.id) && depthCache.get(layer.id).algorithm === 'depth-anything-v2';
          downloadDepthmapBtn.disabled = !hasCachedDepth;
        } else {
          downloadDepthmapGroup.style.display = 'none';
        }
      } else {
        // Image layer
        textContentGroup.style.display = 'none';
        fontGroup.style.display = 'none';
        videoFrameGroup.style.display = 'none';
        depthAlgorithmGroup.style.display = 'block';
        depthGammaGroup.style.display = 'block';
        invertDepthGroup.style.display = 'block';
        downloadDepthmapGroup.style.display = 'block';

        layerDepthAlgorithm.value = layer.depthAlgorithm || 'direct';
        layerRetinexRadius.value = layer.retinexRadius || 32;
        document.getElementById('layerRetinexRadiusValue').textContent = (layer.retinexRadius || 32) + 'px';

        // Show/hide retinex radius based on algorithm
        retinexRadiusGroup.style.display = (layer.depthAlgorithm === 'retinex') ? 'block' : 'none';

        layerDepthGamma.value = layer.depthGamma || 1.0;
        document.getElementById('layerDepthGammaValue').textContent = (layer.depthGamma || 1.0).toFixed(2);
        invertDepth.checked = layer.invertDepth || false;

        // Show/hide download depthmap button based on algorithm
        if (layer.depthAlgorithm === 'depth-anything-v2') {
          downloadDepthmapGroup.style.display = 'block';
          // Enable/disable button based on whether depth is cached
          const hasCachedDepth = depthCache.has(layer.id) && depthCache.get(layer.id).algorithm === 'depth-anything-v2';
          downloadDepthmapBtn.disabled = !hasCachedDepth;
        } else {
          downloadDepthmapGroup.style.display = 'none';
        }
      }

      // Update controls
      layerSize.value = layer.size;
      layerDepth.value = layer.depth;
      layerRotation.value = layer.rotation;
      layerX.value = layer.x;
      layerY.value = layer.y;

      document.getElementById('layerSizeValue').textContent = layer.size;
      document.getElementById('layerDepthValue').textContent = layer.depth;
      document.getElementById('layerRotationValue').textContent = layer.rotation + '°';
      document.getElementById('layerXValue').textContent = layer.x;
      document.getElementById('layerYValue').textContent = layer.y;

      updateLayerList();
    }

    function deleteLayer(id) {
      const index = state.layers.findIndex(l => l.id === id);
      if (index !== -1) {
        state.layers.splice(index, 1);
        // Invalidate depth cache for deleted layer
        depthCache.delete(id);
        if (state.activeLayerId === id) {
          state.activeLayerId = state.layers.length > 0 ? state.layers[0].id : null;
        }
        updateLayerList();
        if (state.activeLayerId) {
          selectLayer(state.activeLayerId);
        } else {
          layerSettings.style.display = 'none';
        }
        generateAutostereogram();
      }
    }

    function updateLayerList() {
      if (state.layers.length === 0) {
        layerList.innerHTML = `
          <div class="help-text" style="font-size: 0.85rem; padding: 15px;">
            No layers yet. Add text, image, or video to begin.
          </div>
        `;
        return;
      }

      layerList.innerHTML = state.layers.map(layer => {
        const isActive = layer.id === state.activeLayerId;
        let name;
        if (layer.type === 'text') {
          name = layer.text.substring(0, 20) + (layer.text.length > 20 ? '...' : '');
        } else if (layer.type === 'video') {
          name = '🎬 Video';
        } else {
          name = 'Image';
        }

        let details = `Depth: ${layer.depth} | Size: ${layer.size}`;
        if (layer.type === 'video') {
          details += ` | ${layer.duration.toFixed(1)}s`;
        }

        return `
          <div class="layer-item ${isActive ? 'active' : ''}" data-id="${layer.id}">
            <div class="layer-info">
              <div class="layer-name">${name}</div>
              <div class="layer-details">${details}</div>
            </div>
            <button class="layer-delete" data-id="${layer.id}">Delete</button>
          </div>
        `;
      }).join('');

      // Add event listeners
      layerList.querySelectorAll('.layer-item').forEach(item => {
        item.addEventListener('click', (e) => {
          if (!e.target.classList.contains('layer-delete')) {
            selectLayer(parseInt(item.dataset.id));
          }
        });
      });

      layerList.querySelectorAll('.layer-delete').forEach(btn => {
        btn.addEventListener('click', (e) => {
          e.stopPropagation();
          deleteLayer(parseInt(btn.dataset.id));
        });
      });
    }

// ============================================================================
// DEPTH MAP GENERATION
// ============================================================================

    /**
     * Generate depth map for all layers based on their type and algorithm settings.
     * Returns a composite ImageData where pixel brightness = depth (0=far, 255=near).
     * Handles text rendering, image processing, and AI depth prediction.
     */
    async function generateDepthMap() {
      const width = state.settings.outputWidth;
      const height = state.settings.outputHeight;

      // Create temporary canvas for depth map
      const depthCanvas = document.createElement('canvas');
      depthCanvas.width = width;
      depthCanvas.height = height;
      const depthCtx = depthCanvas.getContext('2d');

      // Fill with background depth (0 = farthest)
      depthCtx.fillStyle = '#000000';
      depthCtx.fillRect(0, 0, width, height);

      // Render each layer sorted by depth (farthest first)
      const sortedLayers = [...state.layers].sort((a, b) => a.depth - b.depth);

      for (const layer of sortedLayers) {
        depthCtx.save();
        depthCtx.translate(layer.x, layer.y);
        depthCtx.rotate((layer.rotation * Math.PI) / 180);

        // Calculate depth color (0-255 based on depth 0-100)
        const depthColor = Math.floor((layer.depth / 100) * 255);
        depthCtx.fillStyle = `rgb(${depthColor}, ${depthColor}, ${depthColor})`;

        if (layer.type === 'text') {
          depthCtx.font = `bold ${layer.size}px ${layer.font}`;
          depthCtx.textAlign = 'center';
          depthCtx.textBaseline = 'middle';
          //depthCtx.fillText(layer.text, 0, 0);

          // Allow literal "\n" in a single-line input OR real newlines from a textarea
          const text = (layer.text || '').replace(/\\n/g, '\n');
          const lines = text.split(/\r?\n/);
        
          const lineHeight = 1.1 * layer.size;
          for (let i = 0; i < lines.length; i++) {
            const t = lines[i];
            const dy = i * lineHeight;
            depthCtx.fillText(t, 0, dy);
          }
        } else if (layer.type === 'image' || layer.type === 'video') {
          const aspectRatio = layer.image.width / layer.image.height;
          // Scale images 4x larger for better visibility
          const scaledSize = layer.size * 4;
          const drawWidth = scaledSize * aspectRatio;
          const drawHeight = scaledSize;

          // Draw image to temp canvas
          const tempCanvas = document.createElement('canvas');
          tempCanvas.width = drawWidth;
          tempCanvas.height = drawHeight;
          const tempCtx = tempCanvas.getContext('2d');
          tempCtx.drawImage(layer.image, 0, 0, drawWidth, drawHeight);

          // Get image data for processing
          let imgData = tempCtx.getImageData(0, 0, drawWidth, drawHeight);

          // Apply selected depth algorithm with caching
          if (layer.depthAlgorithm === 'depth-anything-v2') {
            // Check cache first
            const cacheEntry = depthCache.get(layer.id);
            let rawDepth;

            if (cacheEntry && cacheEntry.algorithm === 'depth-anything-v2') {
              // Cache hit! Use cached raw depth (no AI inference needed)
              rawDepth = cacheEntry.rawDepth;
              console.log(`Using cached depth for layer ${layer.id}`);
            } else {
              // Cache miss or algorithm changed - run AI model
              try {
                console.log(`Running AI depth prediction for layer ${layer.id}`);
                rawDepth = await applyDepthAnythingV2Raw(layer.originalImage);
                // Cache the raw depth for future use
                depthCache.set(layer.id, {
                  rawDepth: rawDepth,
                  algorithm: 'depth-anything-v2'
                });

                // Enable download button if this is the active layer
                const activeLayer = getActiveLayer();
                if (activeLayer && activeLayer.id === layer.id) {
                  downloadDepthmapBtn.disabled = false;
                }
              } catch (error) {
                console.error('Depth Anything V2 failed, falling back to grayscale:', error);
                // Fallback to grayscale if model fails - don't cache failures
              }
            }

            if (rawDepth) {
              // Apply LUT transformation (fast, parameter-dependent)
              const nearDepth = (layer.depth / 100) * 255;
              const farDepth = 0;
              // Depth Anything V2 produces inverted depth maps (far=bright, near=dark)
              // So we automatically invert it by flipping the invertDepth setting
              const effectiveInvert = !layer.invertDepth;
              imgData = applyDepthLUT(rawDepth, nearDepth, farDepth, effectiveInvert, layer.depthGamma);

              // Resize to target dimensions
              const resizeCanvas = document.createElement('canvas');
              resizeCanvas.width = drawWidth;
              resizeCanvas.height = drawHeight;
              const resizeCtx = resizeCanvas.getContext('2d');
              const tempImgCanvas = document.createElement('canvas');
              tempImgCanvas.width = imgData.width;
              tempImgCanvas.height = imgData.height;
              const tempImgCtx = tempImgCanvas.getContext('2d');
              tempImgCtx.putImageData(imgData, 0, 0);
              resizeCtx.drawImage(tempImgCanvas, 0, 0, drawWidth, drawHeight);
              imgData = resizeCtx.getImageData(0, 0, drawWidth, drawHeight);
            }
          } else if (layer.depthAlgorithm === 'retinex') {
            // Apply shadow-aware processing using Retinex-inspired approach
            // Scale radius proportionally to image size
            const scaleFactor = Math.sqrt((drawWidth * drawHeight) / (1200 * 800));
            const scaledRadius = Math.round(layer.retinexRadius * scaleFactor);
            imgData = applyShadowAwareDepth(imgData, scaledRadius);
          }
          // For 'direct' algorithm, image is already processed (grayscale with auto-levels)

          // Apply depth LUT (depth-anything-v2 already applied LUT in its cache path)
          if (layer.depthAlgorithm !== 'depth-anything-v2') {
            // Create depth LUT for this layer
            // nearDepth = layer depth (white pixels), farDepth = 0 (black pixels)
            const nearDepth = (layer.depth / 100) * 255;
            const farDepth = 0;
            const depthLUT = makeDepthLUT(nearDepth, farDepth, layer.invertDepth, layer.depthGamma);

            // Map grayscale values to depth using LUT
            // Transparent pixels always map to depth 0 (background), regardless of invert
            for (let i = 0; i < imgData.data.length; i += 4) {
              const alpha = imgData.data[i + 3];

              let depth;
              if (alpha < 8) {
                // Transparent pixels always go to background depth
                depth = 0;
              } else {
                // Image is grayscale, so R=G=B
                const gray = imgData.data[i];
                depth = depthLUT[gray];
              }

              imgData.data[i] = depth;
              imgData.data[i + 1] = depth;
              imgData.data[i + 2] = depth;
              imgData.data[i + 3] = 255; // Full opacity in depth map
            }
          }
          tempCtx.putImageData(imgData, 0, 0);

          depthCtx.drawImage(tempCanvas, -drawWidth / 2, -drawHeight / 2);
        }

        depthCtx.restore();
      }

      // Apply blur for soft edges (reduce sparkle)
      //depthCtx.filter = 'blur(2px)';
      //depthCtx.drawImage(depthCanvas, 0, 0);
      //depthCtx.filter = 'none';

      // Get depth map data
      const imageData = depthCtx.getImageData(0, 0, width, height);
      const depthMap = new Uint8Array(width * height);

      for (let i = 0; i < depthMap.length; i++) {
        depthMap[i] = imageData.data[i * 4]; // Use red channel
      }

      return depthMap;
    }

// ============================================================================
// PATTERN GENERATION
// ============================================================================

    // Simple hash function for pseudo-random per-pixel values
    function hashPixel(x, y, seed) {
      let h = seed + x * 374761393 + y * 668265263;
      h = (h ^ (h >>> 13)) * 1274126177;
      return (h ^ (h >>> 16)) >>> 0;
    }

    // Convert hash to [0,1) float
    function hashToFloat(hash) {
      return (hash & 0xFFFFFF) / 0x1000000;
    }

    // Generate Random Dot Pattern Strip
    function generatePatternStrip(width, height, density, colorScheme) {
      const stripCanvas = document.createElement('canvas');
      stripCanvas.width = width;
      stripCanvas.height = height;
      const stripCtx = stripCanvas.getContext('2d');

      const imageData = stripCtx.createImageData(width, height);
      const data = imageData.data;

      // Random seed for this generation
      const seed = Math.floor(Math.random() * 0xFFFFFFFF);

      for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
          const i = (y * width + x) * 4;

//          // Use hash-based random for better distribution
//          const hash1 = hashPixel(x, y, seed);
//          const hash2 = hashPixel(x, y, seed + 1);
//          const hash3 = hashPixel(x, y, seed + 2);
//
//          const rand1 = hashToFloat(hash1);
//          const rand2 = hashToFloat(hash2);
//          const rand3 = hashToFloat(hash3);
//
//          // Determine if this pixel is "on" (higher density = more dots)
//          const threshold = density / 10;
//          const isOn = rand1 < threshold;

          // Use hash-based random for color variation (keep these)
          const hash2 = hashPixel(x, y, seed + 1);
          const hash3 = hashPixel(x, y, seed + 2);
          
          const rand2 = hashToFloat(hash2);
          const rand3 = hashToFloat(hash3);
          
          // Determine if this pixel is "on" (higher density = more dots)
          const threshold = Math.min(1, Math.max(0, density / 10)); // clamp 0..1
          
          let isOn;
          if (BlueNoise.ready) {
            // Sample tileable blue-noise (red channel 0..255)
            const bx = x % BlueNoise.w;
            const by = y % BlueNoise.h;
            const bi = (by * BlueNoise.w + bx) * 4;
            const val = BlueNoise.data[bi]; // 0..255
            isOn = val < threshold * 255;
          } else {
            // Fallback: original white-noise hash
            const hash1 = hashPixel(x, y, seed);
            const rand1 = hashToFloat(hash1);
            isOn = rand1 < threshold;
          }


          let r, g, b;

          if (colorScheme === 'grayscale') {
            const brightness = isOn ? Math.floor(rand2 * 255) : 0;
            r = g = b = brightness;
          } else if (colorScheme === 'color') {
            r = isOn ? Math.floor(rand1 * 255) : 0;
            g = isOn ? Math.floor(rand2 * 255) : 0;
            b = isOn ? Math.floor(rand3 * 255) : 0;
          } else if (colorScheme === 'cyan') {
            const brightness = isOn ? Math.floor(rand2 * 255) : 0;
            r = 0;
            g = brightness;
            b = brightness;
          } else if (colorScheme === 'magenta') {
            const brightness = isOn ? Math.floor(rand2 * 255) : 0;
            r = brightness;
            g = 0;
            b = brightness;
          } else if (colorScheme === 'green') {
            const brightness = isOn ? Math.floor(rand2 * 255) : 0;
            r = 0;
            g = brightness;
            b = 0;
          } else if (colorScheme === 'amber') {
            const brightness = isOn ? Math.floor(rand2 * 255) : 0;
            r = brightness;
            g = Math.floor(brightness * 0.75);
            b = 0;
          } else if (colorScheme === 'rainbow') {
            if (isOn) {
              const hue = rand2 * 360;
              const sat = 0.8;
              const light = 0.5;
              const rgb = hslToRgb(hue, sat, light);
              r = rgb[0];
              g = rgb[1];
              b = rgb[2];
            } else {
              r = g = b = 0;
            }
          } else if (colorScheme === 'neon') {
            if (isOn) {
              const choice = Math.floor(rand2 * 5);
              if (choice === 0) { r = 255; g = 0; b = 255; } // Magenta
              else if (choice === 1) { r = 0; g = 255; b = 255; } // Cyan
              else if (choice === 2) { r = 255; g = 255; b = 0; } // Yellow
              else if (choice === 3) { r = 0; g = 255; b = 0; } // Green
              else { r = 255; g = 0; b = 128; } // Hot pink
            } else {
              r = g = b = 0;
            }
          } else if (colorScheme === 'fire') {
            if (isOn) {
              const intensity = rand2;
              r = 255;
              g = Math.floor(intensity * 165); // 0-165
              b = Math.floor(intensity * intensity * 100); // 0-100
            } else {
              r = g = b = 0;
            }
          } else if (colorScheme === 'ice') {
            if (isOn) {
              const intensity = rand2;
              r = Math.floor(intensity * 200 + 55); // 55-255
              g = Math.floor(intensity * 255); // 0-255
              b = 255;
            } else {
              r = g = b = 0;
            }
          } else if (colorScheme === 'forest') {
            if (isOn) {
              const intensity = rand2;
              r = Math.floor(intensity * 100); // 0-100
              g = Math.floor(intensity * 180 + 75); // 75-255
              b = Math.floor(intensity * 80); // 0-80
            } else {
              r = g = b = 0;
            }
          } else if (colorScheme === 'sunset') {
            if (isOn) {
              const choice = rand2;
              if (choice < 0.33) {
                r = 255; g = Math.floor(rand3 * 100); b = Math.floor(rand3 * 150 + 100); // Purple
              } else if (choice < 0.66) {
                r = 255; g = Math.floor(rand3 * 150 + 100); b = Math.floor(rand3 * 100); // Orange
              } else {
                r = 255; g = Math.floor(rand3 * 100 + 100); b = Math.floor(rand3 * 180 + 75); // Pink
              }
            } else {
              r = g = b = 0;
            }
          } else if (colorScheme === 'ocean') {
            if (isOn) {
              const depth = rand2;
              r = Math.floor(depth * 100); // 0-100
              g = Math.floor(depth * 200 + 55); // 55-255
              b = Math.floor(depth * 100 + 155); // 155-255
            } else {
              r = g = b = 0;
            }
          } else if (colorScheme === 'matrix') {
            if (isOn) {
              const brightness = Math.floor(rand2 * 200 + 55); // 55-255
              r = 0;
              g = brightness;
              b = Math.floor(brightness * 0.3); // Slight blue tint
            } else {
              r = g = b = 0;
            }
          } else if (colorScheme === 'vaporwave') {
            if (isOn) {
              const choice = rand2 < 0.5;
              if (choice) {
                r = 255; g = Math.floor(rand3 * 100 + 100); b = 255; // Pink
              } else {
                r = Math.floor(rand3 * 100); g = Math.floor(rand3 * 200 + 55); b = 255; // Cyan
              }
            } else {
              r = g = b = 0;
            }
          } else if (colorScheme === 'synthwave') {
            if (isOn) {
              const choice = Math.floor(rand2 * 3);
              if (choice === 0) {
                r = Math.floor(rand3 * 100 + 155); g = 0; b = 255; // Purple
              } else if (choice === 1) {
                r = 255; g = Math.floor(rand3 * 100); b = Math.floor(rand3 * 180 + 75); // Pink
              } else {
                r = 0; g = Math.floor(rand3 * 200 + 55); b = 255; // Cyan
              }
            } else {
              r = g = b = 0;
            }
          } else {
            r = g = b = 0;
          }

          data[i] = r;
          data[i + 1] = g;
          data[i + 2] = b;
          data[i + 3] = 255;
        }
      }

      stripCtx.putImageData(imageData, 0, 0);
      return stripCanvas;
    }

    // HSL to RGB conversion for rainbow mode
    function hslToRgb(h, s, l) {
      h = h / 360;
      const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
      const p = 2 * l - q;

      const hueToRgb = (p, q, t) => {
        if (t < 0) t += 1;
        if (t > 1) t -= 1;
        if (t < 1/6) return p + (q - p) * 6 * t;
        if (t < 1/2) return q;
        if (t < 2/3) return p + (q - p) * (2/3 - t) * 6;
        return p;
      };

      return [
        Math.floor(hueToRgb(p, q, h + 1/3) * 255),
        Math.floor(hueToRgb(p, q, h) * 255),
        Math.floor(hueToRgb(p, q, h - 1/3) * 255)
      ];
    }

// ============================================================================
// AUTOSTEREOGRAM GENERATION - MAIN ALGORITHM
// ============================================================================

    /**
     * Generate autostereogram using scanline processing with union-find algorithm.
     *
     * Algorithm overview:
     * 1. Generate depth map from all layers (0=far background, 255=near foreground)
     * 2. Create random pattern strip (width = stripWidth px)
     * 3. For each scanline:
     *    - Calculate disparity for each pixel based on depth
     *    - Use union-find to group pixels that should share the same color
     *    - Handle occlusion (nearer depths take priority)
     *    - Assign colors from pattern strip to pixel groups
     * 4. Apply row-wise phase shifts to break stratification
     *
     * Result: A 2D image encoding 3D depth information via repeating patterns.
     */
    async function generateAutostereogram() {
      if (state.layers.length === 0) {
        ctx.fillStyle = '#000';
        ctx.fillRect(0, 0, previewCanvas.width, previewCanvas.height);
        return;
      }

      const width  = state.settings.outputWidth;
      const height = state.settings.outputHeight;
      const stripW = state.settings.stripWidth;
      const MIN_SEP = 8; // don't let separation collapse; tweak 8–12 if needed

      // Small constants/utilities for symmetry & visibility
      const OCC_THRESH = 2; // tolerate tiny depth noise (0–255 scale)

      // Reflection padding index (periodic mirror): 0..W-1, W..2W-2 -> mirror back, etc.
      function mirrorIndex(i, w) {
        const period = 2 * w - 2;
        if (period <= 0) return 0;
        let m = ((i % period) + period) % period;
        return m < w ? m : period - m;
      }


      // Clamp disparity so separation ∈ [MIN_SEP, stripW]
      const rawMaxDisp = Math.floor(stripW * state.settings.depthScale);
      const maxDisp    = Math.min(rawMaxDisp, stripW - MIN_SEP);

      // Depth map (0..255, where brighter = nearer)
      const depthMap = await generateDepthMap();
    
      // Base pattern strip (width = stripW, tiled across each row)
      const patternStrip = generatePatternStrip(
        stripW,
        height,
        state.settings.patternDensity,
        state.settings.colorScheme
      );
      const stripData = patternStrip.getContext('2d').getImageData(0, 0, stripW, height);
    
      // Output
      const outputData = ctx.createImageData(width, height);
      const out = outputData.data;
    
      // Small union-find helper with path compression
      function findRoot(parent, i) {
        while (parent[i] !== i) {
          parent[i] = parent[parent[i]];
          i = parent[i];
        }
        return i;
      }
    
      // Row-wise phase offsets to break horizontal stratification
      const rowShift = new Uint16Array(height);
      const shiftSeed = (Math.random() * 0xFFFFFFFF) >>> 0; // new every generation
      for (let y = 0; y < height; y++) {
        rowShift[y] = hash32(shiftSeed ^ y) % stripW;
      }

      for (let y = 0; y < height; y++) {
        const row = y * width;
        const parent = new Int32Array(width);
        for (let x = 0; x < width; x++) parent[x] = x;
    
        // 1) Link pixel pairs for this scanline
        for (let x = 0; x < width; x++) {
          const z = depthMap[row + x] / 255;                // 0..1
          const disp = Math.round(z * maxDisp);
          const sep  = Math.max(MIN_SEP, stripW - disp);    // safe separation
    
          //const left  = x - Math.floor(sep / 2);
          // centered pairing around x
          const dither = (sep & 1) && (y & 1) ? 1 : 0;   // alternate the half-pixel
          const left   = Math.floor(x - (sep + dither)/2);

          const right = left + sep;
    
          if (left >= 0 && right < width) {
            // Prefer the nearer surface at this pair position
            const dl = depthMap[row + left];
            const dr = depthMap[row + right];
    
            let L = findRoot(parent, left);
            let R = findRoot(parent, right);
//            if (L !== R) {
//              if (dl >= dr) parent[R] = L; else parent[L] = R;
//            }
            if (L !== R) {
              if (dl > dr) {
                parent[R] = L;
              } else if (dr > dl) {
                parent[L] = R;
              } else {
                // equal depth: alternate by checkerboard to avoid directional bias
                if (((x ^ y) & 1) === 0) parent[R] = L; else parent[L] = R;
              }
            }

          }
        }
    
        // 2a) Compute a centered phase per connected component (removes left-anchoring)
        const comp = new Map(); // root -> {min,max}
        for (let x = 0; x < width; x++) {
          const r = findRoot(parent, x);
          const o = comp.get(r);
          if (o) {
            if (x < o.min) o.min = x;
            if (x > o.max) o.max = x;
          } else {
            comp.set(r, { min: x, max: x });
          }
        }
        const phaseByRoot = new Map();
        for (const [r, o] of comp) {
          const center = Math.round((o.min + o.max) / 2) % stripW;
          phaseByRoot.set(r, center);
        }

        // 2) Paint: one color per set, sampled by the set root’s phase
        const colorByRoot = new Map();
        for (let x = 0; x < width; x++) {
          const root = findRoot(parent, x);
          let color = colorByRoot.get(root);
          if (!color) {
//            const stripX = root % stripW;                   // IMPORTANT: use root, not x
          //  const stripX = phaseByRoot.get(root);
            const stripX = (phaseByRoot.get(root) + rowShift[y]) % stripW;

            const si = (y * stripW + stripX) * 4;
            color = [
              stripData.data[si],
              stripData.data[si + 1],
              stripData.data[si + 2],
              255
            ];
            colorByRoot.set(root, color);
          }
    
          const oi = (row + x) * 4;
          out[oi]     = color[0];
          out[oi + 1] = color[1];
          out[oi + 2] = color[2];
          out[oi + 3] = 255;
        }
      }
    
      ctx.putImageData(outputData, 0, 0);
    }


    // Initialize
    generateAutostereogram();

    // Preload Depth Anything V2 model in background (if available)
    // This improves UX by having the model ready when user adds first image
    setTimeout(() => {
      loadDepthModel().catch(err => {
        console.log('Model preload skipped (model file not found - see models/README.md)');
      });
    }, 1000); // Delay 1 second to let UI render first