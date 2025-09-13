const express = require('express');
const multer = require('multer');
const cors = require('cors');
const path = require('path');
const fs = require('fs');
const { spawn } = require('child_process');

const app = express();
const PORT = 3001;

// Enable CORS
app.use(cors());

// Configure multer for file uploads
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    const uploadDir = path.join(__dirname, 'uploads');
    if (!fs.existsSync(uploadDir)) {
      fs.mkdirSync(uploadDir, { recursive: true });
    }
    cb(null, uploadDir);
  },
  filename: (req, file, cb) => {
    const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1E9);
    cb(null, 'car-' + uniqueSuffix + path.extname(file.originalname));
  }
});

const upload = multer({
  storage: storage,
  limits: {
    fileSize: 10 * 1024 * 1024 // 10MB limit
  },
  fileFilter: (req, file, cb) => {
    const allowedTypes = /jpeg|jpg|png|gif/;
    const extname = allowedTypes.test(path.extname(file.originalname).toLowerCase());
    const mimetype = allowedTypes.test(file.mimetype);

    if (mimetype && extname) {
      return cb(null, true);
    } else {
      cb(new Error('Only image files are allowed!'));
    }
  }
});

// Middleware
app.use(express.json());
app.use('/uploads', express.static(path.join(__dirname, 'uploads')));

// Mock ML Model Analysis Function
async function analyzeCarImage(imagePath) {
  // Используем реальную ML модель
  const pythonScript = path.join(__dirname, '../ml/analyze.py');
  
  return new Promise((resolve, reject) => {
    console.log(`Analyzing with ML model: ${pythonScript}`);
    const python = spawn('python3', [pythonScript, imagePath]);
    
    let result = '';
    let error = '';
    
    python.stdout.on('data', (data) => {
      result += data.toString();
    });
    
    python.stderr.on('data', (data) => {
      error += data.toString();
    });
    
    python.on('close', (code) => {
      if (code === 0) {
        try {
          const analysisResult = JSON.parse(result);
          console.log('ML analysis result:', analysisResult);
          resolve(analysisResult);
        } catch (e) {
          console.error('JSON parse error:', e);
          console.error('Raw result:', result);
          reject(new Error('Invalid JSON response from ML model'));
        }
      } else {
        console.error('Python ML error (code ' + code + '):', error);
        reject(new Error(`ML analysis failed: ${error}`));
      }
    });
  });
}

function getMockPrediction() {
  // Simulate processing time
  const processingTime = Math.round(Math.random() * 1000 + 800); // 800-1800ms
  
  // Mock predictions with random confidence scores
  return {
    cleanliness: Math.random() > 0.3 ? 'clean' : 'dirty', // 70% чистых
    cleanlinessConfidence: 0.7 + Math.random() * 0.3, // 70-100%
    condition: Math.random() > 0.4 ? 'intact' : 'damaged', // 60% целых
    conditionConfidence: 0.6 + Math.random() * 0.4, // 60-100%
    processingTime: processingTime
  };
}

// Routes
app.post('/api/analyze', upload.single('image'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No image file provided' });
    }

    const imagePath = req.file.path;
    console.log(`Processing image: ${imagePath}`);

    // Simulate processing delay
    await new Promise(resolve => setTimeout(resolve, 800));

    // Analyze the image (mock ML model)
    const analysisResult = await analyzeCarImage(imagePath);

    // Clean up uploaded file (optional, for demo purposes)
    setTimeout(() => {
      if (fs.existsSync(imagePath)) {
        fs.unlinkSync(imagePath);
      }
    }, 5000);

    res.json(analysisResult);

  } catch (error) {
    console.error('Analysis error:', error);
    res.status(500).json({ error: 'Internal server error during analysis' });
  }
});

// Health check endpoint
app.get('/api/health', (req, res) => {
  res.json({ status: 'OK', message: 'Car Condition Analyzer API is running' });
});

// Error handling middleware
app.use((error, req, res, next) => {
  if (error instanceof multer.MulterError) {
    if (error.code === 'LIMIT_FILE_SIZE') {
      return res.status(400).json({ error: 'File too large. Maximum size is 10MB.' });
    }
  }
  
  if (error.message === 'Only image files are allowed!') {
    return res.status(400).json({ error: 'Only image files are allowed!' });
  }

  console.error(error);
  res.status(500).json({ error: 'Internal server error' });
});

app.listen(PORT, () => {
  console.log(`Car Condition Analyzer API running on http://localhost:${PORT}`);
  console.log(`Health check: http://localhost:${PORT}/api/health`);
});