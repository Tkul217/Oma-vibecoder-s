import React, { useState } from 'react';
import { Upload, Car, CheckCircle, AlertCircle, Loader2 } from 'lucide-react';

interface AnalysisResult {
  cleanliness: 'clean' | 'dirty';
  cleanlinessConfidence: number;
  condition: 'intact' | 'damaged';
  conditionConfidence: number;
  processingTime: number;
}

function App() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string>('');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string>('');

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      const url = URL.createObjectURL(file);
      setPreviewUrl(url);
      setResult(null);
      setError('');
    }
  };

  const handleDrop = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    const file = event.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
      setSelectedFile(file);
      const url = URL.createObjectURL(file);
      setPreviewUrl(url);
      setResult(null);
      setError('');
    }
  };

  const handleDragOver = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
  };

  const analyzeImage = async () => {
    if (!selectedFile) return;

    setIsAnalyzing(true);
    setError('');

    try {
      const formData = new FormData();
      formData.append('image', selectedFile);

      const response = await fetch('/api/analyze', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Analysis failed');
      }

      const analysisResult: AnalysisResult = await response.json();
      setResult(analysisResult);
    } catch (err) {
      setError('Failed to analyze image. Please try again.');
      console.error('Analysis error:', err);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const getStatusColor = (status: string) => {
    return status === 'clean' || status === 'intact' ? 'text-green-600' : 'text-red-600';
  };

  const getStatusBg = (status: string) => {
    return status === 'clean' || status === 'intact' ? 'bg-green-50 border-green-200' : 'bg-red-50 border-red-200';
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex items-center space-x-3">
            <Car className="h-8 w-8 text-blue-600" />
            <h1 className="text-2xl font-bold text-gray-900">Car Condition Analyzer</h1>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Upload Section */}
          <div className="space-y-6">
            <div className="bg-white rounded-lg shadow-sm p-6">
              <h2 className="text-xl font-semibold text-gray-900 mb-4">Upload Car Photo</h2>
              
              <div
                onDrop={handleDrop}
                onDragOver={handleDragOver}
                className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center hover:border-blue-500 transition-colors"
              >
                <Upload className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                <p className="text-gray-600 mb-2">Drag and drop your car photo here, or</p>
                <label className="inline-flex items-center px-4 py-2 bg-blue-600 text-white rounded-lg cursor-pointer hover:bg-blue-700 transition-colors">
                  <span>Choose File</span>
                  <input
                    type="file"
                    accept="image/*"
                    onChange={handleFileSelect}
                    className="hidden"
                  />
                </label>
              </div>

              {previewUrl && (
                <div className="mt-6">
                  <img
                    src={previewUrl}
                    alt="Car preview"
                    className="w-full h-64 object-cover rounded-lg border"
                  />
                  <button
                    onClick={analyzeImage}
                    disabled={isAnalyzing}
                    className="mt-4 w-full flex items-center justify-center px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-400 transition-colors"
                  >
                    {isAnalyzing ? (
                      <>
                        <Loader2 className="h-5 w-5 mr-2 animate-spin" />
                        Analyzing...
                      </>
                    ) : (
                      'Analyze Car Condition'
                    )}
                  </button>
                </div>
              )}

              {error && (
                <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-lg">
                  <p className="text-red-600">{error}</p>
                </div>
              )}
            </div>
          </div>

          {/* Results Section */}
          <div className="space-y-6">
            {result ? (
              <div className="bg-white rounded-lg shadow-sm p-6">
                <h2 className="text-xl font-semibold text-gray-900 mb-6">Analysis Results</h2>
                
                <div className="space-y-4">
                  {/* Cleanliness Result */}
                  <div className={`p-4 rounded-lg border ${getStatusBg(result.cleanliness)}`}>
                    <div className="flex items-center justify-between mb-2">
                      <h3 className="font-semibold text-gray-900">Cleanliness</h3>
                      {result.cleanliness === 'clean' ? (
                        <CheckCircle className="h-6 w-6 text-green-600" />
                      ) : (
                        <AlertCircle className="h-6 w-6 text-red-600" />
                      )}
                    </div>
                    <p className={`text-lg font-medium ${getStatusColor(result.cleanliness)}`}>
                      {result.cleanliness === 'clean' ? 'Clean' : 'Dirty'}
                    </p>
                    <p className="text-sm text-gray-600">
                      Confidence: {Math.round(result.cleanlinessConfidence * 100)}%
                    </p>
                  </div>

                  {/* Condition Result */}
                  <div className={`p-4 rounded-lg border ${getStatusBg(result.condition)}`}>
                    <div className="flex items-center justify-between mb-2">
                      <h3 className="font-semibold text-gray-900">Physical Condition</h3>
                      {result.condition === 'intact' ? (
                        <CheckCircle className="h-6 w-6 text-green-600" />
                      ) : (
                        <AlertCircle className="h-6 w-6 text-red-600" />
                      )}
                    </div>
                    <p className={`text-lg font-medium ${getStatusColor(result.condition)}`}>
                      {result.condition === 'intact' ? 'Intact' : 'Damaged'}
                    </p>
                    <p className="text-sm text-gray-600">
                      Confidence: {Math.round(result.conditionConfidence * 100)}%
                    </p>
                  </div>

                  {/* Processing Info */}
                  <div className="p-4 bg-gray-50 rounded-lg border">
                    <p className="text-sm text-gray-600">
                      Processing Time: {result.processingTime}ms
                    </p>
                  </div>
                </div>
              </div>
            ) : (
              <div className="bg-white rounded-lg shadow-sm p-6">
                <h2 className="text-xl font-semibold text-gray-900 mb-4">Analysis Results</h2>
                <p className="text-gray-500">Upload and analyze a car photo to see results here.</p>
              </div>
            )}

            {/* Information Section */}
            <div className="bg-white rounded-lg shadow-sm p-6">
              <h2 className="text-xl font-semibold text-gray-900 mb-4">How it Works</h2>
              <div className="space-y-3 text-sm text-gray-600">
                <p>• <strong>Cleanliness Detection:</strong> Analyzes dirt, dust, and overall vehicle cleanliness</p>
                <p>• <strong>Damage Assessment:</strong> Identifies scratches, dents, and structural damage</p>
                <p>• <strong>Confidence Scores:</strong> Provides reliability metrics for each prediction</p>
                <p>• <strong>Fast Processing:</strong> Results typically available in under 2 seconds</p>
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;