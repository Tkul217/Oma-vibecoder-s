import React, { useState } from 'react';
import { Upload, CheckCircle, AlertCircle, Loader2 } from 'lucide-react';

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
      setError('Не удалось проанализировать изображение. Попробуйте еще раз.');
      console.error('Analysis error:', err);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const getStatusColor = (status: string) => {
    return status === 'clean' || status === 'intact' ? 'text-green-600' : 'text-red-600';
  };

  const getStatusBg = (status: string) => {
    return status === 'clean' || status === 'intact' ? 'bg-emerald-50 border-emerald-200' : 'bg-red-50 border-red-200';
  };

  const getStatusText = (status: string, type: 'cleanliness' | 'condition') => {
    if (type === 'cleanliness') {
      return status === 'clean' ? 'Чистый' : 'Грязный';
    } else {
      return status === 'intact' ? 'Целый' : 'Поврежденный';
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-100">
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex items-center space-x-3">
              <svg width="147" height="40" viewBox="0 0 147 40" fill="none" xmlns="http://www.w3.org/2000/svg">
                  <path d="M125.52 14.2285L121.671 25.7751L117.886 14.2285H113.031L118.843 30.1809H124.313L130.193 14.2285H125.52Z" fill="#141414"/>
                  <path d="M98.5129 17.7523V14.2285H94.25V30.1809H98.809V23.4583C98.809 19.7896 101.086 18.5577 103.867 18.5577H105.258V14.2285H104.3C101.656 14.2285 99.3411 15.2389 98.5129 17.7523Z" fill="#141414"/>
                  <path d="M81.5631 10.0332H73.6562V30.1802H81.5631C87.8529 30.1802 92.2297 26.1471 92.207 20.085C92.1842 14.1139 87.8985 10.0332 81.5631 10.0332ZM81.5631 25.7827H78.3043V14.4328H81.5631C85.0496 14.4328 87.4202 16.7351 87.4202 20.085C87.4202 23.4349 85.0496 25.7827 81.5631 25.7827Z" fill="#141414"/>
                  <path d="M64.8578 13.9083C62.8785 13.929 60.8454 14.6806 59.5473 16.5792V14.2272H55.1953V30.1796H59.7523V22.1796C59.7523 19.2624 61.2347 17.8048 63.1477 17.8048C65.0607 17.8048 66.3154 19.0119 66.3154 21.9518V30.1796H70.8744V20.3804C70.8744 16.5978 68.6404 13.8856 64.8578 13.9083Z" fill="#141414"/>
                  <path d="M52.4184 14.2285H47.8594V30.1809H52.4184V14.2285Z" fill="#141414"/>
                  <path d="M52.4184 7.82422H47.8594V12.3356H52.4184V7.82422Z" fill="#141414"/>
                  <path d="M111.387 14.2285H106.828V30.1809H111.387V14.2285Z" fill="#141414"/>
                  <path d="M111.387 7.82422H106.828V12.3356H111.387V7.82422Z" fill="#141414"/>
                  <path d="M0 20C0 14.4513 0 11.6563 0.766046 9.44099C2.17391 5.38302 5.34162 2.19462 9.39959 0.766046C11.6356 0 14.3892 0 19.9172 0C25.4451 0 28.2195 0 30.4348 0.766046C34.4721 2.17391 37.6605 5.36232 39.0683 9.44099C39.8344 11.677 39.8344 14.4513 39.8344 20C39.8344 25.5487 39.8344 28.3437 39.0683 30.559C37.6605 34.617 34.4928 37.8054 30.4348 39.234C28.1988 40 25.4451 40 19.9172 40C14.3892 40 11.6149 40 9.39959 39.234C5.36232 37.8261 2.17391 34.6377 0.766046 30.559C0 28.3437 0 25.5694 0 20Z" fill="#C1F11D"/>
                  <path d="M15.8863 15.9141H11.0312V30.1811H15.8863V15.9141Z" fill="#141414"/>
                  <path d="M21.9453 9.09961H18.1172V13.8636H21.8315C25.2269 13.8636 27.5747 16.2797 27.5747 19.6069C27.5747 22.934 25.2269 25.4185 21.8315 25.4185H18.1172V30.1824H21.9453C27.9619 30.1824 32.4961 25.5779 32.4961 19.5862C32.4961 13.5944 27.9619 9.10375 21.9453 9.10375V9.09961Z" fill="#141414"/>
                  <path d="M15.8863 9.09961H11.0312V13.8636H15.8863V9.09961Z" fill="#141414"/>
                  <path d="M138.602 13.8633C133.521 13.8633 130.148 17.5776 130.148 22.2504C130.148 26.9233 133.681 30.5921 138.832 30.5465C141.885 30.5237 144.186 29.1801 145.987 26.8778L142.705 24.8032C141.817 25.942 140.449 26.6045 138.923 26.6045C137.132 26.6045 135.358 25.5672 134.772 23.6418H146.67C147.49 18.4678 143.912 13.8654 138.602 13.8654V13.8633ZM138.741 17.5548C140.714 17.5548 142.283 18.8674 142.48 20.5631H134.786C135.368 18.5175 137.03 17.5548 138.741 17.5548Z" fill="#141414"/>
              </svg>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Upload Section */}
          <div className="space-y-6">
            <div className="bg-white/80 backdrop-blur-sm rounded-xl shadow-lg border border-white/20 p-6">
              <h2 className="text-xl font-semibold text-gray-900 mb-4">Загрузить фото автомобиля</h2>
              
              <div
                onDrop={handleDrop}
                onDragOver={handleDragOver}
                className="border-2 border-dashed border-gray-300 rounded-xl p-8 text-center hover:border-emerald-500 hover:bg-emerald-50/50 transition-all duration-300"
              >
                <Upload className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                <p className="text-gray-600 mb-2">Перетащите фото автомобиля сюда или</p>
                <label className="inline-flex items-center px-6 py-3 bg-black text-white rounded-xl cursor-pointer hover:bg-gray-800 transition-all duration-200 font-medium">
                  <span>Выбрать файл</span>
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
                    className="w-full h-64 object-cover rounded-xl border shadow-md"
                  />
                  <button
                    onClick={analyzeImage}
                    disabled={isAnalyzing}
                    className="mt-4 w-full flex items-center justify-center px-6 py-3 bg-black text-white rounded-xl hover:bg-gray-800 disabled:bg-gray-400 transition-all duration-200 font-medium"
                  >
                    {isAnalyzing ? (
                      <>
                        <Loader2 className="h-5 w-5 mr-2 animate-spin" />
                        Анализируем...
                      </>
                    ) : (
                      'Анализировать состояние'
                    )}
                  </button>
                </div>
              )}

              {error && (
                <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-xl">
                  <p className="text-red-600">{error}</p>
                </div>
              )}
            </div>
          </div>

          {/* Results Section */}
          <div className="space-y-6">
            {result ? (
              <div className="bg-white/80 backdrop-blur-sm rounded-xl shadow-lg border border-white/20 p-6">
                <h2 className="text-xl font-semibold text-gray-900 mb-6">Результаты анализа</h2>
                
                <div className="space-y-4">
                  {/* Cleanliness Result */}
                  <div className={`p-4 rounded-xl border ${getStatusBg(result.cleanliness)}`}>
                    <div className="flex items-center justify-between mb-2">
                      <h3 className="font-semibold text-gray-900">Чистота</h3>
                      {result.cleanliness === 'clean' ? (
                        <CheckCircle className="h-6 w-6 text-emerald-600" />
                      ) : (
                        <AlertCircle className="h-6 w-6 text-red-500" />
                      )}
                    </div>
                    <p className={`text-lg font-medium ${getStatusColor(result.cleanliness)}`}>
                      {getStatusText(result.cleanliness, 'cleanliness')}
                    </p>
                    <p className="text-sm text-gray-600">
                      Уверенность: {Math.round(result.cleanlinessConfidence * 100)}%
                    </p>
                  </div>

                  {/* Condition Result */}
                  <div className={`p-4 rounded-xl border ${getStatusBg(result.condition)}`}>
                    <div className="flex items-center justify-between mb-2">
                      <h3 className="font-semibold text-gray-900">Физическое состояние</h3>
                      {result.condition === 'intact' ? (
                        <CheckCircle className="h-6 w-6 text-emerald-600" />
                      ) : (
                        <AlertCircle className="h-6 w-6 text-red-500" />
                      )}
                    </div>
                    <p className={`text-lg font-medium ${getStatusColor(result.condition)}`}>
                      {getStatusText(result.condition, 'condition')}
                    </p>
                    <p className="text-sm text-gray-600">
                      Уверенность: {Math.round(result.conditionConfidence * 100)}%
                    </p>
                  </div>

                  {/* Processing Info */}
                  <div className="p-4 bg-gray-50/80 rounded-xl border">
                    <p className="text-sm text-gray-600">
                      Время обработки: {result.processingTime}мс
                    </p>
                  </div>
                </div>
              </div>
            ) : (
              <div className="bg-white/80 backdrop-blur-sm rounded-xl shadow-lg border border-white/20 p-6">
                <h2 className="text-xl font-semibold text-gray-900 mb-4">Результаты анализа</h2>
                <p className="text-gray-500">Загрузите и проанализируйте фото автомобиля, чтобы увидеть результаты.</p>
              </div>
            )}

            {/* Information Section */}
            <div className="bg-white/80 backdrop-blur-sm rounded-xl shadow-lg border border-white/20 p-6">
              <h2 className="text-xl font-semibold text-gray-900 mb-4">Как это работает</h2>
              <div className="space-y-3 text-sm text-gray-600">
                <p>• <strong>Определение чистоты:</strong> Анализирует грязь, пыль и общую чистоту автомобиля</p>
                <p>• <strong>Оценка повреждений:</strong> Выявляет царапины, вмятины и структурные повреждения</p>
                <p>• <strong>Показатели уверенности:</strong> Предоставляет метрики надежности для каждого прогноза</p>
                <p>• <strong>Быстрая обработка:</strong> Результаты обычно доступны менее чем за 2 секунды</p>
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;