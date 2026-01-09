import { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Progress } from '@/components/ui/progress';
import { Upload, Download, Copy, CheckCircle } from 'lucide-react';
import { apiService, type TranscriptionResponse } from '@/services/api';

interface TranscriptionResult {
  filename: string;
  text: string;
  duration: number;
  language: string;
  timestamp: Date;
}

export function Transcribe() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [language, setLanguage] = useState('af');
  const [modelSize, setModelSize] = useState('small');
  const [isTranscribing, setIsTranscribing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [currentResult, setCurrentResult] = useState<TranscriptionResponse | null>(null);
  const [transcriptions, setTranscriptions] = useState<TranscriptionResult[]>([]);
  const [copied, setCopied] = useState(false);

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      setCurrentResult(null);
    }
  };

  const handleTranscribe = async () => {
    if (!selectedFile) return;

    setIsTranscribing(true);
    setProgress(10);
    setCurrentResult(null);

    try {
      // Simulate progress
      const progressInterval = setInterval(() => {
        setProgress((prev) => Math.min(prev + 10, 90));
      }, 1000);

      const result = await apiService.transcribeAudio({
        file: selectedFile,
        language,
        model_size: modelSize,
      });

      clearInterval(progressInterval);
      setProgress(100);

      setCurrentResult(result);

      if (result.success && result.text) {
        const newTranscription: TranscriptionResult = {
          filename: selectedFile.name,
          text: result.text,
          duration: result.duration || 0,
          language: result.language || language,
          timestamp: new Date(),
        };
        setTranscriptions([newTranscription, ...transcriptions]);
      }
    } catch (error) {
      console.error('Transcription error:', error);
      setCurrentResult({
        success: false,
        message: 'Transcription failed. Please try again.',
      });
    } finally {
      setIsTranscribing(false);
      setTimeout(() => setProgress(0), 1000);
    }
  };

  const handleCopyText = (text: string) => {
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleDownloadText = (text: string, filename: string) => {
    const blob = new Blob([text], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${filename}_transcription.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-gray-900 mb-2">Audio Transcription</h1>
        <p className="text-gray-600">Convert voice notes to text using Whisper AI</p>
      </div>

      {/* Upload and Settings Card */}
      <Card>
        <CardHeader>
          <CardTitle>Transcription Settings</CardTitle>
          <CardDescription>Upload an audio file and configure transcription options</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* File Upload */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Audio File
              </label>
              <div className="relative">
                <Input
                  type="file"
                  accept=".mp3,.wav,.opus,.ogg,.m4a,.aac"
                  onChange={handleFileSelect}
                  className="cursor-pointer"
                />
                {selectedFile && (
                  <p className="mt-2 text-sm text-gray-600">
                    Selected: {selectedFile.name} ({(selectedFile.size / 1024 / 1024).toFixed(2)} MB)
                  </p>
                )}
              </div>
            </div>

            {/* Language Selection */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Language
              </label>
              <select
                value={language}
                onChange={(e) => setLanguage(e.target.value)}
                className="w-full h-10 rounded-md border border-gray-300 bg-white px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-600"
              >
                <option value="af">Afrikaans</option>
                <option value="en">English</option>
                <option value="nl">Dutch</option>
              </select>
            </div>

            {/* Model Size */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Model Size (Speed vs Accuracy)
              </label>
              <select
                value={modelSize}
                onChange={(e) => setModelSize(e.target.value)}
                className="w-full h-10 rounded-md border border-gray-300 bg-white px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-600"
              >
                <option value="tiny">Tiny (Fastest)</option>
                <option value="small">Small (Balanced)</option>
                <option value="medium">Medium (Accurate)</option>
                <option value="large">Large (Most Accurate)</option>
              </select>
            </div>
          </div>

          {/* Transcribe Button */}
          <div className="pt-4">
            <Button
              onClick={handleTranscribe}
              disabled={!selectedFile || isTranscribing}
              className="w-full md:w-auto"
              size="lg"
            >
              {isTranscribing ? (
                <>Processing...</>
              ) : (
                <>
                  <Upload className="w-4 h-4 mr-2" />
                  Transcribe Audio
                </>
              )}
            </Button>
          </div>

          {/* Progress Bar */}
          {isTranscribing && (
            <div className="space-y-2">
              <Progress value={progress} />
              <p className="text-sm text-gray-600 text-center">
                Transcribing... {progress}%
              </p>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Current Result */}
      {currentResult && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              {currentResult.success ? (
                <>
                  <CheckCircle className="w-5 h-5 text-green-600" />
                  <span>Transcription Complete</span>
                </>
              ) : (
                <span className="text-red-600">Transcription Failed</span>
              )}
            </CardTitle>
          </CardHeader>
          <CardContent>
            {currentResult.success && currentResult.text ? (
              <div className="space-y-4">
                <div className="bg-gray-50 p-4 rounded-lg">
                  <p className="text-gray-900 whitespace-pre-wrap">{currentResult.text}</p>
                </div>
                <div className="flex space-x-2">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => handleCopyText(currentResult.text!)}
                  >
                    {copied ? (
                      <>
                        <CheckCircle className="w-4 h-4 mr-2" />
                        Copied!
                      </>
                    ) : (
                      <>
                        <Copy className="w-4 h-4 mr-2" />
                        Copy Text
                      </>
                    )}
                  </Button>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => handleDownloadText(currentResult.text!, selectedFile?.name || 'transcription')}
                  >
                    <Download className="w-4 h-4 mr-2" />
                    Download
                  </Button>
                </div>
                {currentResult.duration && (
                  <p className="text-sm text-gray-600">
                    Duration: {currentResult.duration.toFixed(2)}s
                  </p>
                )}
              </div>
            ) : (
              <p className="text-red-600">{currentResult.message}</p>
            )}
          </CardContent>
        </Card>
      )}

      {/* Recent Transcriptions */}
      {transcriptions.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Recent Transcriptions</CardTitle>
            <CardDescription>Your last {transcriptions.length} transcription(s)</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {transcriptions.slice(0, 5).map((transcription, index) => (
                <div key={index} className="border-b border-gray-200 pb-4 last:border-0">
                  <div className="flex justify-between items-start mb-2">
                    <h4 className="font-semibold text-gray-900">{transcription.filename}</h4>
                    <span className="text-xs text-gray-500">
                      {transcription.timestamp.toLocaleTimeString()}
                    </span>
                  </div>
                  <p className="text-sm text-gray-600 line-clamp-2">{transcription.text}</p>
                  <div className="mt-2 flex space-x-2">
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => handleCopyText(transcription.text)}
                    >
                      <Copy className="w-3 h-3 mr-1" />
                      Copy
                    </Button>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => handleDownloadText(transcription.text, transcription.filename)}
                    >
                      <Download className="w-3 h-3 mr-1" />
                      Download
                    </Button>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
