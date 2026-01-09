import { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Progress } from '@/components/ui/progress';
import { FileAudio, Activity, TrendingUp, Volume2 } from 'lucide-react';
import { apiService, type AudioAnalysisResponse } from '@/services/api';

export function AudioAnalysis() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [result, setResult] = useState<AudioAnalysisResponse | null>(null);

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      setResult(null);
    }
  };

  const handleAnalyze = async () => {
    if (!selectedFile) return;

    setIsAnalyzing(true);
    setProgress(10);
    setResult(null);

    try {
      const progressInterval = setInterval(() => {
        setProgress((prev) => Math.min(prev + 10, 90));
      }, 500);

      const analysisResult = await apiService.analyzeAudio(selectedFile);

      clearInterval(progressInterval);
      setProgress(100);
      setResult(analysisResult);
    } catch (error) {
      console.error('Analysis error:', error);
      setResult({
        success: false,
        message: 'Analysis failed. Please try again.',
      });
    } finally {
      setIsAnalyzing(false);
      setTimeout(() => setProgress(0), 1000);
    }
  };

  const getStressLevelColor = (level: number) => {
    if (level < 30) return 'text-green-600 bg-green-50';
    if (level < 60) return 'text-yellow-600 bg-yellow-50';
    return 'text-red-600 bg-red-50';
  };

  const getStressLevelLabel = (level: number) => {
    if (level < 30) return 'Low';
    if (level < 60) return 'Moderate';
    return 'High';
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-gray-900 mb-2">Audio Analysis</h1>
        <p className="text-gray-600">Forensic audio analysis with stress detection and pitch analysis</p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Upload Audio File</CardTitle>
          <CardDescription>Select an audio file for forensic analysis</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div>
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

          <Button
            onClick={handleAnalyze}
            disabled={!selectedFile || isAnalyzing}
            size="lg"
          >
            {isAnalyzing ? (
              'Analyzing...'
            ) : (
              <>
                <FileAudio className="w-4 h-4 mr-2" />
                Analyze Audio
              </>
            )}
          </Button>

          {isAnalyzing && (
            <div className="space-y-2">
              <Progress value={progress} />
              <p className="text-sm text-gray-600 text-center">
                Analyzing audio... {progress}%
              </p>
            </div>
          )}
        </CardContent>
      </Card>

      {result && result.success && (
        <>
          <Card>
            <CardHeader>
              <CardTitle>Stress Analysis</CardTitle>
              <CardDescription>Vocal stress indicators and emotional state</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className={`p-6 rounded-lg ${getStressLevelColor(result.stress_level || 0)}`}>
                  <div className="flex items-center justify-between">
                    <div>
                      <div className="text-sm font-medium mb-1">Overall Stress Level</div>
                      <div className="text-4xl font-bold">{result.stress_level?.toFixed(1)}</div>
                      <div className="text-sm mt-1">{getStressLevelLabel(result.stress_level || 0)}</div>
                    </div>
                    <Activity className="w-16 h-16 opacity-50" />
                  </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="border border-gray-200 rounded-lg p-4">
                    <div className="flex items-center space-x-2 mb-2">
                      <TrendingUp className="w-5 h-5 text-blue-600" />
                      <span className="font-semibold text-gray-900">Pitch Volatility</span>
                    </div>
                    <div className="text-3xl font-bold text-blue-600">
                      {result.pitch_volatility?.toFixed(2)}
                    </div>
                    <p className="text-sm text-gray-600 mt-1">
                      Measures voice pitch variation
                    </p>
                  </div>

                  <div className="border border-gray-200 rounded-lg p-4">
                    <div className="flex items-center space-x-2 mb-2">
                      <Volume2 className="w-5 h-5 text-purple-600" />
                      <span className="font-semibold text-gray-900">Silence Ratio</span>
                    </div>
                    <div className="text-3xl font-bold text-purple-600">
                      {((result.silence_ratio ?? 0) * 100).toFixed(1)}%
                    </div>
                    <p className="text-sm text-gray-600 mt-1">
                      Percentage of silent segments
                    </p>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Audio Characteristics</CardTitle>
              <CardDescription>Technical audio analysis metrics</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="bg-gray-50 p-4 rounded-lg">
                  <div className="text-sm text-gray-600 mb-1">Duration</div>
                  <div className="text-2xl font-bold text-gray-900">
                    {result.duration?.toFixed(2)}s
                  </div>
                </div>

                <div className="bg-gray-50 p-4 rounded-lg">
                  <div className="text-sm text-gray-600 mb-1">Max Intensity</div>
                  <div className="text-2xl font-bold text-gray-900">
                    {result.intensity?.max?.toFixed(2) ?? 'N/A'} dB
                  </div>
                </div>

                <div className="bg-gray-50 p-4 rounded-lg">
                  <div className="text-sm text-gray-600 mb-1">Spectral Centroid</div>
                  <div className="text-2xl font-bold text-gray-900">
                    {result.spectral_centroid?.toFixed(0)} Hz
                  </div>
                </div>

                {result.intensity && (
                  <>
                    <div className="bg-gray-50 p-4 rounded-lg">
                      <div className="text-sm text-gray-600 mb-1">Mean Intensity</div>
                      <div className="text-2xl font-bold text-gray-900">
                        {result.intensity.mean.toFixed(2)} dB
                      </div>
                    </div>

                    <div className="bg-gray-50 p-4 rounded-lg">
                      <div className="text-sm text-gray-600 mb-1">Intensity Std Dev</div>
                      <div className="text-2xl font-bold text-gray-900">
                        {result.intensity.std.toFixed(2)} dB
                      </div>
                    </div>
                  </>
                )}
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Interpretation Guide</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3 text-sm">
                <div>
                  <span className="font-semibold">Stress Level:</span> Indicates overall vocal stress. 
                  High values may suggest emotional distress, deception, or anxiety.
                </div>
                <div>
                  <span className="font-semibold">Pitch Volatility:</span> Measures voice pitch changes. 
                  Higher values indicate more emotional or stressed speech patterns.
                </div>
                <div>
                  <span className="font-semibold">Silence Ratio:</span> Percentage of pauses in speech. 
                  Unusual patterns may indicate hesitation or careful word selection.
                </div>
                <div>
                  <span className="font-semibold">Spectral Centroid:</span> Indicates voice brightness. 
                  Changes may reflect emotional state or vocal strain.
                </div>
              </div>
            </CardContent>
          </Card>
        </>
      )}

      {result && !result.success && (
        <Card>
          <CardHeader>
            <CardTitle className="text-red-600">Analysis Failed</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-gray-700">{result.message}</p>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
