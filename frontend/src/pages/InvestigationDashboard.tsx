import { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Progress } from '@/components/ui/progress';
import { 
  Upload, AlertTriangle, TrendingUp, Activity, 
  Users, FileAudio, MessageSquare, Brain, Download 
} from 'lucide-react';
import { useInvestigation } from '@/store/InvestigationContext';
import { apiService } from '@/services/api';

export function InvestigationDashboard() {
  const { investigation, addVoiceNote, setCurrentAnalysis } = useInvestigation();
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);

  const handleFileUpload = async (file: File) => {
    setUploading(true);
    setProgress(10);

    try {
      // Step 1: Transcribe
      setProgress(30);
      const transcription = await apiService.transcribeAudio({
        file,
        language: 'af',
        model_size: 'small'
      });

      // Step 2: Analyze
      setProgress(60);
      const analysis = await apiService.analyzeAudio(file);

      // Step 3: Combine results
      setProgress(90);
      const voiceNote = {
        id: `vn_${Date.now()}`,
        filename: file.name,
        timestamp: new Date(),
        transcription: transcription.text,
        stressLevel: analysis.stress_level,
        pitchVolatility: analysis.pitch_volatility,
        silenceRatio: analysis.silence_ratio,
        duration: analysis.duration,
        intensity: analysis.intensity,
        spectralCentroid: analysis.spectral_centroid,
        deceptionIndicators: detectDeception(analysis, transcription.text),
      };

      addVoiceNote(voiceNote);
      setCurrentAnalysis(voiceNote);
      setProgress(100);
      
    } catch (error) {
      console.error('Upload failed:', error);
    } finally {
      setUploading(false);
      setTimeout(() => setProgress(0), 1000);
    }
  };

  const detectDeception = (analysis: any, text?: string) => {
    const indicators: string[] = [];
    
    if (analysis.stress_level > 70) {
      indicators.push('High vocal stress detected');
    }
    if (analysis.pitch_volatility > 50) {
      indicators.push('Unusual pitch variation');
    }
    if (analysis.silence_ratio > 0.4) {
      indicators.push('Excessive pauses (possible deception)');
    }
    if (text && (text.includes('honestly') || text.includes('to be honest'))) {
      indicators.push('Qualifier language detected');
    }
    
    return indicators;
  };

  const getStressColor = (level?: number) => {
    if (!level) return 'bg-gray-200';
    if (level < 30) return 'bg-green-500';
    if (level < 60) return 'bg-yellow-500';
    return 'bg-red-500';
  };

  const currentNote = investigation?.currentAnalysis;
  const allNotes = investigation?.voiceNotes || [];
  const timeline = investigation?.timeline || [];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Forensic Investigation Dashboard</h1>
          <p className="text-gray-600">Real-time voice analysis, deception detection, and pattern recognition</p>
        </div>
        {investigation && (
          <div className="text-right">
            <div className="text-sm text-gray-600">Investigation</div>
            <div className="font-semibold">{investigation.participantA} vs {investigation.participantB}</div>
          </div>
        )}
      </div>

      {/* Quick Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-sm text-gray-600">Voice Notes</div>
                <div className="text-3xl font-bold">{allNotes.length}</div>
              </div>
              <FileAudio className="w-8 h-8 text-blue-600" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-sm text-gray-600">Deception Alerts</div>
                <div className="text-3xl font-bold text-red-600">
                  {investigation?.patterns.deceptionCount || 0}
                </div>
              </div>
              <AlertTriangle className="w-8 h-8 text-red-600" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-sm text-gray-600">High Stress</div>
                <div className="text-3xl font-bold text-orange-600">
                  {investigation?.patterns.highStressCount || 0}
                </div>
              </div>
              <Activity className="w-8 h-8 text-orange-600" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-sm text-gray-600">Contradictions</div>
                <div className="text-3xl font-bold text-purple-600">
                  {investigation?.patterns.contradictions.length || 0}
                </div>
              </div>
              <Brain className="w-8 h-8 text-purple-600" />
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Main Analysis Area */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left: Upload & Current Analysis */}
        <div className="lg:col-span-2 space-y-6">
          {/* Upload */}
          <Card>
            <CardHeader>
              <CardTitle>Upload & Analyze Voice Note</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <Input
                type="file"
                accept=".mp3,.wav,.opus,.ogg,.m4a,.aac"
                onChange={(e) => {
                  const file = e.target.files?.[0];
                  if (file) {
                    setSelectedFile(file);
                    handleFileUpload(file);
                  }
                }}
                disabled={uploading}
              />
              
              {uploading && (
                <div className="space-y-2">
                  <Progress value={progress} />
                  <p className="text-sm text-gray-600 text-center">
                    {progress < 30 ? 'Uploading...' : 
                     progress < 60 ? 'Transcribing...' : 
                     progress < 90 ? 'Analyzing forensics...' : 
                     'Complete!'}
                  </p>
                </div>
              )}
            </CardContent>
          </Card>

          {/* Current Analysis */}
          {currentNote && (
            <Card className="border-2 border-blue-500">
              <CardHeader>
                <div className="flex justify-between items-start">
                  <CardTitle>Current Analysis: {currentNote.filename}</CardTitle>
                  <Button variant="outline" size="sm">
                    <Download className="w-4 h-4 mr-2" />
                    Export
                  </Button>
                </div>
              </CardHeader>
              <CardContent className="space-y-6">
                {/* Stress Indicator */}
                <div>
                  <div className="flex justify-between items-center mb-2">
                    <span className="text-sm font-medium">Stress Level</span>
                    <span className="text-2xl font-bold">
                      {currentNote.stressLevel?.toFixed(1)}
                    </span>
                  </div>
                  <div className="w-full h-4 bg-gray-200 rounded-full overflow-hidden">
                    <div
                      className={`h-full ${getStressColor(currentNote.stressLevel)} transition-all`}
                      style={{ width: `${currentNote.stressLevel}%` }}
                    />
                  </div>
                  <div className="flex justify-between text-xs text-gray-600 mt-1">
                    <span>Low</span>
                    <span>Moderate</span>
                    <span>High</span>
                  </div>
                </div>

                {/* Deception Indicators */}
                {currentNote.deceptionIndicators && currentNote.deceptionIndicators.length > 0 && (
                  <div className="bg-red-50 border border-red-200 rounded-lg p-4">
                    <div className="flex items-center space-x-2 mb-2">
                      <AlertTriangle className="w-5 h-5 text-red-600" />
                      <span className="font-semibold text-red-900">Deception Indicators Detected</span>
                    </div>
                    <ul className="space-y-1">
                      {currentNote.deceptionIndicators.map((indicator, i) => (
                        <li key={i} className="text-sm text-red-800">• {indicator}</li>
                      ))}
                    </ul>
                  </div>
                )}

                {/* Transcription */}
                <div>
                  <h4 className="font-semibold mb-2">Transcription</h4>
                  <div className="bg-gray-50 p-4 rounded-lg">
                    <p className="text-gray-900 whitespace-pre-wrap">
                      {currentNote.transcription || 'Processing...'}
                    </p>
                  </div>
                </div>

                {/* Forensic Metrics */}
                <div className="grid grid-cols-2 gap-4">
                  <div className="bg-blue-50 p-3 rounded-lg">
                    <div className="text-xs text-gray-600">Pitch Volatility</div>
                    <div className="text-xl font-bold text-blue-600">
                      {currentNote.pitchVolatility?.toFixed(2)}
                    </div>
                  </div>
                  <div className="bg-purple-50 p-3 rounded-lg">
                    <div className="text-xs text-gray-600">Silence Ratio</div>
                    <div className="text-xl font-bold text-purple-600">
                      {((currentNote.silenceRatio || 0) * 100).toFixed(1)}%
                    </div>
                  </div>
                  <div className="bg-green-50 p-3 rounded-lg">
                    <div className="text-xs text-gray-600">Duration</div>
                    <div className="text-xl font-bold text-green-600">
                      {currentNote.duration?.toFixed(1)}s
                    </div>
                  </div>
                  <div className="bg-orange-50 p-3 rounded-lg">
                    <div className="text-xs text-gray-600">Spectral Centroid</div>
                    <div className="text-xl font-bold text-orange-600">
                      {currentNote.spectralCentroid?.toFixed(0)} Hz
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          )}
        </div>

        {/* Right: Timeline */}
        <div className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Investigation Timeline</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3 max-h-[800px] overflow-y-auto">
                {timeline.length === 0 ? (
                  <p className="text-gray-500 text-center py-8">
                    No voice notes yet. Upload files to begin analysis.
                  </p>
                ) : (
                  timeline.map((note) => (
                    <div
                      key={note.id}
                      onClick={() => setCurrentAnalysis(note)}
                      className={`p-3 rounded-lg border cursor-pointer transition-all ${
                        currentNote?.id === note.id
                          ? 'border-blue-500 bg-blue-50'
                          : 'border-gray-200 hover:border-gray-300'
                      }`}
                    >
                      <div className="flex justify-between items-start mb-2">
                        <span className="text-sm font-medium truncate flex-1">
                          {note.filename}
                        </span>
                        {note.deceptionIndicators && note.deceptionIndicators.length > 0 && (
                          <AlertTriangle className="w-4 h-4 text-red-600 flex-shrink-0 ml-2" />
                        )}
                      </div>
                      
                      <div className="flex items-center space-x-2 mb-2">
                        <div className="flex-1 h-2 bg-gray-200 rounded-full overflow-hidden">
                          <div
                            className={`h-full ${getStressColor(note.stressLevel)}`}
                            style={{ width: `${note.stressLevel}%` }}
                          />
                        </div>
                        <span className="text-xs font-medium">
                          {note.stressLevel?.toFixed(0)}
                        </span>
                      </div>
                      
                      <p className="text-xs text-gray-600 line-clamp-2">
                        {note.transcription || 'Processing...'}
                      </p>
                      
                      <div className="text-xs text-gray-500 mt-2">
                        {note.timestamp.toLocaleTimeString()}
                      </div>
                    </div>
                  ))
                )}
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
