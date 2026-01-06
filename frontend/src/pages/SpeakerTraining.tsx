import { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Users, Upload, CheckCircle, AlertCircle } from 'lucide-react';
import { apiService, type SpeakerProfile } from '@/services/api';

export function SpeakerTraining() {
  const [speakers, setSpeakers] = useState<SpeakerProfile[]>([]);
  const [participantA, setParticipantA] = useState('Speaker A');
  const [participantB, setParticipantB] = useState('Speaker B');
  const [isInitializing, setIsInitializing] = useState(false);
  const [trainingFile, setTrainingFile] = useState<{ [key: string]: File | null }>({});
  const [trainingStatus, setTrainingStatus] = useState<{ [key: string]: string }>({});

  const loadSpeakers = async () => {
    const profiles = await apiService.getSpeakerProfiles();
    setSpeakers(profiles);
  };

  useEffect(() => {
    loadSpeakers();
  }, []);

  const handleInitialize = async () => {
    if (!participantA || !participantB) {
      alert('Please enter names for both participants');
      return;
    }

    setIsInitializing(true);
    const result = await apiService.initializeSpeakers(participantA, participantB);
    
    if (result.success) {
      await loadSpeakers();
    } else {
      alert(`Initialization failed: ${result.message}`);
    }
    setIsInitializing(false);
  };

  const handleFileSelect = (participantId: string, file: File | null) => {
    setTrainingFile({ ...trainingFile, [participantId]: file });
  };

  const handleTrain = async (participantId: string) => {
    const file = trainingFile[participantId];
    if (!file) {
      alert('Please select an audio file');
      return;
    }

    setTrainingStatus({ ...trainingStatus, [participantId]: 'training' });
    const result = await apiService.trainSpeaker(participantId, file);
    
    if (result.success) {
      setTrainingStatus({ ...trainingStatus, [participantId]: 'success' });
      await loadSpeakers();
      setTimeout(() => {
        setTrainingStatus({ ...trainingStatus, [participantId]: '' });
        setTrainingFile({ ...trainingFile, [participantId]: null });
      }, 3000);
    } else {
      setTrainingStatus({ ...trainingStatus, [participantId]: 'error' });
      alert(`Training failed: ${result.message}`);
      setTimeout(() => {
        setTrainingStatus({ ...trainingStatus, [participantId]: '' });
      }, 3000);
    }
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-gray-900 mb-2">Speaker Training</h1>
        <p className="text-gray-600">Train AI to distinguish between speakers using voice fingerprinting</p>
      </div>

      {speakers.length < 2 ? (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Users className="w-5 h-5" />
              <span>Initialize Investigation</span>
            </CardTitle>
            <CardDescription>Set up the two participants for this investigation</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Participant A Name
                </label>
                <Input
                  type="text"
                  value={participantA}
                  onChange={(e) => setParticipantA(e.target.value)}
                  placeholder="Enter name for Participant A"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Participant B Name
                </label>
                <Input
                  type="text"
                  value={participantB}
                  onChange={(e) => setParticipantB(e.target.value)}
                  placeholder="Enter name for Participant B"
                />
              </div>
            </div>
            <Button
              onClick={handleInitialize}
              disabled={isInitializing || !participantA || !participantB}
              size="lg"
            >
              {isInitializing ? 'Initializing...' : 'Initialize Investigation'}
            </Button>
          </CardContent>
        </Card>
      ) : (
        <>
          <Card>
            <CardHeader>
              <CardTitle>Current Participants</CardTitle>
              <CardDescription>Speaker profiles and training status</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {speakers.map((speaker) => (
                  <div key={speaker.participant_id} className="border border-gray-200 rounded-lg p-4">
                    <div className="flex items-center justify-between mb-4">
                      <h3 className="text-lg font-semibold text-gray-900">
                        {speaker.primary_username}
                      </h3>
                      <span className="text-xs text-gray-500 uppercase">
                        {speaker.assigned_role.replace('_', ' ')}
                      </span>
                    </div>
                    
                    <div className="grid grid-cols-3 gap-2 mb-4">
                      <div className="bg-blue-50 p-2 rounded text-center">
                        <div className="text-xs text-gray-600">Messages</div>
                        <div className="text-lg font-bold text-blue-600">
                          {speaker.message_count}
                        </div>
                      </div>
                      <div className="bg-green-50 p-2 rounded text-center">
                        <div className="text-xs text-gray-600">Voice Notes</div>
                        <div className="text-lg font-bold text-green-600">
                          {speaker.voice_note_count}
                        </div>
                      </div>
                      <div className="bg-purple-50 p-2 rounded text-center">
                        <div className="text-xs text-gray-600">Confidence</div>
                        <div className="text-lg font-bold text-purple-600">
                          {speaker.confidence_score.toFixed(2)}
                        </div>
                      </div>
                    </div>

                    <div className="text-sm text-gray-600 mb-4">
                      Voice Fingerprints: {speaker.voice_fingerprints}
                    </div>

                    <div className="space-y-2">
                      <Input
                        type="file"
                        accept=".mp3,.wav,.opus,.ogg,.m4a,.aac"
                        onChange={(e) => handleFileSelect(speaker.participant_id, e.target.files?.[0] || null)}
                        className="text-sm"
                      />
                      {trainingFile[speaker.participant_id] && (
                        <p className="text-xs text-gray-600">
                          {trainingFile[speaker.participant_id]?.name}
                        </p>
                      )}
                      <Button
                        onClick={() => handleTrain(speaker.participant_id)}
                        disabled={!trainingFile[speaker.participant_id] || trainingStatus[speaker.participant_id] === 'training'}
                        className="w-full"
                        size="sm"
                      >
                        {trainingStatus[speaker.participant_id] === 'training' ? (
                          'Training...'
                        ) : trainingStatus[speaker.participant_id] === 'success' ? (
                          <>
                            <CheckCircle className="w-4 h-4 mr-2" />
                            Trained Successfully
                          </>
                        ) : trainingStatus[speaker.participant_id] === 'error' ? (
                          <>
                            <AlertCircle className="w-4 h-4 mr-2" />
                            Training Failed
                          </>
                        ) : (
                          <>
                            <Upload className="w-4 h-4 mr-2" />
                            Train Voice Sample
                          </>
                        )}
                      </Button>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Training Instructions</CardTitle>
            </CardHeader>
            <CardContent>
              <ol className="space-y-2 text-sm text-gray-700">
                <li className="flex items-start">
                  <span className="font-bold mr-2">1.</span>
                  <span>Upload clear voice samples for each participant (at least 10-30 seconds)</span>
                </li>
                <li className="flex items-start">
                  <span className="font-bold mr-2">2.</span>
                  <span>Use samples with minimal background noise for best results</span>
                </li>
                <li className="flex items-start">
                  <span className="font-bold mr-2">3.</span>
                  <span>Upload multiple samples per speaker to improve accuracy</span>
                </li>
                <li className="flex items-start">
                  <span className="font-bold mr-2">4.</span>
                  <span>The system will extract voice fingerprints and build speaker profiles</span>
                </li>
              </ol>
            </CardContent>
          </Card>
        </>
      )}
    </div>
  );
}
