import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Mic, FileAudio, Users, BarChart3 } from 'lucide-react';
import { apiService } from '@/services/api';

export function Home() {
  const [audioFileCount, setAudioFileCount] = useState<number>(0);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      const count = await apiService.getAudioFileCount();
      setAudioFileCount(count);
      setLoading(false);
    };
    fetchData();
  }, []);

  const features = [
    {
      icon: Mic,
      title: 'Audio Transcription',
      description: 'Convert voice notes to text using Whisper AI with Afrikaans support',
      link: '/transcribe',
      color: 'text-blue-600',
    },
    {
      icon: Users,
      title: 'Speaker Training',
      description: 'Train AI to distinguish between speakers using voice fingerprinting',
      link: '/speaker-training',
      color: 'text-green-600',
    },
    {
      icon: FileAudio,
      title: 'Audio Analysis',
      description: 'Forensic audio analysis with stress detection and pitch analysis',
      link: '/audio-analysis',
      color: 'text-purple-600',
    },
    {
      icon: BarChart3,
      title: 'Chat Analysis',
      description: 'Analyze WhatsApp chat messages and conversation patterns',
      link: '/chat-analysis',
      color: 'text-orange-600',
    },
  ];

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-4xl font-bold text-gray-900 mb-2">
          Welcome to Die Waarheid
        </h1>
        <p className="text-lg text-gray-600">
          Forensic-Grade WhatsApp Communication Analysis Platform
        </p>
      </div>

      {/* Stats Card */}
      <Card>
        <CardHeader>
          <CardTitle>System Status</CardTitle>
          <CardDescription>Current data and system information</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="p-4 bg-blue-50 rounded-lg">
              <div className="text-sm text-gray-600 mb-1">Audio Files</div>
              <div className="text-3xl font-bold text-blue-600">
                {loading ? '...' : audioFileCount.toLocaleString()}
              </div>
            </div>
            <div className="p-4 bg-green-50 rounded-lg">
              <div className="text-sm text-gray-600 mb-1">Status</div>
              <div className="text-3xl font-bold text-green-600">Ready</div>
            </div>
            <div className="p-4 bg-purple-50 rounded-lg">
              <div className="text-sm text-gray-600 mb-1">Version</div>
              <div className="text-3xl font-bold text-purple-600">1.0.0</div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Features Grid */}
      <div>
        <h2 className="text-2xl font-bold text-gray-900 mb-4">Features</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {features.map((feature) => {
            const Icon = feature.icon;
            return (
              <Card key={feature.title} className="hover:shadow-lg transition-shadow">
                <CardHeader>
                  <div className="flex items-center space-x-3">
                    <Icon className={`w-8 h-8 ${feature.color}`} />
                    <CardTitle>{feature.title}</CardTitle>
                  </div>
                  <CardDescription>{feature.description}</CardDescription>
                </CardHeader>
                <CardContent>
                  <Link to={feature.link}>
                    <Button className="w-full">Get Started</Button>
                  </Link>
                </CardContent>
              </Card>
            );
          })}
        </div>
      </div>

      {/* Quick Start */}
      <Card>
        <CardHeader>
          <CardTitle>Quick Start Guide</CardTitle>
          <CardDescription>Get started with Die Waarheid in 3 simple steps</CardDescription>
        </CardHeader>
        <CardContent>
          <ol className="space-y-4">
            <li className="flex items-start space-x-3">
              <span className="flex-shrink-0 w-8 h-8 bg-blue-600 text-white rounded-full flex items-center justify-center font-bold">
                1
              </span>
              <div>
                <h3 className="font-semibold text-gray-900">Initialize Speaker Training</h3>
                <p className="text-gray-600">Set up the two participants and upload voice samples</p>
              </div>
            </li>
            <li className="flex items-start space-x-3">
              <span className="flex-shrink-0 w-8 h-8 bg-blue-600 text-white rounded-full flex items-center justify-center font-bold">
                2
              </span>
              <div>
                <h3 className="font-semibold text-gray-900">Transcribe Audio Files</h3>
                <p className="text-gray-600">Convert your voice notes to searchable text</p>
              </div>
            </li>
            <li className="flex items-start space-x-3">
              <span className="flex-shrink-0 w-8 h-8 bg-blue-600 text-white rounded-full flex items-center justify-center font-bold">
                3
              </span>
              <div>
                <h3 className="font-semibold text-gray-900">Analyze Results</h3>
                <p className="text-gray-600">Review forensic analysis and generate reports</p>
              </div>
            </li>
          </ol>
        </CardContent>
      </Card>
    </div>
  );
}
