import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { BarChart3, MessageSquare, Users, TrendingUp } from 'lucide-react';

export function ChatAnalysis() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-gray-900 mb-2">Chat Analysis</h1>
        <p className="text-gray-600">Analyze WhatsApp chat messages and conversation patterns</p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Coming Soon</CardTitle>
          <CardDescription>Chat analysis features are under development</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="border border-gray-200 rounded-lg p-4">
              <div className="flex items-center space-x-3 mb-2">
                <MessageSquare className="w-6 h-6 text-blue-600" />
                <h3 className="font-semibold text-gray-900">Message Analysis</h3>
              </div>
              <p className="text-sm text-gray-600">
                Analyze message frequency, timing patterns, and communication styles
              </p>
            </div>

            <div className="border border-gray-200 rounded-lg p-4">
              <div className="flex items-center space-x-3 mb-2">
                <Users className="w-6 h-6 text-green-600" />
                <h3 className="font-semibold text-gray-900">Participant Profiling</h3>
              </div>
              <p className="text-sm text-gray-600">
                Build psychological profiles based on communication patterns
              </p>
            </div>

            <div className="border border-gray-200 rounded-lg p-4">
              <div className="flex items-center space-x-3 mb-2">
                <TrendingUp className="w-6 h-6 text-purple-600" />
                <h3 className="font-semibold text-gray-900">Pattern Detection</h3>
              </div>
              <p className="text-sm text-gray-600">
                Detect contradictions, gaslighting, and manipulation patterns
              </p>
            </div>

            <div className="border border-gray-200 rounded-lg p-4">
              <div className="flex items-center space-x-3 mb-2">
                <BarChart3 className="w-6 h-6 text-orange-600" />
                <h3 className="font-semibold text-gray-900">Sentiment Analysis</h3>
              </div>
              <p className="text-sm text-gray-600">
                Track emotional tone and sentiment changes over time
              </p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
