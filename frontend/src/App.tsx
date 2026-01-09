import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { InvestigationProvider } from './store/InvestigationContext';
import { Layout } from './components/Layout';
import { Home } from './pages/Home';
import { InvestigationDashboard } from './pages/InvestigationDashboard';
import { SpeakerTraining } from './pages/SpeakerTraining';
import { Transcribe } from './pages/Transcribe';
import { AudioAnalysis } from './pages/AudioAnalysis';
import { ChatAnalysis } from './pages/ChatAnalysis';

function App() {
  return (
    <InvestigationProvider>
      <Router>
        <Layout>
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/dashboard" element={<InvestigationDashboard />} />
            <Route path="/speaker-training" element={<SpeakerTraining />} />
            <Route path="/transcribe" element={<Transcribe />} />
            <Route path="/audio-analysis" element={<AudioAnalysis />} />
            <Route path="/chat-analysis" element={<ChatAnalysis />} />
          </Routes>
        </Layout>
      </Router>
    </InvestigationProvider>
  );
}

export default App
