import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { InvestigationProvider } from './store/InvestigationContext';
import { Layout } from './components/Layout';
import { InvestigationDashboard } from './pages/InvestigationDashboard';
import { SpeakerTraining } from './pages/SpeakerTraining';

function App() {
  return (
    <InvestigationProvider>
      <Router>
        <Layout>
          <Routes>
            <Route path="/" element={<InvestigationDashboard />} />
            <Route path="/speaker-training" element={<SpeakerTraining />} />
          </Routes>
        </Layout>
      </Router>
    </InvestigationProvider>
  );
}

export default App
