import React, { createContext, useContext, useState, ReactNode } from 'react';

interface VoiceNote {
  id: string;
  filename: string;
  timestamp: Date;
  speaker?: string;
  transcription?: string;
  stressLevel?: number;
  deceptionIndicators?: string[];
  pitchVolatility?: number;
  silenceRatio?: number;
  duration?: number;
  intensity?: {
    max: number;
    mean: number;
    std: number;
  };
  spectralCentroid?: number;
  audioUrl?: string;
}

interface Investigation {
  id: string;
  participantA: string;
  participantB: string;
  voiceNotes: VoiceNote[];
  currentAnalysis?: VoiceNote;
  timeline: VoiceNote[];
  patterns: {
    deceptionCount: number;
    highStressCount: number;
    contradictions: string[];
  };
}

interface InvestigationContextType {
  investigation: Investigation | null;
  setInvestigation: (investigation: Investigation) => void;
  addVoiceNote: (note: VoiceNote) => void;
  setCurrentAnalysis: (note: VoiceNote) => void;
  updateVoiceNote: (id: string, updates: Partial<VoiceNote>) => void;
  clearInvestigation: () => void;
}

const InvestigationContext = createContext<InvestigationContextType | undefined>(undefined);

export function InvestigationProvider({ children }: { children: ReactNode }) {
  const [investigation, setInvestigationState] = useState<Investigation | null>(null);

  const setInvestigation = (inv: Investigation) => {
    setInvestigationState(inv);
    localStorage.setItem('investigation', JSON.stringify(inv));
  };

  const addVoiceNote = (note: VoiceNote) => {
    if (!investigation) return;
    
    const updated = {
      ...investigation,
      voiceNotes: [...investigation.voiceNotes, note],
      timeline: [...investigation.timeline, note].sort((a, b) => 
        new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime()
      ),
    };
    
    // Update patterns
    if (note.stressLevel && note.stressLevel > 60) {
      updated.patterns.highStressCount++;
    }
    if (note.deceptionIndicators && note.deceptionIndicators.length > 0) {
      updated.patterns.deceptionCount++;
    }
    
    setInvestigation(updated);
  };

  const setCurrentAnalysis = (note: VoiceNote) => {
    if (!investigation) return;
    setInvestigation({ ...investigation, currentAnalysis: note });
  };

  const updateVoiceNote = (id: string, updates: Partial<VoiceNote>) => {
    if (!investigation) return;
    
    const updated = {
      ...investigation,
      voiceNotes: investigation.voiceNotes.map(note =>
        note.id === id ? { ...note, ...updates } : note
      ),
      timeline: investigation.timeline.map(note =>
        note.id === id ? { ...note, ...updates } : note
      ),
    };
    
    setInvestigation(updated);
  };

  const clearInvestigation = () => {
    setInvestigationState(null);
    localStorage.removeItem('investigation');
  };

  // Load from localStorage on mount
  React.useEffect(() => {
    const stored = localStorage.getItem('investigation');
    if (stored) {
      try {
        const parsed = JSON.parse(stored);
        // Convert timestamp strings back to Date objects
        parsed.voiceNotes = parsed.voiceNotes.map((note: VoiceNote) => ({
          ...note,
          timestamp: new Date(note.timestamp),
        }));
        parsed.timeline = parsed.timeline.map((note: VoiceNote) => ({
          ...note,
          timestamp: new Date(note.timestamp),
        }));
        setInvestigationState(parsed);
      } catch (e) {
        console.error('Failed to load investigation from storage', e);
      }
    }
  }, []);

  return (
    <InvestigationContext.Provider
      value={{
        investigation,
        setInvestigation,
        addVoiceNote,
        setCurrentAnalysis,
        updateVoiceNote,
        clearInvestigation,
      }}
    >
      {children}
    </InvestigationContext.Provider>
  );
}

export function useInvestigation() {
  const context = useContext(InvestigationContext);
  if (context === undefined) {
    throw new Error('useInvestigation must be used within InvestigationProvider');
  }
  return context;
}
