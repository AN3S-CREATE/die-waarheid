const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
const API_KEY = import.meta.env.VITE_API_KEY;

export interface TranscriptionRequest {
  file: File;
  language?: string;
  model_size?: string;
}

export interface TranscriptionResponse {
  success: boolean;
  text?: string;
  duration?: number;
  language?: string;
  message?: string;
}

export interface AudioAnalysisResponse {
  success: boolean;
  stress_level?: number;
  pitch_volatility?: number;
  silence_ratio?: number;
  duration?: number;
  intensity?: {
    max: number;
    mean: number;
    std: number;
  };
  spectral_centroid?: number;
  message?: string;
}

export interface SpeakerProfile {
  participant_id: string;
  primary_username: string;
  assigned_role: string;
  voice_note_count: number;
  message_count: number;
  confidence_score: number;
  voice_fingerprints: number;
}

class ApiService {
  private buildHeaders(headers?: HeadersInit) {
    const merged = new Headers(headers);
    if (API_KEY) {
      merged.set('Authorization', `Bearer ${API_KEY}`);
    }
    return merged;
  }

  private async fetchWithTimeout(url: string, options: RequestInit, timeout = 30000) {
    const controller = new AbortController();
    const id = setTimeout(() => controller.abort(), timeout);
    
    try {
      const response = await fetch(url, {
        ...options,
        headers: this.buildHeaders(options.headers),
        signal: controller.signal,
      });
      clearTimeout(id);
      return response;
    } catch (error) {
      clearTimeout(id);
      throw error;
    }
  }

  async transcribeAudio(request: TranscriptionRequest): Promise<TranscriptionResponse> {
    const formData = new FormData();
    formData.append('file', request.file);
    if (request.language) formData.append('language', request.language);
    if (request.model_size) formData.append('model_size', request.model_size);

    try {
      const response = await this.fetchWithTimeout(
        `${API_BASE_URL}/api/transcribe`,
        {
          method: 'POST',
          body: formData,
        },
        120000 // 2 minute timeout for transcription
      );

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Transcription error:', error);
      return {
        success: false,
        message: error instanceof Error ? error.message : 'Transcription failed',
      };
    }
  }

  async analyzeAudio(file: File): Promise<AudioAnalysisResponse> {
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await this.fetchWithTimeout(
        `${API_BASE_URL}/api/analyze`,
        {
          method: 'POST',
          body: formData,
        },
        60000 // 1 minute timeout
      );

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Analysis error:', error);
      return {
        success: false,
        message: error instanceof Error ? error.message : 'Analysis failed',
      };
    }
  }

  async getSpeakerProfiles(): Promise<SpeakerProfile[]> {
    try {
      const response = await this.fetchWithTimeout(
        `${API_BASE_URL}/api/speakers`,
        { method: 'GET' }
      );

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Get speakers error:', error);
      return [];
    }
  }

  async initializeSpeakers(participantA: string, participantB: string): Promise<{ success: boolean; message: string }> {
    try {
      const response = await this.fetchWithTimeout(
        `${API_BASE_URL}/api/speakers/initialize`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ participant_a: participantA, participant_b: participantB }),
        }
      );

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Initialize speakers error:', error);
      return {
        success: false,
        message: error instanceof Error ? error.message : 'Initialization failed',
      };
    }
  }

  async trainSpeaker(participantId: string, file: File): Promise<{ success: boolean; message: string }> {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('participant_id', participantId);

    try {
      const response = await this.fetchWithTimeout(
        `${API_BASE_URL}/api/speakers/train`,
        {
          method: 'POST',
          body: formData,
        },
        60000
      );

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Train speaker error:', error);
      return {
        success: false,
        message: error instanceof Error ? error.message : 'Training failed',
      };
    }
  }

  async getAudioFileCount(): Promise<number> {
    try {
      const response = await this.fetchWithTimeout(
        `${API_BASE_URL}/api/files/count`,
        { method: 'GET' }
      );

      if (!response.ok) {
        return 0;
      }

      const data = await response.json();
      return data.count || 0;
    } catch (error) {
      console.error('Get file count error:', error);
      return 0;
    }
  }
}

export const apiService = new ApiService();
