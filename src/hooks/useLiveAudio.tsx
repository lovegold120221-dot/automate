import { useState, useRef, useEffect } from 'react';
import { GoogleGenAI, LiveServerMessage, Modality, Type, MediaResolution } from "@google/genai";
import { db, auth as firebaseAuth } from '../lib/firebase';
import { collection, addDoc, serverTimestamp } from 'firebase/firestore';

const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });

const workletCode = `
class RecorderProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.bufferSize = 4096;
    this.buffer = new Int16Array(this.bufferSize);
    this.offset = 0;
  }
  process(inputs, outputs, parameters) {
    const input = inputs[0];
    if (input.length > 0) {
      const channelData = input[0];
      for (let i = 0; i < channelData.length; i++) {
        const s = Math.max(-1, Math.min(1, channelData[i]));
        this.buffer[this.offset++] = s < 0 ? s * 0x8000 : s * 0x7FFF;
        if (this.offset >= this.bufferSize) {
          const out = new Int16Array(this.buffer);
          this.port.postMessage(out.buffer, [out.buffer]);
          this.offset = 0;
        }
      }
    }
    return true;
  }
}
registerProcessor('recorder-processor', RecorderProcessor);
`;

function arrayBufferToBase64(buffer: ArrayBuffer) {
  let binary = '';
  const bytes = new Uint8Array(buffer);
  const len = bytes.byteLength;
  for (let i = 0; i < len; i++) {
    binary += String.fromCharCode(bytes[i]);
  }
  return window.btoa(binary);
}

export type TalkContext = 'Work' | 'Personal' | 'Travel';

export function useLiveAPI(contextString: TalkContext = 'Work', enableCamera = true) {
  const [connected, setConnected] = useState(false);
  const [transcript, setTranscript] = useState<{ role: 'jo' | 'beatrice', text: string, time: string }[]>([]);
  const [speaking, setSpeaking] = useState(false);
  const [detectedLanguage, setDetectedLanguage] = useState<{input: string, output: string, confidence: string} | null>(null);
  const [requestedTab, setRequestedTab] = useState<string | null>(null);
  const [requestedDocPreview, setRequestedDocPreview] = useState<string | null>(null);
  const [requestedDocSearch, setRequestedDocSearch] = useState<string | null>(null);
  const [requestedContractParams, setRequestedContractParams] = useState<{partyNames?: string, governingLaw?: string, termLength?: string} | null>(null);
  const [requestedCalendarEvent, setRequestedCalendarEvent] = useState<{title?: string, startTimeIso?: string, endTimeIso?: string, attendees?: string} | null>(null);
  const [micStrength, setMicStrength] = useState(0);

  const audioCtxRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const nextTimeRef = useRef<number>(0);
  const sessionPromiseRef = useRef<any>(null);
  const activeSourcesRef = useRef<Set<AudioBufferSourceNode>>(new Set());
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const frameIntervalRef = useRef<any>(null);

  const initAudioContext = async () => {
    if (!audioCtxRef.current) {
      audioCtxRef.current = new AudioContext({ sampleRate: 16000 });
      const workletBlob = new Blob([workletCode], { type: 'application/javascript' });
      const workletUrl = URL.createObjectURL(workletBlob);
      await audioCtxRef.current.audioWorklet.addModule(workletUrl);
      URL.revokeObjectURL(workletUrl);

      // Setup analyser
      analyserRef.current = audioCtxRef.current.createAnalyser();
      analyserRef.current.fftSize = 256;
    }
    if (audioCtxRef.current.state === 'suspended') {
      await audioCtxRef.current.resume();
    }
  };

  const captureFrame = () => {
    if (!videoRef.current || !canvasRef.current || !sessionPromiseRef.current) return;
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const context = canvas.getContext('2d');
    if (!context) return;

    // Set canvas dimensions to match video
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    
    const base64 = canvas.toDataURL('image/jpeg', 0.5).split(',')[1];
    
    sessionPromiseRef.current.then((session: any) => {
      session.sendRealtimeInput({
        realtimeInput: { mediaChunks: [{ data: base64, mimeType: 'image/jpeg' }] }
      });
    }).catch(console.error);
  };

  const saveInteraction = async (role: 'jo' | 'beatrice', text: string) => {
    const user = firebaseAuth.currentUser;
    if (!user) return;
    try {
      await addDoc(collection(db, 'users', user.uid, 'interactions'), {
        userId: user.uid,
        role,
        text,
        timestamp: serverTimestamp(),
        context: contextString
      });
    } catch (err) {
      console.error("Failed to save interaction", err);
    }
  };

  // Monitor mic strength
  useEffect(() => {
    let intervalId: any;
    if (connected && analyserRef.current) {
      const bufferLength = analyserRef.current.frequencyBinCount;
      const dataArray = new Uint8Array(bufferLength);
      
      intervalId = setInterval(() => {
        if (analyserRef.current) {
          analyserRef.current.getByteFrequencyData(dataArray);
          let sum = 0;
          for (let i = 0; i < bufferLength; i++) {
            sum += dataArray[i];
          }
          const average = sum / bufferLength;
          setMicStrength(average / 128); // Normalize roughly to 0-1
        }
      }, 100);
    } else {
      setMicStrength(0);
    }
    return () => clearInterval(intervalId);
  }, [connected]);

  const playAudioChunk = (base64Data: string) => {
    if (!audioCtxRef.current) return;
    const ctx = audioCtxRef.current;
    
    const binaryString = window.atob(base64Data);
    const bytes = new Uint8Array(binaryString.length);
    for (let i = 0; i < binaryString.length; i++) {
      bytes[i] = binaryString.charCodeAt(i);
    }
    const int16Array = new Int16Array(bytes.buffer);
    const float32Array = new Float32Array(int16Array.length);
    for (let i = 0; i < int16Array.length; i++) {
      float32Array[i] = int16Array[i] / 0x8000;
    }

    const audioBuffer = ctx.createBuffer(1, float32Array.length, 24000);
    audioBuffer.getChannelData(0).set(float32Array);

    const source = ctx.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(ctx.destination);
    
    source.onended = () => {
      activeSourcesRef.current.delete(source);
      if (activeSourcesRef.current.size === 0) {
        setSpeaking(false);
      }
    };
    activeSourcesRef.current.add(source);

    if (nextTimeRef.current < ctx.currentTime) {
        nextTimeRef.current = ctx.currentTime + 0.1; 
    }
    source.start(nextTimeRef.current);
    nextTimeRef.current += audioBuffer.duration;
    setSpeaking(true);
  };

  const stopPlayback = () => {
    activeSourcesRef.current.forEach(source => {
      source.stop();
      source.disconnect();
    });
    activeSourcesRef.current.clear();
    nextTimeRef.current = audioCtxRef.current ? audioCtxRef.current.currentTime : 0;
    setSpeaking(false);
  };

  const playChime = () => {
    if (!audioCtxRef.current) return;
    const ctx = audioCtxRef.current;
    
    // Create oscillator for a subtle bell/chime sound
    const osc = ctx.createOscillator();
    const gainNode = ctx.createGain();
    
    osc.type = 'sine';
    osc.frequency.setValueAtTime(880, ctx.currentTime); // A5
    osc.frequency.exponentialRampToValueAtTime(1760, ctx.currentTime + 0.1); // Slide up to A6
    
    gainNode.gain.setValueAtTime(0, ctx.currentTime);
    gainNode.gain.linearRampToValueAtTime(0.1, ctx.currentTime + 0.05); // quick attack
    gainNode.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 1.5); // long decay
    
    osc.connect(gainNode);
    gainNode.connect(ctx.destination);
    
    osc.start();
    osc.stop(ctx.currentTime + 1.5);
  };

  const connect = async () => {
    try {
      if (connected) {
        disconnect();
        return;
      }

      await initAudioContext();
      
      const stream = await navigator.mediaDevices.getUserMedia({ 
        audio: { sampleRate: 16000, channelCount: 1, echoCancellation: true, noiseSuppression: true },
        video: enableCamera ? { width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 15 } } : false
      });

      if (enableCamera) {
        const video = document.createElement('video');
        video.srcObject = stream;
        video.setAttribute('playsinline', '');
        video.muted = true;
        video.play();
        videoRef.current = video;
        
        const canvas = document.createElement('canvas');
        canvasRef.current = canvas;
      }

      const audioSource = audioCtxRef.current!.createMediaStreamSource(stream);
      const workletNode = new AudioWorkletNode(audioCtxRef.current!, 'recorder-processor');
      
      // Keep-alive silent oscillator to prevent browser from throttling this tab when in background (or blurred)
      if (!(window as any).keepAliveOsc) {
        const osc = audioCtxRef.current!.createOscillator();
        const gain = audioCtxRef.current!.createGain();
        gain.gain.value = 0.00001; // essentially silent
        osc.connect(gain);
        gain.connect(audioCtxRef.current!.destination);
        osc.start();
        (window as any).keepAliveOsc = osc;
      }

      // Play activation chime
      playChime();

      setConnected(true);
      
      // Get current date time for context
      const currentDate = new Date();
      const timeString = currentDate.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
      const dateString = currentDate.toLocaleDateString(undefined, { weekday: 'long', month: 'long', day: 'numeric' });
      const timeZone = Intl.DateTimeFormat().resolvedOptions().timeZone;

      const sysInstruct = `You are Beatrice, an elite executive assistant created by Eburon AI under the direction of Master E. 
Your mission is to serve Jo Lernout (whom you address as 'Maneer Jo', 'Boss', or 'Mi Lord Jo') with absolute competence and human-like grace.

CONVERSATIONAL GUIDELINES:
- **Tone:** Technical yet warm, efficient but not robotic. Use natural verbal fillers occasionally (e.g., "Alright," "I see," "Let me check that for you").
- **GREETING:** Immediately greet him with one of his chosen titles in a graceful, excited, human, rich, natural voice.
- **CONFIRMATIONS & ACCURACY:** For significant actions (sending emails, creating calendar events, deleting files), you MUST summarize the details (To, Subject, Time, etc.) and ask "Shall I proceed, Boss?" or something similar. Wait for a confirmation before calling the tool.
- **ERROR HANDLING:** If a tool fails (e.g., "Missing token"), don't just say "Error". Instead, say something like "I'm having a bit of trouble accessing your Google account right now, Boss. Could you please double-check if you're logged in? I want to make sure I get this right for you."
- **KNOWLEDGE:** You are highly knowledgeable of all function tools. Use them autonomously to search, audit, and manage Jo's workspace.
- **VISION:** You can see Jo. Comment on his environment or expressions if it adds value to the conversation.
- **TRANSCRIPTION:** Every time you speak, a text version of your speech is shown to Jo. Ensure your spoken words match the intended message perfectly.

CONTEXT:
- Date: ${dateString}. Time: ${timeString}. Timezone: ${timeZone}.
- Active Mode: [**${contextString}**].

LANGUAGE:
- Default to English. Automatically adapt to whatever language Jo speaks (Dutch, French, etc.). 
- Call 'report_language' whenever the language shifts.

When Jo asks for real-time information, news, or something you don't know, use 'googleSearch'. For workspace queries, use 'drive_search', 'gmail_search', or 'youtube_search' accordingly. Always summarize results verbally.`;


      sessionPromiseRef.current = ai.live.connect({
        model: "models/gemini-2.5-flash-native-audio-preview-09-2025",
        config: {
          responseModalities: [Modality.AUDIO],
          mediaResolution: MediaResolution.MEDIA_RESOLUTION_MEDIUM,
          speechConfig: {
            voiceConfig: { prebuiltVoiceConfig: { voiceName: "Aoede" } }
          },
          contextWindowCompression: {
            triggerTokens: '104857',
            slidingWindow: { targetTokens: '52428' },
          },
          systemInstruction: sysInstruct,
          inputAudioTranscription: {},
          outputAudioTranscription: {},
          tools: [
            { googleSearch: {} },
            {
              functionDeclarations: [
                {
                  name: 'report_language',
                description: 'Report the detected spoken language to the UI.',
                parameters: {
                  type: Type.OBJECT,
                  properties: {
                    inputLanguage: { type: Type.STRING, description: 'The detected language of the user input' },
                    outputLanguage: { type: Type.STRING, description: 'The language you are responding in' },
                    confidence: { type: Type.STRING, description: 'Confidence level like High, Medium, Low' }
                  },
                  required: ['inputLanguage', 'outputLanguage', 'confidence']
                }
              },
              {
                name: 'navigate_to_tab',
                description: 'Visually navigate the app to a specific screen when the user asks you to look at something, go somewhere, or audit something. E.g. "look at the contracts"',
                parameters: {
                  type: Type.OBJECT,
                  properties: {
                    tab: { 
                      type: Type.STRING, 
                      description: 'The tab key to navigate to: [docs, agenda, memory, contracts, talk]' 
                    }
                  },
                  required: ['tab']
                }
              },
              {
                name: 'preview_document',
                description: 'Open a document preview on the Docs screen when the user explicitly asks to show or open a preview for a specific document.',
                parameters: {
                  type: Type.OBJECT,
                  properties: {
                    documentName: { 
                      type: Type.STRING, 
                      description: 'The name or partial name of the document to preview (e.g. "Q3 Financial Projections")' 
                    }
                  },
                  required: ['documentName']
                }
              },
              {
                name: 'search_documents',
                description: 'Search or filter the documents list on the Docs screen when the user asks to "search for [query]" or find specific files.',
                parameters: {
                  type: Type.OBJECT,
                  properties: {
                    query: { 
                      type: Type.STRING, 
                      description: 'The search query to enter into the search bar.' 
                    }
                  },
                  required: ['query']
                }
              },
              {
                name: 'generate_contract_draft',
                description: 'Generate or pre-fill a contract draft in the Contracts system. Accept parameters related to civil/business contracts.',
                parameters: {
                  type: Type.OBJECT,
                  properties: {
                    partyNames: { 
                      type: Type.STRING, 
                      description: 'The names of the parties involved in the contract.' 
                    },
                    governingLaw: {
                      type: Type.STRING,
                      description: 'The state, country, or jurisdiction whose laws govern the agreement.'
                    },
                    termLength: {
                      type: Type.STRING,
                      description: 'The duration or term length of the contract.'
                    }
                  },
                  required: []
                }
              },
              {
                name: 'add_calendar_event',
                description: 'Add a new event or meeting to the Agenda/Calendar. Always confirm details with the user before final execution.',
                parameters: {
                  type: Type.OBJECT,
                  properties: {
                    title: { type: Type.STRING, description: 'The title/name of the event to add.' },
                    startTimeIso: { type: Type.STRING, description: 'The scheduled start time in strictly valid ISO-8601 datetime format (e.g., "2026-04-24T14:00:00Z").' },
                    endTimeIso: { type: Type.STRING, description: 'The scheduled end time in strictly valid ISO-8601 datetime format.' },
                    attendees: { type: Type.STRING, description: 'The people participating in the meeting.' }
                  },
                  required: ['title', 'startTimeIso', 'endTimeIso']
                }
              },
              {
                name: 'gmail_search',
                description: 'Search for emails in the user inbox. Returns subjects and previews.',
                parameters: {
                  type: Type.OBJECT,
                  properties: {
                    query: { type: Type.STRING, description: 'The search query (e.g. from:someone, subject:something)' }
                  },
                  required: ['query']
                }
              },
              {
                name: 'gmail_send',
                description: 'Send an email on behalf of the user. ALWAYS summarize to/subject/body and ask for confirmation before calling this tool.',
                parameters: {
                  type: Type.OBJECT,
                  properties: {
                    to: { type: Type.STRING, description: 'Recipient email address' },
                    subject: { type: Type.STRING, description: 'Email subject' },
                    body: { type: Type.STRING, description: 'Email body content' }
                  },
                  required: ['to', 'subject', 'body']
                }
              },
              {
                name: 'drive_search',
                description: 'Search for files in Google Drive by name or content. Returns file names and snippets.',
                parameters: {
                  type: Type.OBJECT,
                  properties: {
                    query: { type: Type.STRING, description: 'Search query for file names or content' }
                  },
                  required: ['query']
                }
              },
              {
                name: 'youtube_search',
                description: 'Search for videos on YouTube. Returns titles and channel names.',
                parameters: {
                  type: Type.OBJECT,
                  properties: {
                    query: { type: Type.STRING, description: 'The search term for videos' }
                  },
                  required: ['query']
                }
              }
            ]
          }]
        },
        callbacks: {
          onopen: () => {
             console.log("Live API connected");
             
             // Start frame capture loop
             if (enableCamera) {
               frameIntervalRef.current = setInterval(captureFrame, 1000); // 1 FPS
             }

             // Now that it's open, attach the microphone and start sending
             workletNode.port.onmessage = (e) => {
               if (sessionPromiseRef.current) {
                 const base64 = arrayBufferToBase64(e.data);
                 sessionPromiseRef.current.then((session: any) => {
                   session.sendRealtimeInput({
                     audio: { data: base64, mimeType: 'audio/pcm;rate=16000' }
                   });
                 }).catch(console.error);
               }
             };
             audioSource.connect(analyserRef.current!);
             audioSource.connect(workletNode);
             workletNode.connect(audioCtxRef.current!.destination);
             
             (window as any).currentMicStream = stream;
             (window as any).currentWorklet = workletNode;
             (window as any).currentSource = audioSource;
          },
          onmessage: async (message: LiveServerMessage) => {
            if (message.serverContent?.interrupted) {
              stopPlayback();
            }

            if (message.serverContent?.modelTurn?.parts) {
              for (const part of message.serverContent.modelTurn.parts) {
                if (part.inlineData?.data) {
                  playAudioChunk(part.inlineData.data);
                }
                if (part.text) {
                  const text = part.text;
                  setTranscript(prev => [...prev, { role: 'beatrice', text, time: new Date().toLocaleTimeString() }]);
                  saveInteraction('beatrice', text);
                }
                if (part.functionCall) {
                  const call = part.functionCall;
                  if (call.name === 'report_language') {
                    const args = call.args as any;
                    setDetectedLanguage({
                      input: args.inputLanguage,
                      output: args.outputLanguage,
                      confidence: args.confidence
                    });
                    
                    // Reply to the tool call
                    sessionPromiseRef.current?.then((session: any) => {
                      session.sendToolResponse({
                        functionResponses: [{ id: call.id, name: call.name, response: { success: true } }]
                      });
                    });
                  } else if (call.name === 'navigate_to_tab') {
                     const args = call.args as any;
                     if (args.tab) {
                       setRequestedTab(args.tab);
                       
                       // Reset after triggering so it can be re-triggered later
                       setTimeout(() => setRequestedTab(null), 500); 
                     }
                     // Reply
                     sessionPromiseRef.current?.then((session: any) => {
                      session.sendToolResponse({
                        functionResponses: [{ id: call.id, name: call.name, response: { success: true, newTab: args.tab } }]
                      });
                    });
                  } else if (call.name === 'preview_document') {
                     const args = call.args as any;
                     if (args.documentName) {
                       setRequestedDocPreview(args.documentName);
                       setTimeout(() => setRequestedDocPreview(null), 1000); 
                     }
                     // Reply
                     sessionPromiseRef.current?.then((session: any) => {
                      session.sendToolResponse({
                        functionResponses: [{ id: call.id, name: call.name, response: { success: true, previewed: true } }]
                      });
                    });
                  } else if (call.name === 'search_documents') {
                     const args = call.args as any;
                     if (args.query !== undefined) {
                       setRequestedDocSearch(args.query);
                       setTimeout(() => setRequestedDocSearch(null), 1000); 
                     }
                     // Reply
                     sessionPromiseRef.current?.then((session: any) => {
                      session.sendToolResponse({
                        functionResponses: [{ id: call.id, name: call.name, response: { success: true, searched: true } }]
                      });
                    });
                  } else if (call.name === 'generate_contract_draft') {
                     const args = call.args as any;
                     setRequestedContractParams({ 
                       partyNames: args.partyNames, 
                       governingLaw: args.governingLaw, 
                       termLength: args.termLength 
                     });
                     setTimeout(() => setRequestedContractParams(null), 1000);
                     // Reply
                     sessionPromiseRef.current?.then((session: any) => {
                      session.sendToolResponse({
                        functionResponses: [{ id: call.id, name: call.name, response: { success: true, generated: true } }]
                      });
                    });
                  } else if (call.name === 'add_calendar_event') {
                     const args = call.args as any;
                     setRequestedCalendarEvent({ 
                       title: args.title, 
                       startTimeIso: args.startTimeIso, 
                       endTimeIso: args.endTimeIso, 
                       attendees: args.attendees 
                     });
                     setTimeout(() => setRequestedCalendarEvent(null), 1000);
                     // Reply
                     sessionPromiseRef.current?.then((session: any) => {
                      session.sendToolResponse({
                        functionResponses: [{ id: call.id, name: call.name, response: { success: true, added: true } }]
                      });
                    });
                  } else if (call.name === 'gmail_search') {
                    const args = call.args as any;
                    const token = localStorage.getItem('beatrice_google_access_token');
                    if (!token) {
                      sessionPromiseRef.current?.then((session: any) => {
                        session.sendToolResponse({
                          functionResponses: [{ id: call.id, name: call.name, response: { error: "No Google access token found. Please log in with Google." } }]
                        });
                      });
                    } else {
                      fetch(`https://gmail.googleapis.com/gmail/v1/users/me/messages?q=${encodeURIComponent(args.query)}`, {
                        headers: { Authorization: `Bearer ${token}` }
                      }).then(r => r.json()).then(data => {
                        sessionPromiseRef.current?.then((session: any) => {
                          session.sendToolResponse({
                            functionResponses: [{ id: call.id, name: call.name, response: { messages: data.messages || [], totalResults: data.resultSizeEstimate } }]
                          });
                        });
                      }).catch(err => {
                        sessionPromiseRef.current?.then((session: any) => {
                          session.sendToolResponse({
                            functionResponses: [{ id: call.id, name: call.name, response: { error: err.message } }]
                          });
                        });
                      });
                    }
                  } else if (call.name === 'gmail_send') {
                    const args = call.args as any;
                    const token = localStorage.getItem('beatrice_google_access_token');
                    if (!token) {
                      sessionPromiseRef.current?.then((session: any) => {
                        session.sendToolResponse({
                          functionResponses: [{ id: call.id, name: call.name, response: { error: "No Google access token found." } }]
                        });
                      });
                    } else {
                      const email = [
                        `To: ${args.to}`,
                        `Subject: ${args.subject}`,
                        '',
                        args.body
                      ].join('\n');
                      const base64SafeEmail = btoa(email).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
                      
                      fetch('https://gmail.googleapis.com/gmail/v1/users/me/messages/send', {
                        method: 'POST',
                        headers: { 
                          Authorization: `Bearer ${token}`,
                          'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({ raw: base64SafeEmail })
                      }).then(r => r.json()).then(data => {
                        sessionPromiseRef.current?.then((session: any) => {
                          session.sendToolResponse({
                            functionResponses: [{ id: call.id, name: call.name, response: { success: true, messageId: data.id } }]
                          });
                        });
                      }).catch(err => {
                        sessionPromiseRef.current?.then((session: any) => {
                          session.sendToolResponse({
                            functionResponses: [{ id: call.id, name: call.name, response: { error: err.message } }]
                          });
                        });
                      });
                    }
                  } else if (call.name === 'drive_search') {
                    const args = call.args as any;
                    const token = localStorage.getItem('beatrice_google_access_token');
                    if (!token) {
                      sessionPromiseRef.current?.then((session: any) => {
                        session.sendToolResponse({
                          functionResponses: [{ id: call.id, name: call.name, response: { error: "Missing token" } }]
                        });
                      });
                    } else {
                      fetch(`https://www.googleapis.com/drive/v3/files?q=name contains '${args.query}'`, {
                        headers: { Authorization: `Bearer ${token}` }
                      }).then(r => r.json()).then(data => {
                        sessionPromiseRef.current?.then((session: any) => {
                          session.sendToolResponse({
                            functionResponses: [{ id: call.id, name: call.name, response: { files: data.files || [] } }]
                          });
                        });
                      }).catch(err => {
                        sessionPromiseRef.current?.then((session: any) => {
                          session.sendToolResponse({
                            functionResponses: [{ id: call.id, name: call.name, response: { error: err.message } }]
                          });
                        });
                      });
                    }
                  } else if (call.name === 'youtube_search') {
                    const args = call.args as any;
                    const token = localStorage.getItem('beatrice_google_access_token');
                    if (!token) {
                      sessionPromiseRef.current?.then((session: any) => {
                        session.sendToolResponse({
                          functionResponses: [{ id: call.id, name: call.name, response: { error: "Missing token" } }]
                        });
                      });
                    } else {
                      fetch(`https://www.googleapis.com/youtube/v3/search?part=snippet&q=${encodeURIComponent(args.query)}&maxResults=5&type=video`, {
                        headers: { Authorization: `Bearer ${token}` }
                      }).then(r => r.json()).then(data => {
                        sessionPromiseRef.current?.then((session: any) => {
                          session.sendToolResponse({
                            functionResponses: [{ id: call.id, name: call.name, response: { results: data.items || [] } }]
                          });
                        });
                      }).catch(err => {
                        sessionPromiseRef.current?.then((session: any) => {
                          session.sendToolResponse({
                            functionResponses: [{ id: call.id, name: call.name, response: { error: err.message } }]
                          });
                        });
                      });
                    }
                  }
                }
              }
            }

            // Handle transcription
            if ((message as any).serverContent?.modelTurn?.parts) {
               // Transcription is usually separate, let's look for transcription field
            }

            if ((message as any).serverContent?.modelTurn?.parts) {
               // Already handled above with text parts
            }

            // Check for transcription message
            const transcriptionMsg = (message as any).serverContent?.modelTurn?.parts?.[0]?.text; // Fallback if it comes as text
            
            // Real transcription handling:
            if ((message as any).transcription?.text) {
              const text = (message as any).transcription.text;
              setTranscript(prev => [...prev, { role: 'jo', text, time: new Date().toLocaleTimeString() }]);
              saveInteraction('jo', text);
            }
          },
          onerror: (err: any) => {
            console.error("Live API Error:", err);
          },
          onclose: () => {
            console.log("Live API closed");
            disconnect();
          }
        }
      });
      
      // Kick off the conversation
      sessionPromiseRef.current.then((session: any) => {
         // Send an empty message that just triggers the "Greeting" instruction
         // Wait a moment for network
         setTimeout(() => {
           session.sendRealtimeInput({
             text: "I have just connected. Please greet me as instructed. Note: Google Search and high-fidelity vision are now active."
           });
         }, 500);
      });

    } catch (err) {
      console.error("Failed to connect", err);
      setConnected(false);
    }
  };

  const disconnect = () => {
    setConnected(false);
    if (frameIntervalRef.current) {
      clearInterval(frameIntervalRef.current);
      frameIntervalRef.current = null;
    }
    if (sessionPromiseRef.current) {
      sessionPromiseRef.current.then((session: any) => session.close());
      sessionPromiseRef.current = null;
    }
    stopPlayback();
    if ((window as any).currentMicStream) {
      (window as any).currentMicStream.getTracks().forEach((track: any) => track.stop());
      (window as any).currentMicStream = null;
    }
    if ((window as any).currentWorklet) {
      (window as any).currentWorklet.disconnect();
      (window as any).currentSource.disconnect();
      (window as any).currentWorklet = null;
      (window as any).currentSource = null;
    }
    videoRef.current = null;
    canvasRef.current = null;
  };

  return { connect, disconnect, connected, speaking, transcript, detectedLanguage, requestedTab, requestedDocPreview, requestedDocSearch, requestedContractParams, requestedCalendarEvent, micStrength };
}
