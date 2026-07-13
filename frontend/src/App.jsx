import { useState, useRef, useEffect, useCallback } from 'react'
import './index.css'
import Login from './Login'
import { saveSessions, loadSessions } from './firestore'

const GROQ_API_KEY = import.meta.env.VITE_GROQ_API_KEY
const GROQ_URL = 'https://api.groq.com/openai/v1/chat/completions'
const MAX_FILE_SIZE = 10 * 1024 * 1024 // 10MB

const GREETINGS = [
  (name) => `Hi ${name}! Ready to explore something new today?`,
  (name) => `Welcome back, ${name}! What's on your mind?`,
  (name) => `Hey ${name}! Ask me anything — I'm all yours.`,
  (name) => `Good to see you, ${name}! Let's figure something out together.`,
  (name) => `Hello ${name}! What shall we talk about?`,
  (name) => `Hi ${name}! Got a question, a curiosity, or a wild idea? Let's go.`,
  (name) => `Hey there, ${name}! Science, stories, or code — I'm ready.`,
  (name) => `Welcome, ${name}! What would you like to learn today?`,
]

function getSessionKey(email) {
  return `nanosage_sessions_${email}`
}

function TypingDots() {
  return (
    <div className="flex items-center gap-1.5 py-2 px-1">
      <span className="dot-1 w-2 h-2 rounded-full block" />
      <span className="dot-2 w-2 h-2 rounded-full block" />
      <span className="dot-3 w-2 h-2 rounded-full block" />
    </div>
  )
}

function ChatMessage({ msg }) {
  const isUser = msg.role === 'user'
  const isThinking = !isUser && msg.content === '' && msg.streaming
  return (
    <div className={`msg-enter flex w-full gap-3 ${isUser ? 'justify-end' : 'justify-start'} mb-6`}>
      <div className={`flex flex-col max-w-[85%] sm:max-w-[75%] ${isUser ? 'items-end' : 'items-start'}`}>
        {/* Show image thumbnail if the message has an image */}
        {msg.imagePreview && (
          <div className="image-preview-bubble mb-2 rounded-xl overflow-hidden" style={{maxWidth: '240px'}}>
            <img src={msg.imagePreview} alt="Uploaded" className="w-full h-auto rounded-xl" style={{display:'block'}} />
          </div>
        )}
        {isUser ? (
          <div className="user-bubble px-4 py-3 rounded-2xl rounded-tr-sm text-sm text-white leading-relaxed break-words">
            {msg.displayContent || msg.content}
          </div>
        ) : (
          <div className="ai-card px-4 py-3 rounded-xl text-sm leading-relaxed break-words">
            {isThinking ? <TypingDots /> : (
              <>
                <span className="whitespace-pre-wrap">{(msg.content || '…').trimEnd()}</span>
                {msg.streaming && <span className="cursor-blink inline-block w-0.5 h-4 ml-1 align-middle" style={{background:'#c9956b'}} />}
              </>
            )}
          </div>
        )}
        {msg.timestamp && (
          <span className="text-[10px] mt-1 px-1" style={{color:'rgba(212,165,116,0.3)'}}>
            {msg.timestamp}
          </span>
        )}
      </div>
    </div>
  )
}

function WelcomeScreen({ greeting }) {
  return (
    <div className="flex-1 flex flex-col items-center justify-center text-center p-6 max-w-lg mx-auto h-full my-auto">
      <img src="/logo.png" className="w-40 h-40" style={{objectFit:"contain", filter:"drop-shadow(0 0 16px rgba(181,98,106,0.5))", marginBottom:"-16px"}} alt="NanoSage" />
      <h2 className="text-2xl font-bold tracking-tight mb-3" style={{color:'#f5e6d8'}}>{greeting}</h2>
      <p className="text-sm" style={{color:'rgba(212,165,116,0.4)'}}>Type a message below to get started.</p>
    </div>
  )
}

function buildPrompt(message, history) {
  let prompt = "### System:\nYou are NanoSage, a helpful AI assistant.\n\n"
  if (history) {
    for (const turn of history) {
      if (turn.user && turn.assistant) {
        prompt += `### Instruction:\n${turn.user}\n\n### Response:\n${turn.assistant}\n\n`
      }
    }
  }
  prompt += `### Instruction:\n${message}\n\n### Response:\n`
  return prompt
}

// ── PDF text extraction using PDF.js ──
async function extractPdfText(file) {
  const pdfjsLib = window.pdfjsLib
  if (!pdfjsLib) throw new Error('PDF.js not loaded. Please refresh the page.')

  const arrayBuffer = await file.arrayBuffer()
  const pdf = await pdfjsLib.getDocument({ data: arrayBuffer }).promise
  let fullText = ''

  for (let i = 1; i <= pdf.numPages; i++) {
    const page = await pdf.getPage(i)
    const content = await page.getTextContent()
    const strings = content.items.map(item => item.str)
    fullText += strings.join(' ') + '\n'
  }

  return fullText.trim()
}

// ── Convert image file to base64 data URL ──
function fileToBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => resolve(reader.result)
    reader.onerror = reject
    reader.readAsDataURL(file)
  })
}

export default function App() {
  const [user, setUser] = useState(() => {
    try { return JSON.parse(localStorage.getItem('nanosage_user') || 'null') }
    catch { return null }
  })

  const greetingIdx = useRef(Math.floor(Math.random() * GREETINGS.length))
  const [sessions, setSessions] = useState([])
  const [activeSession, setActiveSession] = useState(null)
  const [isGenerating, setIsGenerating] = useState(false)
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [input, setInput] = useState('')

  // Voice input state
  const [isRecording, setIsRecording] = useState(false)
  const recognitionRef = useRef(null)

  // PDF upload state
  const [pdfContext, setPdfContext] = useState(null) // { name, text }
  const [pdfLoading, setPdfLoading] = useState(false)

  // Image upload state
  const [imageAttachment, setImageAttachment] = useState(null) // { name, base64, preview }

  // Error toast state
  const [errorMsg, setErrorMsg] = useState('')
  const errorTimeout = useRef(null)

  const messagesEndRef = useRef(null)
  const abortRef = useRef(null)
  const textareaRef = useRef(null)
  const fileInputRef = useRef(null)

  const messages = activeSession?.messages || []

  // ── Show error toast ──
  const showError = useCallback((msg) => {
    setErrorMsg(msg)
    if (errorTimeout.current) clearTimeout(errorTimeout.current)
    errorTimeout.current = setTimeout(() => setErrorMsg(''), 5000)
  }, [])

  useEffect(() => {
    if (!user?.email) { setSessions([]); setActiveSession(null); return }
    isLoadingRef.current = true
    const load = async () => {
      try {
        const cloud = await loadSessions(user.email)
        const saved = localStorage.getItem(getSessionKey(user.email))
        const local = saved ? JSON.parse(saved) : []
        if (cloud.length > 0) {
          const merged = [...cloud]
          for (const localSession of local) {
            const cloudIdx = merged.findIndex(s => s.id === localSession.id)
            if (cloudIdx === -1) {
              merged.push(localSession)
            } else if (localSession.messages.length > merged[cloudIdx].messages.length) {
              merged[cloudIdx] = localSession
            }
          }
          merged.sort((a, b) => Number(b.id) - Number(a.id))
          setSessions(merged)
        } else {
          setSessions(local)
        }
      } catch { setSessions([]) }
      finally { isLoadingRef.current = false }
    }
    load()
    setActiveSession(null)
  }, [user?.email])

  const isLoadingRef = useRef(true)
  useEffect(() => {
    if (!user?.email) return
    if (isLoadingRef.current) return
    localStorage.setItem(getSessionKey(user.email), JSON.stringify(sessions))
    saveSessions(user.email, sessions)
  }, [sessions, user?.email])

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [])

  useEffect(() => { scrollToBottom() }, [messages, scrollToBottom])

  const handleLogin = (userData) => {
    localStorage.setItem('nanosage_user', JSON.stringify(userData))
    setUser(userData)
  }

  const handleLogout = () => {
    localStorage.removeItem('nanosage_user')
    setUser(null)
    setSessions([])
    setActiveSession(null)
  }

  const updateSession = (sid, newMsgs, title) => {
    setSessions(prev => prev.map(s =>
      s.id === sid ? { ...s, messages: newMsgs, title: title || s.title } : s
    ))
  }

  // ── Voice Input ──
  const startRecording = () => {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition
    if (!SpeechRecognition) {
      showError('Your browser does not support voice input. Try Chrome or Edge.')
      return
    }

    const recognition = new SpeechRecognition()
    recognition.lang = 'en-US'
    recognition.interimResults = false
    recognition.maxAlternatives = 1
    recognition.continuous = false

    recognition.onresult = (event) => {
      const transcript = event.results[0][0].transcript
      setInput(prev => prev ? prev + ' ' + transcript : transcript)
    }

    recognition.onerror = (event) => {
      if (event.error !== 'aborted') {
        showError(`Voice error: ${event.error}`)
      }
      setIsRecording(false)
    }

    recognition.onend = () => {
      setIsRecording(false)
    }

    recognitionRef.current = recognition
    recognition.start()
    setIsRecording(true)
  }

  const stopRecording = () => {
    if (recognitionRef.current) {
      recognitionRef.current.stop()
      recognitionRef.current = null
    }
    setIsRecording(false)
  }

  const toggleRecording = () => {
    if (isRecording) stopRecording()
    else startRecording()
  }

  // ── File Upload Handler ──
  const handleFileUpload = async (e) => {
    const file = e.target.files?.[0]
    if (!file) return
    // Reset file input so re-selecting the same file triggers onChange
    e.target.value = ''

    if (file.size > MAX_FILE_SIZE) {
      showError(`File is too large (${(file.size / 1024 / 1024).toFixed(1)}MB). Maximum is 10MB.`)
      return
    }

    const ext = file.name.toLowerCase().split('.').pop()

    if (ext === 'pdf') {
      try {
        setPdfLoading(true)
        const text = await extractPdfText(file)
        if (!text) {
          showError('Could not extract text from this PDF. It may be scanned/image-based.')
          return
        }
        setPdfContext({ name: file.name, text })
        // Clear any image attachment when PDF is loaded
        setImageAttachment(null)
      } catch (err) {
        showError(`PDF error: ${err.message}`)
      } finally {
        setPdfLoading(false)
      }
    } else if (['jpg', 'jpeg', 'png', 'webp'].includes(ext)) {
      try {
        const base64 = await fileToBase64(file)
        setImageAttachment({ name: file.name, base64, preview: base64 })
        // Clear PDF context when image is loaded
        setPdfContext(null)
      } catch (err) {
        showError(`Image error: ${err.message}`)
      }
    } else {
      showError('Unsupported file type. Please upload a PDF, JPG, PNG, or WebP file.')
    }
  }

  // ── Send Message (updated for PDF + Image) ──
  const sendMessage = async (text) => {
    if (!text.trim() || isGenerating) return

    const now = () => new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })

    // Determine what we're sending
    const hasPdf = !!pdfContext
    const hasImage = !!imageAttachment
    const currentPdfContext = pdfContext
    const currentImageAttachment = imageAttachment

    // Clear attachments after capturing
    setPdfContext(null)
    setImageAttachment(null)

    let session = activeSession
    if (!session) {
      const newSession = {
        id: Date.now().toString(),
        title: text.slice(0, 30) || 'New Chat',
        messages: []
      }
      setSessions(prev => [newSession, ...prev])
      setActiveSession(newSession)
      session = newSession
    }

    // Build the actual content to send to the API
    let apiContent = text
    if (hasPdf) {
      apiContent = `Here is a PDF document:\n${currentPdfContext.text}\n\nNow answer: ${text}`
    }

    // For display: show the user's typed text (not the full PDF dump)
    const displayContent = hasPdf ? `📄 ${currentPdfContext.name}\n${text}` : text

    const userMsg = {
      id: Date.now(),
      role: 'user',
      content: apiContent,
      displayContent: displayContent,
      timestamp: now(),
      imagePreview: hasImage ? currentImageAttachment.preview : null,
    }
    const aiMsgId = Date.now() + 1
    const aiMsg = { id: aiMsgId, role: 'assistant', content: '', streaming: true, timestamp: null }
    const withUser = [...session.messages, userMsg, aiMsg]

    const updatedSession = { ...session, messages: withUser }
    setActiveSession(updatedSession)
    setIsGenerating(true)

    const historyPairs = []
    for (let i = 0; i < session.messages.length - 1; i++) {
      const cur = session.messages[i], nxt = session.messages[i + 1]
      if (cur.role === 'user' && nxt?.role === 'assistant' && nxt.content) {
        historyPairs.push({ user: cur.content, assistant: nxt.content })
      }
    }

    try {
      const controller = new AbortController()
      abortRef.current = controller

      // Build messages
      let apiMessages
      if (hasImage) {
        const base64Data = currentImageAttachment.base64
        const mediaType = base64Data.split(';')[0].split(':')[1]
        const base64Only = base64Data.split(',')[1]
        apiMessages = [
          { role: 'system', content: `You are NanoSage, a helpful AI assistant. Current date: ${new Date().toLocaleString()}.` },
          ...historyPairs.slice(-3).flatMap(h => [
            { role: 'user', content: h.user },
            { role: 'assistant', content: h.assistant }
          ]),
          { role: 'user', content: [
            { type: 'image_url', image_url: { url: `data:${mediaType};base64,${base64Only}` } },
            { type: 'text', text: text }
          ]}
        ]
      } else {
        apiMessages = [
          { role: 'system', content: `You are NanoSage, a helpful AI assistant. Current date: ${new Date().toLocaleString()}.` },
          ...historyPairs.slice(-3).flatMap(h => [
            { role: 'user', content: h.user },
            { role: 'assistant', content: h.assistant }
          ]),
          { role: 'user', content: apiContent }
        ]
      }

      const response = await fetch(GROQ_URL, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${GROQ_API_KEY}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model: hasImage ? 'meta-llama/llama-4-scout-17b-16e-instruct' : 'llama-3.1-8b-instant',
          messages: apiMessages,
          max_tokens: 1024,
          temperature: 0.7,
        }),
        signal: controller.signal,
      })

      if (!response.ok) throw new Error(`Groq API error: ${response.status}`)

      const groqData = await response.json()
      let accumulated = groqData.choices?.[0]?.message?.content || '(no response)'

      // Clean up response — remove echoed instruction
      if (accumulated.includes('### Response:')) {
        accumulated = accumulated.split('### Response:').pop().trim()
      }
      // Remove trailing incomplete sentences
      accumulated = accumulated.trim()

      setActiveSession(prev => {
        if (!prev) return prev
        return {
          ...prev,
          messages: prev.messages.map(m =>
            m.id === aiMsgId ? { ...m, content: accumulated, streaming: true } : m
          )
        }
      })

      const finalMsgs = withUser.map(m =>
        m.id === aiMsgId ? { ...m, content: accumulated || '(no response)', streaming: false, timestamp: now() } : m
      )
      const finalSession = { ...session, messages: finalMsgs }
      setActiveSession(finalSession)
      updateSession(session.id, finalMsgs, session.title)

    } catch (err) {
      const errMsg = err.name === 'AbortError' ? '(generation stopped)' : `⚠️ Error: ${err.message}`
      const finalMsgs = withUser.map(m =>
        m.id === aiMsgId ? { ...m, content: errMsg, streaming: false, timestamp: '—' } : m
      )
      setActiveSession(prev => prev ? { ...prev, messages: finalMsgs } : prev)
      updateSession(session.id, finalMsgs)
    } finally {
      setIsGenerating(false)
      abortRef.current = null
    }
  }

  const handleNewChat = () => {
    abortRef.current?.abort()
    setActiveSession(null)
    setIsGenerating(false)
    setPdfContext(null)
    setImageAttachment(null)
  }

  const selectSession = (s) => {
    if (isGenerating) return
    setActiveSession(s)
    setSidebarOpen(false)
  }

  const deleteSession = (id, e) => {
    e.stopPropagation()
    if (isGenerating && activeSession?.id === id) { abortRef.current?.abort(); setIsGenerating(false) }
    setSessions(prev => prev.filter(s => s.id !== id))
    if (activeSession?.id === id) setActiveSession(null)
  }

  const clearChat = () => {
    abortRef.current?.abort()
    setIsGenerating(false)
    if (activeSession) {
      const cleared = { ...activeSession, messages: [] }
      setActiveSession(cleared)
      updateSession(activeSession.id, [])
    }
  }

  const submit = () => {
    const t = input.trim()
    if (!t || isGenerating) return
    sendMessage(t); setInput('')
    if (textareaRef.current) textareaRef.current.style.height = 'auto'
  }

  const handleKey = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); submit() }
  }

  const handleChange = (e) => {
    setInput(e.target.value)
    const el = textareaRef.current
    if (el) { el.style.height = 'auto'; el.style.height = Math.min(el.scrollHeight, 160) + 'px' }
  }

  if (!user) return <Login onLogin={handleLogin} />

  const greeting = GREETINGS[greetingIdx.current](user.name?.split(' ')[0] || 'there')

  return (
    <div className="app-root flex h-screen w-full overflow-hidden">
      <div className="orb orb1" /><div className="orb orb2" /><div className="orb orb3" /><div className="orb orb4" />

      <aside className={`sidebar fixed md:static inset-y-0 left-0 z-30 w-64 transform transition-transform duration-300 ease-in-out md:translate-x-0 ${sidebarOpen ? 'translate-x-0' : '-translate-x-full'} flex flex-col`}>
        <div className="p-3 sidebar-header">
          <button onClick={() => { handleNewChat(); setSidebarOpen(false) }} className="new-chat-btn w-full flex items-center justify-center gap-2 px-4 py-2.5 rounded-lg text-sm font-medium transition-all duration-200 cursor-pointer">
            <svg xmlns="http://www.w3.org/2000/svg" className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2"><path strokeLinecap="round" strokeLinejoin="round" d="M12 4v16m8-8H4" /></svg>
            New Chat
          </button>
        </div>

        <div className="flex-1 overflow-y-auto p-2 space-y-1">
          {sessions.length === 0 ? (
            <div className="text-xs text-center py-8" style={{color:'rgba(212,165,116,0.3)'}}>No chat history</div>
          ) : sessions.map(s => (
            <div key={s.id} onClick={() => selectSession(s)}
              className={`session-item group flex items-center justify-between px-3 py-2 rounded-lg cursor-pointer transition-all ${activeSession?.id === s.id ? 'session-active' : ''}`}>
              <div className="flex items-center gap-2.5 min-w-0 flex-1">
                <svg xmlns="http://www.w3.org/2000/svg" className="w-4 h-4 shrink-0 opacity-50" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
                </svg>
                <span className="text-xs truncate font-medium">{s.title}</span>
              </div>
              <button onClick={(e) => deleteSession(s.id, e)} className="opacity-0 group-hover:opacity-100 p-1 rounded transition-all shrink-0 ml-1 cursor-pointer delete-btn">
                <svg xmlns="http://www.w3.org/2000/svg" className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                </svg>
              </button>
            </div>
          ))}
        </div>

        <div className="p-3 sidebar-footer">
          <div className="flex items-center gap-2 mb-2">
            {user.picture && <img src={user.picture} className="w-6 h-6 rounded-full" alt={user.name} />}
            <div className="flex-1 min-w-0">
              <div className="text-xs font-medium truncate" style={{color:'rgba(245,230,216,0.7)'}}>{user.name}</div>
              <div className="text-[10px] truncate" style={{color:'rgba(212,165,116,0.3)'}}>{user.email}</div>
            </div>
          </div>
          <button onClick={handleLogout} className="w-full text-[10px] py-1.5 rounded-lg transition-all cursor-pointer" style={{color:'rgba(212,165,116,0.4)', border:'1px solid rgba(212,165,116,0.1)'}}>
            Sign out
          </button>
        </div>
      </aside>

      {sidebarOpen && <div className="fixed inset-0 z-20 bg-black/60 md:hidden" onClick={() => setSidebarOpen(false)} />}

      <div className="flex-1 flex flex-col h-full min-w-0 relative z-10">
        <header className="main-header flex items-center justify-between px-4 py-3 h-14 shrink-0">
          <div className="flex items-center gap-2">
            <button onClick={() => setSidebarOpen(true)} className="md:hidden p-1.5 rounded-lg cursor-pointer" style={{color:'rgba(212,165,116,0.5)'}}>
              <svg xmlns="http://www.w3.org/2000/svg" className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2"><path strokeLinecap="round" strokeLinejoin="round" d="M4 6h16M4 12h16M4 18h16" /></svg>
            </button>
            <span className="font-bold text-lg tracking-wide" style={{color:'#f5e6d8'}}>NanoSage</span>
            <div className="status-badge flex items-center gap-1.5 ml-1 px-2 py-0.5 rounded-full">
              <span className="w-1.5 h-1.5 rounded-full online-pulse" style={{background:'#c9956b'}} />
              <span className="text-[9px] font-semibold tracking-wider uppercase" style={{color:'#c9956b'}}>Online</span>
            </div>
          </div>
          <button onClick={clearChat} className="p-1.5 rounded-lg cursor-pointer clear-btn">
            <svg xmlns="http://www.w3.org/2000/svg" className="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M3 6h18" /><path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6" /><path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2" />
            </svg>
          </button>
        </header>

        <div className="flex-1 overflow-y-auto messages-area flex flex-col">
          <div className="max-w-3xl w-full mx-auto px-4 py-6 flex-1 flex flex-col">
            {messages.length === 0 ? (
              <WelcomeScreen greeting={greeting} />
            ) : (
              <div className="w-full flex-1">
                {messages.map(msg => <ChatMessage key={msg.id} msg={msg} />)}
                <div ref={messagesEndRef} />
              </div>
            )}
          </div>
        </div>

        <footer className="w-full max-w-3xl mx-auto px-4 pb-4 pt-2 shrink-0">
          {/* Attachment chips */}
          {(pdfContext || imageAttachment || pdfLoading) && (
            <div className="attachment-bar flex items-center gap-2 mb-2 px-2">
              {pdfLoading && (
                <div className="attachment-chip flex items-center gap-2 px-3 py-1.5 rounded-full text-xs">
                  <span className="pdf-spinner" />
                  <span>Extracting PDF text…</span>
                </div>
              )}
              {pdfContext && (
                <div className="attachment-chip flex items-center gap-2 px-3 py-1.5 rounded-full text-xs">
                  <span>📄</span>
                  <span className="truncate max-w-[180px]">{pdfContext.name}</span>
                  <button
                    onClick={() => setPdfContext(null)}
                    className="attachment-remove"
                    title="Remove PDF"
                  >✕</button>
                </div>
              )}
              {imageAttachment && (
                <div className="attachment-chip flex items-center gap-2 px-3 py-1.5 rounded-full text-xs">
                  <img src={imageAttachment.preview} alt="" className="w-6 h-6 rounded object-cover" />
                  <span className="truncate max-w-[180px]">{imageAttachment.name}</span>
                  <button
                    onClick={() => setImageAttachment(null)}
                    className="attachment-remove"
                    title="Remove image"
                  >✕</button>
                </div>
              )}
            </div>
          )}

          {/* Voice recording indicator */}
          {isRecording && (
            <div className="recording-indicator flex items-center gap-2 mb-2 px-3 py-1.5 rounded-full text-xs mx-2">
              <span className="recording-dot" />
              <span>Listening…</span>
            </div>
          )}

          <div className="input-glass relative flex items-end rounded-3xl p-1.5 transition-all">
            {/* Hidden file input */}
            <input
              ref={fileInputRef}
              type="file"
              accept=".pdf,.jpg,.jpeg,.png,.webp"
              onChange={handleFileUpload}
              className="hidden"
              style={{display: 'none'}}
            />

            {/* Attach button */}
            <button
              onClick={() => fileInputRef.current?.click()}
              disabled={isGenerating || pdfLoading}
              className="attach-btn w-9 h-9 rounded-full flex items-center justify-center transition-all shrink-0"
              title="Attach PDF or Image"
            >
              <svg xmlns="http://www.w3.org/2000/svg" className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2">
                <path strokeLinecap="round" strokeLinejoin="round" d="M15.172 7l-6.586 6.586a2 2 0 102.828 2.828l6.414-6.586a4 4 0 00-5.656-5.656l-6.415 6.585a6 6 0 108.486 8.486L20.5 13" />
              </svg>
            </button>

            <textarea ref={textareaRef} rows={1} value={input} onChange={handleChange} onKeyDown={handleKey}
              placeholder="Message NanoSage..." disabled={isGenerating}
              className="flex-1 bg-transparent border-0 outline-none text-sm px-3 py-2.5 resize-none min-h-[40px] max-h-[160px] leading-relaxed"
              style={{color:'#e8d5c4'}} />

            {/* Mic button */}
            <button
              onClick={toggleRecording}
              disabled={isGenerating}
              className={`mic-btn w-9 h-9 rounded-full flex items-center justify-center transition-all shrink-0 ${isRecording ? 'mic-recording' : ''}`}
              title={isRecording ? 'Stop recording' : 'Voice input'}
            >
              <svg xmlns="http://www.w3.org/2000/svg" className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2">
                <path strokeLinecap="round" strokeLinejoin="round" d="M12 1a3 3 0 00-3 3v8a3 3 0 006 0V4a3 3 0 00-3-3z" />
                <path strokeLinecap="round" strokeLinejoin="round" d="M19 10v2a7 7 0 01-14 0v-2" />
                <line x1="12" y1="19" x2="12" y2="23" />
                <line x1="8" y1="23" x2="16" y2="23" />
              </svg>
            </button>

            {/* Send button */}
            <button onClick={submit} disabled={isGenerating || !input.trim()}
              className={`send-btn w-9 h-9 rounded-full flex items-center justify-center text-white transition-all shrink-0 ml-1 ${isGenerating || !input.trim() ? 'send-disabled' : 'send-active'}`}>
              <svg xmlns="http://www.w3.org/2000/svg" className="w-4 h-4 fill-current" viewBox="0 0 24 24"><path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z" /></svg>
            </button>
          </div>
          <p className="text-center text-[10px] mt-2 font-medium tracking-wide" style={{color:'rgba(212,165,116,0.2)'}}>
            NanoSage · Built from scratch by Rishika Batra
          </p>
        </footer>
      </div>

      {/* Error toast */}
      {errorMsg && (
        <div className="error-toast" onClick={() => setErrorMsg('')}>
          <span className="error-icon">⚠️</span>
          <span>{errorMsg}</span>
        </div>
      )}
    </div>
  )
}
