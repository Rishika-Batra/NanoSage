import { useState, useRef, useEffect, useCallback } from 'react'
import './index.css'
import Login from './Login'

const API_BASE = 'https://nanosage.onrender.com'
const HF_TOKEN = import.meta.env.VITE_HF_TOKEN
const HF_URL = 'https://api-inference.huggingface.co/models/TinyLlama/TinyLlama-1.1B-Chat-v1.0'

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
        {isUser ? (
          <div className="user-bubble px-4 py-3 rounded-2xl rounded-tr-sm text-sm text-white leading-relaxed break-words">
            {msg.content}
          </div>
        ) : (
          <div className="ai-card px-4 py-3 rounded-xl text-sm leading-relaxed break-words">
            {isThinking ? <TypingDots /> : (
              <>
                <span className="whitespace-pre-wrap">{(msg.content || '…').trimEnd()}</span>{msg.streaming && <span className="cursor-blink inline-block w-0.5 h-4 ml-1 align-middle" style={{background:'#c9956b'}} />}
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
      <img src="/logo.png" className="w-40 h-40" style={{marginBottom:"-16px"}} style={{objectFit:"contain", filter:"drop-shadow(0 0 16px rgba(181,98,106,0.5))"}} alt="NanoSage" />
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

export default function App() {
  const [user, setUser] = useState(() => {
    try { return JSON.parse(localStorage.getItem('nanosage_user') || 'null') }
    catch { return null }
  })

  const greetingIdx = useRef(Math.floor(Math.random() * GREETINGS.length))
  const [sessions, setSessions] = useState([])
  const [activeSession, setActiveSession] = useState(null) // full session object
  const [isGenerating, setIsGenerating] = useState(false)
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [input, setInput] = useState('')

  const messagesEndRef = useRef(null)
  const abortRef = useRef(null)
  const textareaRef = useRef(null)

  const messages = activeSession?.messages || []

  // Load sessions on login
  useEffect(() => {
    if (!user?.email) { setSessions([]); setActiveSession(null); return }
    try {
      const saved = localStorage.getItem(getSessionKey(user.email))
      setSessions(saved ? JSON.parse(saved) : [])
    } catch { setSessions([]) }
    setActiveSession(null)
  }, [user?.email])

  // Save sessions
  useEffect(() => {
    if (!user?.email) return
    localStorage.setItem(getSessionKey(user.email), JSON.stringify(sessions))
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

  const sendMessage = async (text) => {
    if (!text.trim() || isGenerating) return

    const now = () => new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })

    // Create or use existing session
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

    const userMsg = { id: Date.now(), role: 'user', content: text, timestamp: now() }
    const aiMsgId = Date.now() + 1
    const aiMsg = { id: aiMsgId, role: 'assistant', content: '', streaming: true, timestamp: null }
    const withUser = [...session.messages, userMsg, aiMsg]

    // Update active session with user message immediately
    const updatedSession = { ...session, messages: withUser }
    setActiveSession(updatedSession)
    setIsGenerating(true)

    // Build history
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

      const response = await fetch(`${API_BASE}/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text, history: historyPairs.slice(-3) }),
        signal: controller.signal,
      })

      if (!response.ok) throw new Error(`HTTP ${response.status}`)

      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let accumulated = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        const chunk = decoder.decode(value, { stream: true })
        for (const line of chunk.split('\n')) {
          if (!line.startsWith('data: ')) continue
          const data = line.slice(6).trim()
          if (data === '[DONE]') break
          try {
            const parsed = JSON.parse(data)
            if (parsed.token) {
              accumulated += parsed.token
              const snap = accumulated
              setActiveSession(prev => {
                if (!prev) return prev
                return {
                  ...prev,
                  messages: prev.messages.map(m =>
                    m.id === aiMsgId ? { ...m, content: snap, streaming: true } : m
                  )
                }
              })
            }
          } catch {}
        }
      }

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
          <div className="input-glass relative flex items-end rounded-3xl p-1.5 transition-all">
            <textarea ref={textareaRef} rows={1} value={input} onChange={handleChange} onKeyDown={handleKey}
              placeholder="Message NanoSage..." disabled={isGenerating}
              className="flex-1 bg-transparent border-0 outline-none text-sm px-4 py-2.5 resize-none min-h-[40px] max-h-[160px] leading-relaxed"
              style={{color:'#e8d5c4'}} />
            <button onClick={submit} disabled={isGenerating || !input.trim()}
              className={`send-btn w-9 h-9 rounded-full flex items-center justify-center text-white transition-all shrink-0 ml-2 ${isGenerating || !input.trim() ? 'send-disabled' : 'send-active'}`}>
              <svg xmlns="http://www.w3.org/2000/svg" className="w-4 h-4 fill-current" viewBox="0 0 24 24"><path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z" /></svg>
            </button>
          </div>
          <p className="text-center text-[10px] mt-2 font-medium tracking-wide" style={{color:'rgba(212,165,116,0.2)'}}>
            NanoSage · Built from scratch in PyTorch 🔥
          </p>
        </footer>
      </div>
    </div>
  )
}
