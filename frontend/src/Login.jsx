import { useEffect, useMemo } from 'react'

const GOOGLE_CLIENT_ID = '1025936262790-tvet0qfnrrhq67fjq1o96bt249tknbni.apps.googleusercontent.com'

export default function Login({ onLogin }) {
  useEffect(() => {
    const script = document.createElement('script')
    script.src = 'https://accounts.google.com/gsi/client'
    script.async = true
    script.defer = true
    script.onload = () => {
      window.google.accounts.id.initialize({
        client_id: GOOGLE_CLIENT_ID,
        callback: (response) => {
          const payload = parseJwt(response.credential)
          onLogin({
            name: payload.name,
            email: payload.email,
            picture: payload.picture,
          })
        },
      })
      window.google.accounts.id.renderButton(
        document.getElementById('google-btn'),
        { theme: 'outline', size: 'large', width: 280 }
      )
    }
    document.body.appendChild(script)
    return () => { document.body.removeChild(script) }
  }, [])

  const parseJwt = (token) => {
    const base64 = token.split('.')[1].replace(/-/g, '+').replace(/_/g, '/')
    return JSON.parse(atob(base64))
  }

  const sparkles = useMemo(() => {
    const colors = ['#d4a574', '#c9956b', '#e8c4a8', '#f5e6d8']
    return Array.from({ length: 45 }).map((_, i) => ({
      id: i,
      left: `${Math.random() * 100}%`,
      top: `${Math.random() * 100}%`,
      size: `${Math.random() * 4 + 2}px`,
      color: colors[Math.floor(Math.random() * colors.length)],
      duration: `${Math.random() * 3 + 2}s`,
      delay: `${Math.random() * 5}s`,
    }))
  }, [])

  return (
    <div className="login-root">
      <div className="orb orb1" />
      <div className="orb orb2" />
      <div className="orb orb3" />
      <div className="orb orb4" />

      <div className="login-waves" aria-hidden="true">
        <svg className="wave wave1" viewBox="0 0 2400 200" preserveAspectRatio="none">
          <defs>
            <linearGradient id="glow1" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#8b3a4a" stopOpacity="0.8" />
              <stop offset="100%" stopColor="#8b3a4a" stopOpacity="0.1" />
            </linearGradient>
          </defs>
          <path fill="url(#glow1)" d="M0,80 C300,180 900,20 1200,80 C1500,180 2100,20 2400,80 L2400,200 L0,200 Z" />
        </svg>
        <svg className="wave wave2" viewBox="0 0 2400 200" preserveAspectRatio="none">
          <defs>
            <linearGradient id="glow2" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#b5626a" stopOpacity="0.8" />
              <stop offset="100%" stopColor="#b5626a" stopOpacity="0.1" />
            </linearGradient>
          </defs>
          <path fill="url(#glow2)" d="M0,100 C300,40 900,160 1200,100 C1500,40 2100,160 2400,100 L2400,200 L0,200 Z" />
        </svg>
        <svg className="wave wave3" viewBox="0 0 2400 200" preserveAspectRatio="none">
          <defs>
            <linearGradient id="glow3" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#c9956b" stopOpacity="0.9" />
              <stop offset="100%" stopColor="#c9956b" stopOpacity="0.0" />
            </linearGradient>
          </defs>
          <path fill="url(#glow3)" d="M0,130 C300,190 900,70 1200,130 C1500,190 2100,70 2400,130 L2400,200 L0,200 Z" />
        </svg>
      </div>

      <div className="login-sparkles" aria-hidden="true">
        {sparkles.map(s => (
          <div 
            key={s.id} 
            className="sparkle" 
            style={{
              left: s.left,
              top: s.top,
              width: s.size,
              height: s.size,
              backgroundColor: s.color,
              boxShadow: `0 0 ${parseInt(s.size)*2}px ${s.color}`,
              animationDuration: s.duration,
              animationDelay: s.delay
            }} 
          />
        ))}
      </div>

      <div className="login-card">
        <img src="/logo.png" style={{width:"100px", height:"100px", objectFit:"contain", marginBottom:"4px", filter:"drop-shadow(0 0 16px rgba(181,98,106,0.5))"}} alt="NanoSage" />
        <h1 className="login-title">NanoSage</h1>
        <p className="login-subtitle">Your intelligent AI assistant, built from scratch.</p>

        <div className="login-divider" />

        <p className="login-prompt">Sign in to continue</p>
        <div id="google-btn" className="google-btn-wrap" />

        <p className="login-footer">
          By signing in you agree to use NanoSage responsibly.
        </p>
      </div>
    </div>
  )
}
