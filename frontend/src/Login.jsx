import { useEffect } from 'react'

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

  return (
    <div className="login-root">
      <div className="orb orb1" />
      <div className="orb orb2" />
      <div className="orb orb3" />
      <div className="orb orb4" />

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
