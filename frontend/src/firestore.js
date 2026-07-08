import { initializeApp } from "firebase/app"
import { getFirestore, doc, setDoc, getDocs, deleteDoc, collection } from "firebase/firestore"

const firebaseConfig = {
  apiKey: import.meta.env.VITE_FIREBASE_API_KEY,
  authDomain: import.meta.env.VITE_FIREBASE_AUTH_DOMAIN,
  projectId: import.meta.env.VITE_FIREBASE_PROJECT_ID,
  storageBucket: import.meta.env.VITE_FIREBASE_STORAGE_BUCKET,
  messagingSenderId: import.meta.env.VITE_FIREBASE_MESSAGING_SENDER_ID,
  appId: import.meta.env.VITE_FIREBASE_APP_ID
}

const app = initializeApp(firebaseConfig)
const db = getFirestore(app)

export async function saveSessions(email, sessions) {
  try {
    const ref = doc(db, "users", email)
    await setDoc(ref, { sessions: JSON.stringify(sessions) })
  } catch (e) {
    console.error("Firestore save error:", e)
  }
}

export async function loadSessions(email) {
  try {
    const ref = doc(db, "users", email)
    const { getDoc } = await import("firebase/firestore")
    const snap = await getDoc(ref)
    if (snap.exists()) {
      return JSON.parse(snap.data().sessions || "[]")
    }
    return []
  } catch (e) {
    console.error("Firestore load error:", e)
    return []
  }
}
