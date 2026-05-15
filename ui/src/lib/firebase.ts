/**
 * Firebase client init (P16.7).
 *
 * The web SDK keys are public by design — they identify the project,
 * they don't authorize anything by themselves. Real authentication
 * happens server-side: backend verifies the ID token Firebase issues
 * to the browser, and only after that does it mint a tenant Bearer.
 */
import { initializeApp, type FirebaseApp } from 'firebase/app';
import { GoogleAuthProvider, getAuth, type Auth } from 'firebase/auth';

const config = {
  apiKey: import.meta.env.VITE_FIREBASE_API_KEY,
  authDomain: import.meta.env.VITE_FIREBASE_AUTH_DOMAIN,
  projectId: import.meta.env.VITE_FIREBASE_PROJECT_ID,
  appId: import.meta.env.VITE_FIREBASE_APP_ID,
};

let app: FirebaseApp | null = null;
let _auth: Auth | null = null;

function ensureApp(): FirebaseApp {
  if (!app) {
    if (!config.apiKey || !config.authDomain || !config.projectId) {
      throw new Error(
        'Firebase config missing — set VITE_FIREBASE_API_KEY, VITE_FIREBASE_AUTH_DOMAIN, VITE_FIREBASE_PROJECT_ID before building.',
      );
    }
    app = initializeApp(config);
  }
  return app;
}

export function firebaseAuth(): Auth {
  if (!_auth) _auth = getAuth(ensureApp());
  return _auth;
}

export const googleProvider = new GoogleAuthProvider();
