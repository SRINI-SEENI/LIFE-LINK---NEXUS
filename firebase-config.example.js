// Firebase Configuration Template
// IMPORTANT: This is a TEMPLATE file. For deployment:
// 1. Copy this file to firebase-config.js
// 2. Replace the placeholder values with your actual Firebase credentials
// 3. The firebase-config.js file is ignored by git for security

// NOTE: Firebase API keys ARE public knowledge (visible in network tab anyway)
// Real security comes from Firebase Rules, NOT from hiding the key

export const firebaseConfig = {
  apiKey: "YOUR_API_KEY_HERE",
  authDomain: "YOUR_PROJECT.firebaseapp.com",
  projectId: "YOUR_PROJECT_ID",
  storageBucket: "YOUR_PROJECT.firebasestorage.app",
  messagingSenderId: "YOUR_SENDER_ID",
  appId: "YOUR_APP_ID"
};

// STEPS TO DEPLOY:
// 1. In Firebase Console → Project Settings → General tab
// 2. Copy your Web app credentials
// 3. Replace the values above with your actual credentials
// 4. Save as firebase-config.js in the root directory
// 5. The app will automatically load it
