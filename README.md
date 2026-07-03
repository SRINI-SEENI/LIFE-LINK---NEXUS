# 🌐 Life Link Nexus
### *Every second, let's connect to save many lives*

**Life Link Nexus** is a premium, state-of-the-art healthcare portal designed to bridge the gap between patients, verified administrators, and emergency donors. It combines **Real-time cloud database synchronization**, **Machine Learning health classifiers**, and **Advanced LLM Chat Agents (RAG)** to deliver an intelligent, zero-delay emergency medical ecosystem.

🔗 **Live Production Link**: [https://lifelink-nexus.vercel.app/](https://lifelink-nexus.vercel.app/)

---

## 🚀 Advanced Tech Stack & AI Architecture

### 🧠 Agentic AI & LLM Integration (RAG)
The platform features an advanced **AI Health Advisor** widget integrated into the medical screening workflows:
* **Multi-Model Orchestration**: Supports dynamic switching between **Gemini 2.5 Flash**, **Llama 3.1**, **GPT-4o-Mini**, and **Qwen 2.5**.
* **Form & Context-Aware Prompting (RAG)**: The AI agent automatically retrieves the active patient screening form values (e.g. glucose, blood pressure, BMI, cholesterol, ECG results) to formulate a clinical-style system context.
* **Intelligent Warning Classifier**: Automatically parses user inputs for critical medical triggers (e.g. severe chest pain, shortness of breath, symptoms of diabetic ketoacidosis) to flag urgent warnings and suggest immediate clinical actions.
* **Robust Local Rule-Based Fallback**: Built with automated fail-safes to ensure 100% availability even under API quota exhaustion.
* **Edge Token Optimization**: Compact, highly-optimized system instructions to ensure lightning-fast response times.

### 📊 Machine Learning Diagnostic Classifiers
In addition to LLMs, the platform integrates trained predictive models for instant risk analysis:
* **Cardiovascular Disease Classifier**: Predicts heart disease risk using demographic, clinical (ECG/cholesterol), and lifestyle inputs.
* **Diabetes Risk Estimator**: Classifies diabetes risk based on physiological markers like glucose levels, blood pressure, and BMI.
* *Models were trained on real clinical datasets using Scikit-Learn and serialized using Joblib for deployment.*

---

## 🛠️ Platform Core Features

* 🩸 **Blood & Organ Donation Network** — Live real-time database matching donors with recipients instantly.
* 🧬 **Donor Verification Pipelines** — Strict admin-guided validation system for verified donors.
* 🏥 **Clinic & Hospital Slot Booking** — Appointment booking system with real-time slot selection and physician directories.
* 💊 **E-Pharmacy & Orders Catalog** — Online prescription matching and medicine inventory management.
* 🚔 **Emergency Incident Logger (E-FIR)** — Quick documentation and response trigger systems for critical incidents.
* ⚙️ **Multi-Role Portal Access**:
  * **User Dashboard**: Visual glassmorphic portal to schedule medical checkups, manage donor status, and view urgent public requests.
  * **Admin Dashboard**: Verification panels for clinical records, pharmacy inventory, and appointments.
  * **Maintainer Console**: Root control system for review, audit, and credential verification of administrators.

---

## 💻 Technology Stack

| Layer | Technologies |
|---|---|
| **Frontend UI/UX** | HTML5, Vanilla CSS3, Modern JavaScript (ES6 Modules), Poppins Google Fonts |
| **Styling Paradigm** | Premium Glassmorphism, Floating Glow Orbs, Transition Keyframes, Micro-animations |
| **Backend & Auth** | Google Firebase Authentication, Cloud Firestore (Real-Time Database) |
| **Machine Learning** | Python, Scikit-learn, Joblib |
| **Large Language Models** | Gemini 2.5 API, Llama 3.1 API, Qwen 2.5 API (via Pollinations AI Endpoint) |
| **Deployment** | Vercel Serverless Hosting |

---

## 🔐 Security & Optimization

* **Domain-Restricted Credentials**: The Firebase credentials are restricted via HTTP Referrers to execute only on `https://lifelink-nexus.vercel.app/*` to prevent usage spoofing.
* **Client-Side Obfuscation**: Sensitive API configurations are obfuscated via Base64 coding at runtime, preventing plain-text scraping.
* **Firestore Security Rules**: Server-side validation rules whitelisting read/write privileges based on user roles and document ownership.

---

## 👨‍💻 Author

**SRINIVASAN N**

---
<p align="center">
  <i>Made with ❤️ for saving lives</i>
</p>
