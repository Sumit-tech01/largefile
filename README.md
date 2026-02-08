# 🎤 Voice Sentiment Analysis

Real-time emotion detection from your voice using AI. Deploy easily to Vercel or Netlify!

## ✨ Features

- 🎤 **Live Recording** - Record from microphone with visualizer
- 📁 **File Upload** - Upload WAV, MP3, WebM, OGG
- 😊 **Emotion Detection** - 6 emotions: Joy, Sadness, Anger, Fear, Love, Surprise
- 📊 **Live Visualizations** - Interactive charts
- 🚀 **Easy Deploy** - Vercel/Netlify ready

## 🚀 Quick Deploy

### Option 1: Vercel (Recommended)

```bash
# Install Vercel CLI
npm i -g vercel

# Deploy
vercel

# Or push to GitHub and import in Vercel dashboard
```

### Option 2: Netlify

```bash
# Install Netlify CLI
npm i -g netlify-cli

# Deploy
netlify deploy --prod

# Or connect GitHub repo in Netlify dashboard
```

### Option 3: Build Static Files

```bash
# Build frontend
cd frontend
npm install
npm run build

# Output in frontend/dist/ - upload to any hosting!
```

---

## 📁 Project Structure

```
voice-sentiment/
├── frontend/              # STATIC - Deploy to Vercel/Netlify
│   ├── src/
│   │   ├── js/
│   │   │   ├── app.js       # Main app
│   │   │   ├── recorder.js  # Audio recording
│   │   │   ├── chart.js     # Visualizations
│   │   │   └── api.js       # API client
│   │   ├── css/
│   │   └── index.html
│   ├── package.json
│   └── vite.config.js
├── api/                   # Vercel Serverless Functions
│   ├── health.py
│   └── analyze.py
├── backend/               # Optional: Full ML Backend
│   ├── app/
│   └── requirements.txt
├── vercel.json           # Vercel config
└── README.md
```

---

## 🎯 Usage

### Local Development

```bash
# Install and run
npm run dev
# Open http://localhost:5173
```

### Build for Production

```bash
# Build static files
npm run build

# Output: frontend/dist/
# Deploy to any static hosting!
```

---

## 🌍 Deploy to Vercel

### 1. Push to GitHub

```bash
git init
git add .
git commit -m "Initial commit"
gh repo create my-voice-app --public --push
```

### 2. Import in Vercel

1. Go to https://vercel.com
2. Click **"Add New..."** → **"Project"**
3. Import your GitHub repository
4. Settings (auto-detected):
   - Framework: **Vite**
   - Build Command: `npm run build`
   - Output Directory: `frontend/dist`
5. Click **Deploy**

### 3. Your URL

```
https://your-app-name.vercel.app
```

---

## 🌍 Deploy to Netlify

### 1. Push to GitHub

```bash
git init
git add .
git commit -m "Initial commit"
gh repo create my-voice-app --public --push
```

### 2. Import in Netlify

1. Go to https://netlify.com
2. Click **"Add new site"** → **"Import an existing project"**
3. Connect GitHub and select your repository
4. Settings:
   - Base directory: `frontend`
   - Build command: `npm run build`
   - Publish directory: `frontend/dist`
5. Click **Deploy site**

### 3. Add Custom Domain (Optional)

1. Go to **Domain Management** in Netlify
2. Click **"Add custom domain"**
3. Enter your domain
4. Update DNS records as shown

---

## 🎤 How to Use

1. **Open your deployed site**
2. **Click "Live Recording"** or "Upload Audio"
3. **Allow microphone** when prompted
4. **Record or upload** audio
5. **View results** with emotions chart

---

## 🔧 Tech Stack

| Layer | Technology |
|-------|------------|
| **Frontend** | Vite 5, Vanilla JS, Tailwind CSS, Chart.js |
| **Audio** | MediaRecorder API, Web Audio API |
| **Deployment** | Vercel / Netlify (static files) |
| **ML (Demo)** | Keyword-based sentiment |
| **ML (Full)** | OpenAI Whisper, Transformers |

---

## 📝 API Endpoints

### Vercel Serverless (Demo Mode)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check |
| `/api/analyze` | POST | Text sentiment (demo) |

### Backend API (Full ML - Deploy Separately)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/analyze/upload` | POST | Upload audio file |
| `/api/v1/analyze/text` | POST | Analyze text |
| `/api/v1/health` | GET | Health check |

---

## 🐳 Full Backend (Optional)

For full Whisper + Transformers ML:

```bash
# Deploy backend to Railway/Render
cd backend
pip install -r requirements.txt
gunicorn app.main:app -w 4 -b 0.0.0.0:5000
```

---

## 📄 License

MIT

---

## 🙏 Credits

- [Chart.js](https://www.chartjs.org/)
- [Tailwind CSS](https://tailwindcss.com/)
- [Vite](https://vitejs.dev/)
- [OpenAI Whisper](https://github.com/openai/whisper)

---

Built with ❤️ using [VS Code + AI Workflow Framework](PLAN.md)

