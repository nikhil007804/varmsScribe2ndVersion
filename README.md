# VarmaScribe Insurance

AI-powered medical documentation and insurance claim optimization system that converts doctor-patient conversations into structured clinical notes and insurance-ready documentation.

## 🚀 Features

- **Audio Transcription** - Convert medical conversations to text with speaker diarization
- **Clinical Notes Generation** - Generate SOAP and H&P formatted notes
- **Insurance Documentation** - Create claim-ready medical necessity narratives
- **Claim Analysis** - AI-powered risk assessment and CPT code recommendations
- **Missing Elements Detection** - Identify gaps in documentation for claim approval
- **Quick Fix Mode** - Add missing information and regenerate documentation

## 📋 Prerequisites

- Python 3.8+
- API keys for AssemblyAI and Google Gemini

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd VarmaScribe-Insurance
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure API keys**
   
   Create a `.env` file in the project root:
   ```env
   ASSEMBLE_API_KEY=your_assemblyai_api_key_here
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

   **Getting API Keys:**
   
   - **AssemblyAI**: Sign up at [assemblyai.com](https://assemblyai.com) and get your API key from the dashboard
   - **Gemini**: Get API key from [Google AI Studio](https://makersuite.google.com/app/apikey)

## 🚀 Running the Application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 📖 Usage

1. **Upload Audio File**
   - Supported formats: WAV, MP3, M4A, MP4, WEBM
   - Click "🚀 Generate Documentation" to start the pipeline

2. **Pipeline Process**
   - Step 1: Transcribe audio with speaker diarization
   - Step 2: Generate SOAP note
   - Step 3: Generate H&P note
   - Step 4: Create insurance-ready documentation
   - Step 5: Analyze claim rejection risk

3. **Review Results**
   - View all generated documentation in organized tabs
   - Check claim analysis for risk assessment
   - Review suggested CPT codes
   - Identify missing documentation elements

4. **Fix Missing Information**
   - Click "🔧" on any missing element
   - Add factual clinical information
   - Regenerate documentation with updates

5. **Download**
   - Download insurance-ready notes as text files

## 🏗️ Architecture

```
VarmaScribe-Insurance/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── .env                  # API keys (create this file)
└── README.md            # This file
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `ASSEMBLE_API_KEY` | AssemblyAI API key for transcription | Yes |
| `GEMINI_API_KEY` | Google Gemini API key for AI generation | Yes |

### Model Configuration

- **Transcription**: AssemblyAI with speaker diarization
- **AI Generation**: Gemini 2.5 Flash
- **Output Formats**: SOAP, H&P, Claim Justification

## 📊 Output Formats

### SOAP Note
- Subjective
- Objective
- Assessment
- Plan

### H&P Note
- Chief Complaint
- History of Present Illness (HPI)
- Past Medical History (PMH)
- Medications
- Allergies
- Review of Systems (ROS)
- Physical Exam
- Assessment
- Plan

### Insurance Documentation
- Clinical Summary
- Symptoms/Functional Impact
- Relevant History
- Exam/Objective Findings
- Assessment (Problem List)
- Plan and Medical Necessity Rationale
- Risk/Complexity

### Claim Analysis
- Risk Level (Low/Medium/High)
- Risk Explanation
- Missing Documentation Elements
- Improvement Suggestions
- Suggested CPT Codes

## 🚨 Important Notes

- **HIPAA Compliance**: This tool processes medical conversations. Ensure compliance with HIPAA regulations and patient privacy requirements.
- **Clinical Accuracy**: The AI generates documentation based on provided audio. Always review and verify clinical accuracy before use.
- **API Limits**: Be aware of API rate limits and quotas for both AssemblyAI and Gemini services.

## 🐛 Troubleshooting

### Common Issues

1. **"Missing API key" error**
   - Verify `.env` file exists with correct keys
   - Check for typos in variable names

2. **"Quota exceeded" error**
   - Check API usage limits
   - Wait for quota reset or upgrade plan

3. **Audio transcription fails**
   - Verify audio file format is supported
   - Check audio file quality and duration

4. **App won't start**
   - Ensure all dependencies installed: `pip install -r requirements.txt`
   - Check Python version (3.8+)

### Debug Mode

The app includes a debug info panel in the sidebar that shows:
- Current working directory
- .env file status
- API key presence (last 4 digits shown)

## 📝 License

This project is proprietary software. All rights reserved.

## 🤝 Support

For technical support or questions:
- Check the troubleshooting section above
- Review API documentation for AssemblyAI and Gemini
- Contact the development team

---

**⚠️ Medical Disclaimer**: This tool is for documentation assistance only. Always verify clinical accuracy and follow appropriate medical documentation standards.
