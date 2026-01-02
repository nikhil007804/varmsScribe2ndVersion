import os
import time
import tempfile
from typing import Optional
from dataclasses import dataclass

import assemblyai as aai
import google.generativeai as genai
import streamlit as st
from dotenv import load_dotenv


GEMINI_MODEL_NAME = "gemini-2.5-flash"
INSURANCE_STYLE_DEFAULT = "Claim Justification"


@dataclass
class ClaimAnalysis:
    risk_level: str
    risk_explanation: str
    missing_elements: list[tuple[str, str]]
    improvement_suggestions: list[str]
    suggested_cpt_codes: list[tuple[str, str]]


def _read_dotenv_value(dotenv_path: str, key: str) -> Optional[str]:
    try:
        with open(dotenv_path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip().lstrip("\ufeff")
                if k != key:
                    continue
                v = v.strip().strip("\"").strip("'")
                return v or None
    except OSError:
        return None
    return None


def _load_config() -> tuple[Optional[str], Optional[str]]:
    dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
    load_dotenv(dotenv_path=dotenv_path, override=True)

    assemblyai_key = os.getenv("ASSEMBLE_API_KEY")
    if not assemblyai_key:
        assemblyai_key = _read_dotenv_value(dotenv_path, "ASSEMBLE_API_KEY")
        if assemblyai_key:
            os.environ["ASSEMBLE_API_KEY"] = assemblyai_key

    gemini_key = os.getenv("GEMINI_API_KEY")
    if not gemini_key:
        gemini_key = _read_dotenv_value(dotenv_path, "GEMINI_API_KEY")
        if gemini_key:
            os.environ["GEMINI_API_KEY"] = gemini_key

    return assemblyai_key, gemini_key


def _configure_clients(assemblyai_key: Optional[str], gemini_key: Optional[str]) -> None:
    if assemblyai_key:
        aai.settings.api_key = assemblyai_key
    if gemini_key:
        genai.configure(api_key=gemini_key)


def _transcribe_audio_bytes_with_diarization(audio_bytes: bytes, file_name: str) -> tuple[str, str]:
    if not aai.settings.api_key:
        raise RuntimeError("Missing required API key. Please configure required keys in the server environment.")

    transcriber = aai.Transcriber()
    config = aai.TranscriptionConfig(speaker_labels=True)
    suffix = os.path.splitext(file_name)[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        transcript = transcriber.transcribe(tmp_path, config=config)
        if transcript.status == aai.TranscriptStatus.error:
            raise RuntimeError(transcript.error or "Transcription failed")

        plain = transcript.text or ""
        diarized_lines: list[str] = []
        if getattr(transcript, "utterances", None):
            for u in transcript.utterances:
                speaker = getattr(u, "speaker", None)
                text = getattr(u, "text", "")
                if text:
                    diarized_lines.append(f"Speaker {speaker}: {text}")

        diarized = "\n".join(diarized_lines) if diarized_lines else plain
        return plain, diarized
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def _gemini_generate_with_retry(prompt: str, model_name: str, max_retries: int = 3) -> str:
    if not getattr(genai, "configure", None):
        raise RuntimeError("AI generation client not available")
    
    for attempt in range(max_retries):
        try:
            model = genai.GenerativeModel(model_name)
            resp = model.generate_content(prompt)
            if hasattr(resp, "text") and resp.text:
                return resp.text
            return ""
        except Exception as e:
            error_str = str(e).lower()
            if "quota" in error_str or "429" in error_str or "rate limit" in error_str:
                if attempt < max_retries - 1:
                    wait_time = min(60, (2 ** attempt) * 5)
                    st.warning(f"⏳ API quota reached. Retrying in {wait_time} seconds... (Attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                    continue
                else:
                    raise RuntimeError("⚠️ API quota exceeded. Please wait a few minutes and try again, or consider upgrading your API plan.") from e
            else:
                raise
    return ""


def _gemini_generate(prompt: str, model_name: str) -> str:
    return _gemini_generate_with_retry(prompt, model_name)


def _build_clinical_note_prompt(conversation_text: str, note_style: str) -> str:
    return (
        "You are a medical documentation assistant. "
        "Convert the following doctor-patient conversation into a structured clinical note. "
        "Do not invent facts. If something is missing, write 'Not documented'. "
        "Do not change clinical decisions; only document what is present.\n\n"
        f"NOTE FORMAT: {note_style}\n"
        "If NOTE FORMAT is SOAP, output exactly these headers: Subjective, Objective, Assessment, Plan.\n"
        "If NOTE FORMAT is H&P, output exactly these headers: Chief Complaint, HPI, PMH, Medications, Allergies, ROS, "
        "Physical Exam, Assessment, Plan.\n\n"
        "CONVERSATION:\n"
        f"{conversation_text}\n"
    )


def _build_insurance_prompt(clinical_note: str, insurance_style: str) -> str:
    return (
        "You are an insurance documentation assistant. "
        "Rewrite the provided clinical note into insurer-friendly documentation. "
        "Do not add new diagnoses, findings, vitals, exam elements, test results, or treatments. "
        "Do not alter clinical decisions. "
        "Your job is to clarify medical necessity using only what is present.\n\n"
        f"OUTPUT STYLE: {insurance_style}\n"
        "Output must be concise, structured, and ready to support a claim.\n"
        "Include the following sections (even if 'Not documented'):\n"
        "1) Clinical Summary\n"
        "2) Symptoms/Functional Impact\n"
        "3) Relevant History\n"
        "4) Exam/Objective Findings\n"
        "5) Assessment (Problem List)\n"
        "6) Plan and Medical Necessity Rationale\n"
        "7) Risk/Complexity (brief)\n\n"
        "CLINICAL NOTE:\n"
        f"{clinical_note}\n"
    )


def _build_claim_analysis_prompt(insurance_note: str, clinical_note: str) -> str:
    return (
        "You are an insurance claim review assistant. Analyze the provided documentation for claim rejection risk.\n\n"
        "CRITICAL RULES:\n"
        "- Base your analysis ONLY on what is documented\n"
        "- Do NOT invent medical facts\n"
        "- Do NOT suggest adding false information\n"
        "- Flag what is missing, not what should be fabricated\n\n"
        "ANALYZE FOR:\n"
        "1) RISK LEVEL (Low/Medium/High) based on:\n"
        "   - Presence of objective findings\n"
        "   - Severity documentation\n"
        "   - Medical necessity clarity\n"
        "   - Consistency between symptoms and plan\n\n"
        "2) MISSING ELEMENTS that insurers commonly require:\n"
        "   - Objective exam findings\n"
        "   - Severity indicators (duration, frequency, impact)\n"
        "   - Failed conservative treatments\n"
        "   - Functional impact on daily activities\n"
        "   - Baseline measurements or vitals\n"
        "   - Prior authorization elements\n\n"
        "3) IMPROVEMENT SUGGESTIONS:\n"
        "   - Indicate what TYPE of information is missing\n"
        "   - Suggest WHERE additional documentation would help\n"
        "   - Do NOT provide specific medical values or findings\n\n"
        "4) CPT CODE RECOMMENDATIONS:\n"
        "   - Suggest appropriate E/M CPT codes based on documented complexity\n"
        "   - Consider: history, exam, medical decision making\n"
        "   - Common codes: 99202-99205 (new patient), 99211-99215 (established)\n"
        "   - Explain why each code fits the documentation level\n\n"
        "OUTPUT FORMAT (use this exact structure):\n"
        "RISK_LEVEL: [Low Risk|Medium Risk|High Risk]\n"
        "RISK_EXPLANATION: [2-3 sentences explaining the risk assessment]\n\n"
        "MISSING_ELEMENTS:\n"
        "- [Element name]: [Why it matters for insurance]\n"
        "- [Element name]: [Why it matters for insurance]\n\n"
        "IMPROVEMENT_SUGGESTIONS:\n"
        "- [Suggestion]\n"
        "- [Suggestion]\n\n"
        "CPT_CODES:\n"
        "- [Code]: [Description and why it fits]\n"
        "- [Code]: [Description and why it fits]\n\n"
        "INSURANCE DOCUMENTATION:\n"
        f"{insurance_note}\n\n"
        "CLINICAL NOTE:\n"
        f"{clinical_note}\n"
    )


def _parse_claim_analysis(analysis_text: str) -> ClaimAnalysis:
    lines = analysis_text.strip().split("\n")
    
    risk_level = "Medium Risk"
    risk_explanation = ""
    missing_elements = []
    improvement_suggestions = []
    suggested_cpt_codes = []
    
    current_section = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        if line.startswith("RISK_LEVEL:"):
            risk_level = line.replace("RISK_LEVEL:", "").strip()
        elif line.startswith("RISK_EXPLANATION:"):
            risk_explanation = line.replace("RISK_EXPLANATION:", "").strip()
        elif line.startswith("MISSING_ELEMENTS:"):
            current_section = "missing"
        elif line.startswith("IMPROVEMENT_SUGGESTIONS:"):
            current_section = "suggestions"
        elif line.startswith("CPT_CODES:"):
            current_section = "cpt"
        elif line.startswith("INSURANCE DOCUMENTATION:") or line.startswith("CLINICAL NOTE:"):
            break
        elif line.startswith("-"):
            content = line[1:].strip()
            if current_section == "missing" and ":" in content:
                parts = content.split(":", 1)
                missing_elements.append((parts[0].strip(), parts[1].strip()))
            elif current_section == "suggestions":
                improvement_suggestions.append(content)
            elif current_section == "cpt" and ":" in content:
                parts = content.split(":", 1)
                suggested_cpt_codes.append((parts[0].strip(), parts[1].strip()))
    
    return ClaimAnalysis(
        risk_level=risk_level,
        risk_explanation=risk_explanation,
        missing_elements=missing_elements,
        improvement_suggestions=improvement_suggestions,
        suggested_cpt_codes=suggested_cpt_codes,
    )


def _init_state() -> None:
    defaults = {
        "transcript_text": "",
        "diarized_transcript_text": "",
        "soap_note": "",
        "hp_note": "",
        "insurance_note": "",
        "claim_analysis": None,
        "fix_mode": False,
        "additional_info": "",
        "current_step": 0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _render_progress_card():
    """Render the progress tracking card"""
    steps = [
        ("Transcript", "Audio converted to text", bool(st.session_state.transcript_text)),
        ("SOAP note", "Structured clinical summary", bool(st.session_state.soap_note)),
        ("H&P note", "History & physical format", bool(st.session_state.hp_note)),
        ("Insurance note", "Claim-ready narrative", bool(st.session_state.insurance_note)),
        ("Claim review", "Risk + missing elements", bool(st.session_state.claim_analysis)),
    ]
    
    completed = sum(1 for _, _, done in steps if done)
    pct = int(round((completed / len(steps)) * 100))
    
    st.markdown("<div class='vs-card'>", unsafe_allow_html=True)
    st.markdown("#### Pipeline Progress")
    
    # Progress percentage badge
    color = "#10b981" if pct == 100 else "#6366f1" if pct > 0 else "#9ca3af"
    st.markdown(
        f"""
        <div style='margin-bottom: 12px;'>
            <span style='display:inline-flex;align-items:center;padding:8px 16px;border-radius:20px;
                        background:{color};color:white;font-weight:700;font-size:16px;'>
                {pct}% Complete
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # Progress bar
    st.progress(pct / 100)
    
    # Step indicators
    st.markdown("<div style='margin-top: 20px;'>", unsafe_allow_html=True)
    for i, (label, desc, done) in enumerate(steps):
        is_active = st.session_state.current_step == i and not done
        
        if done:
            icon = "✅"
            color = "#10b981"
            bg_color = "rgba(16, 185, 129, 0.1)"
        elif is_active:
            icon = "⏳"
            color = "#6366f1"
            bg_color = "rgba(99, 102, 241, 0.1)"
        else:
            icon = "⭕"
            color = "#9ca3af"
            bg_color = "transparent"
        
        st.markdown(
            f"""
            <div style='display:flex;align-items:center;padding:10px;margin:6px 0;
                        border-radius:8px;background:{bg_color};border-left:3px solid {color};'>
                <span style='font-size:20px;margin-right:12px;'>{icon}</span>
                <div style='flex:1;'>
                    <div style='font-weight:600;color:{color};'>{label}</div>
                    <div style='font-size:12px;color:#6b7280;'>{desc}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def main() -> None:
    st.set_page_config(
        page_title="VarmaScribe Insurance", 
        layout="wide", 
        initial_sidebar_state="collapsed"
    )
    _init_state()

    assemblyai_key, gemini_key = _load_config()

    st.markdown(
        """
<style>
    /* Modern color scheme */
    :root {
        --primary: #6366f1;
        --success: #10b981;
        --warning: #f59e0b;
        --danger: #ef4444;
    }
    
    /* Hero section */
    .vs-hero {
        padding: 24px;
        border-radius: 16px;
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.1), rgba(16, 185, 129, 0.08));
        border: 1px solid rgba(99, 102, 241, 0.2);
        margin-bottom: 24px;
    }
    .vs-title { 
        font-size: 36px; 
        font-weight: 800; 
        margin: 0 0 8px 0;
        background: linear-gradient(135deg, #6366f1, #10b981);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .vs-subtitle { 
        font-size: 16px; 
        color: rgba(49, 51, 63, 0.75); 
        margin: 0; 
    }
    
    /* Card styling */
    .vs-card {
        border: 1px solid rgba(0, 0, 0, 0.1);
        border-radius: 12px;
        padding: 20px;
        background: white;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }
    
    /* Risk badge styling */
    .risk-badge {
        display: inline-block;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 14px;
    }
    .risk-low { background: #d1fae5; color: #065f46; }
    .risk-medium { background: #fef3c7; color: #92400e; }
    .risk-high { background: #fee2e2; color: #991b1b; }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .vs-title { font-size: 28px; }
        .vs-card { padding: 16px; }
    }
</style>
        """,
        unsafe_allow_html=True,
    )

    # Hero header
    st.markdown(
        """
<div class="vs-hero">
    <div class="vs-title">⚕️ VarmaScribe</div>
    <p class="vs-subtitle">AI-powered medical documentation and insurance claim optimization</p>
</div>
        """,
        unsafe_allow_html=True,
    )

    _configure_clients(assemblyai_key, gemini_key)
    insurance_style = INSURANCE_STYLE_DEFAULT

    # Main layout
    col1, col2 = st.columns([2, 1], gap="large")

    with col1:
        st.markdown("<div class='vs-card'>", unsafe_allow_html=True)
        st.markdown("#### 📁 Upload Audio File")
        st.caption("Supported: WAV, MP3, M4A, MP4, WEBM")

        audio_file = st.file_uploader(
            "Audio file",
            type=["wav", "mp3", "m4a", "mp4", "webm"],
            accept_multiple_files=False,
            label_visibility="collapsed",
        )

        col_btn1, col_btn2 = st.columns([3, 1])
        with col_btn1:
            process_button = st.button(
                "🚀 Generate Documentation",
                type="primary",
                use_container_width=True,
            )
        with col_btn2:
            reset_button = st.button("🔄 Reset", use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        _render_progress_card()

    # Reset functionality
    if reset_button:
        for key in ["transcript_text", "diarized_transcript_text", "soap_note", 
                    "hp_note", "insurance_note", "claim_analysis", "fix_mode", 
                    "additional_info", "current_step"]:
            if key in ["claim_analysis"]:
                st.session_state[key] = None
            elif key in ["fix_mode"]:
                st.session_state[key] = False
            elif key == "current_step":
                st.session_state[key] = 0
            else:
                st.session_state[key] = ""
        st.rerun()

    # Processing pipeline
    if process_button:
        if not audio_file:
            st.error("⚠️ Please upload an audio file first.")
        elif not assemblyai_key or not gemini_key:
            st.error("⚠️ Missing API keys. Please configure in .env file.")
        else:
            try:
                # Step 1: Transcribe
                st.session_state.current_step = 0
                with st.spinner("🎤 Transcribing audio with speaker diarization..."):
                    plain, diarized = _transcribe_audio_bytes_with_diarization(
                        audio_file.getvalue(), audio_file.name
                    )
                    st.session_state.transcript_text = plain
                    st.session_state.diarized_transcript_text = diarized
                st.success("✅ Transcription complete")
                st.rerun()

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.session_state.current_step = 0

    # Continue pipeline if transcript exists but other steps incomplete
    if st.session_state.transcript_text and not st.session_state.soap_note:
        try:
            st.session_state.current_step = 1
            with st.spinner("📝 Generating SOAP note..."):
                soap_prompt = _build_clinical_note_prompt(
                    st.session_state.diarized_transcript_text, "SOAP"
                )
                st.session_state.soap_note = _gemini_generate(soap_prompt, GEMINI_MODEL_NAME).strip()
            st.success("✅ SOAP note generated")
            st.rerun()
        except Exception as e:
            st.error(f"❌ Error generating SOAP note: {str(e)}")

    if st.session_state.soap_note and not st.session_state.hp_note:
        try:
            st.session_state.current_step = 2
            with st.spinner("📋 Generating H&P note..."):
                hp_prompt = _build_clinical_note_prompt(
                    st.session_state.diarized_transcript_text, "H&P"
                )
                st.session_state.hp_note = _gemini_generate(hp_prompt, GEMINI_MODEL_NAME).strip()
            st.success("✅ H&P note generated")
            st.rerun()
        except Exception as e:
            st.error(f"❌ Error generating H&P note: {str(e)}")

    if st.session_state.hp_note and not st.session_state.insurance_note:
        try:
            st.session_state.current_step = 3
            with st.spinner("💼 Generating insurance documentation..."):
                insurance_prompt = _build_insurance_prompt(
                    st.session_state.soap_note, insurance_style
                )
                st.session_state.insurance_note = _gemini_generate(
                    insurance_prompt, GEMINI_MODEL_NAME
                ).strip()
            st.success("✅ Insurance note generated")
            st.rerun()
        except Exception as e:
            st.error(f"❌ Error generating insurance note: {str(e)}")

    if st.session_state.insurance_note and not st.session_state.claim_analysis:
        try:
            st.session_state.current_step = 4
            with st.spinner("🔍 Analyzing claim readiness..."):
                analysis_prompt = _build_claim_analysis_prompt(
                    st.session_state.insurance_note,
                    st.session_state.soap_note
                )
                analysis_text = _gemini_generate(analysis_prompt, GEMINI_MODEL_NAME).strip()
                st.session_state.claim_analysis = _parse_claim_analysis(analysis_text)
            st.success("✅ Claim analysis complete!")
            st.session_state.current_step = 5
            st.rerun()
        except Exception as e:
            st.error(f"❌ Error analyzing claim: {str(e)}")

    # Display results
    if st.session_state.claim_analysis or st.session_state.insurance_note:
        st.divider()
        
        # Claim Analysis Section
        if st.session_state.claim_analysis:
            analysis = st.session_state.claim_analysis
            
            st.markdown("### 🎯 Claim Decision Support")
            
            # Risk overview
            risk_color_map = {
                "Low Risk": "risk-low",
                "Medium Risk": "risk-medium",
                "High Risk": "risk-high"
            }
            risk_icon = {
                "Low Risk": "🟢",
                "Medium Risk": "🟡",
                "High Risk": "🔴"
            }
            
            col_r1, col_r2 = st.columns([1, 2])
            with col_r1:
                risk_class = risk_color_map.get(analysis.risk_level, "risk-medium")
                st.markdown(
                    f"""
                    <div class='risk-badge {risk_class}'>
                        {risk_icon.get(analysis.risk_level, '⚪')} {analysis.risk_level}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            with col_r2:
                st.info(f"**Analysis:** {analysis.risk_explanation}")
            
            # CPT Codes
            if analysis.suggested_cpt_codes:
                with st.expander("💰 Suggested CPT Codes", expanded=True):
                    for code, description in analysis.suggested_cpt_codes:
                        st.markdown(f"**{code}** — {description}")
            
            # Missing elements with fix functionality
            if analysis.missing_elements:
                with st.expander("⚠️ Missing Documentation Elements", expanded=True):
                    for idx, (element, reason) in enumerate(analysis.missing_elements):
                        col_a, col_b = st.columns([5, 1])
                        with col_a:
                            st.markdown(f"**{element}**")
                            st.caption(reason)
                        with col_b:
                            if st.button("🔧", key=f"fix_{idx}", help="Add this information"):
                                st.session_state.fix_mode = True
                                st.session_state.fix_element = element
                                st.rerun()
            
            # Fix Mode
            if st.session_state.get("fix_mode"):
                st.markdown("---")
                st.markdown("### 🔧 Add Missing Information")
                st.info(f"**Adding:** {st.session_state.get('fix_element', 'N/A')}")
                
                additional_info = st.text_area(
                    "Enter factual clinical information from the encounter",
                    height=100,
                    placeholder="Example: BP 120/80, HR 88, Patient reports 8/10 pain...",
                )
                
                col_f1, col_f2 = st.columns(2)
                with col_f1:
                    if st.button("✅ Regenerate Documentation", type="primary", use_container_width=True):
                        if additional_info.strip():
                            with st.spinner("Regenerating with additional information..."):
                                try:
                                    enhanced = st.session_state.diarized_transcript_text + f"\n\nADDITIONAL INFO:\n{additional_info}"
                                    
                                    soap_prompt = _build_clinical_note_prompt(enhanced, "SOAP")
                                    st.session_state.soap_note = _gemini_generate(soap_prompt, GEMINI_MODEL_NAME).strip()
                                    
                                    hp_prompt = _build_clinical_note_prompt(enhanced, "H&P")
                                    st.session_state.hp_note = _gemini_generate(hp_prompt, GEMINI_MODEL_NAME).strip()
                                    
                                    ins_prompt = _build_insurance_prompt(st.session_state.soap_note, insurance_style)
                                    st.session_state.insurance_note = _gemini_generate(ins_prompt, GEMINI_MODEL_NAME).strip()
                                    
                                    analysis_prompt = _build_claim_analysis_prompt(
                                        st.session_state.insurance_note, st.session_state.soap_note
                                    )
                                    analysis_text = _gemini_generate(analysis_prompt, GEMINI_MODEL_NAME).strip()
                                    st.session_state.claim_analysis = _parse_claim_analysis(analysis_text)
                                    
                                    st.session_state.fix_mode = False
                                    st.success("✅ Documentation updated!")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Error: {str(e)}")
                
                with col_f2:
                    if st.button("❌ Cancel", use_container_width=True):
                        st.session_state.fix_mode = False
                        st.rerun()
            
            # Improvement suggestions
            if analysis.improvement_suggestions:
                with st.expander("💡 Improvement Suggestions"):
                    for i, suggestion in enumerate(analysis.improvement_suggestions, 1):
                        st.markdown(f"{i}. {suggestion}")
        
        # Documentation tabs
        st.divider()
        st.markdown("### 📄 Generated Documentation")
        
        tabs = st.tabs([
            "📝 Transcript",
            "👥 Diarized",
            "🩺 SOAP",
            "📋 H&P",
            "💼 Insurance"
        ])

        with tabs[0]:
            if st.session_state.transcript_text:
                st.text_area("Plain Transcript", st.session_state.transcript_text, height=400, key="t1")

        with tabs[1]:
            if st.session_state.diarized_transcript_text:
                st.text_area("Diarized Transcript", st.session_state.diarized_transcript_text, height=400, key="t2")

        with tabs[2]:
            if st.session_state.soap_note:
                st.text_area("SOAP Note", st.session_state.soap_note, height=400, key="t3")

        with tabs[3]:
            if st.session_state.hp_note:
                st.text_area(
                    "H&P Note",
                    st.session_state.hp_note,
                    height=400,
                    key="t4"
                )

        with tabs[4]:
            if st.session_state.insurance_note:
                st.text_area(
                    "Insurance Documentation",
                    st.session_state.insurance_note,
                    height=400,
                    key="t5"
                )

                st.download_button(
                    label="📥 Download Insurance Note",
                    data=st.session_state.insurance_note,
                    file_name="insurance_note.txt",
                    mime="text/plain",
                )


if __name__ == "__main__":
    main()
