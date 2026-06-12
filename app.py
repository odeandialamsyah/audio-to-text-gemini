import os
import time
import tempfile
import subprocess
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv
from google import genai
import imageio_ffmpeg


# =========================
# CONFIG
# =========================
load_dotenv()

st.set_page_config(
    page_title="Audio to Text - Gemini",
    page_icon="🎙️",
    layout="centered"
)

API_KEY = os.getenv("GEMINI_API_KEY")


# =========================
# STYLE
# =========================
st.markdown(
    """
    <style>
    .main-title {
        text-align: center;
        font-size: 36px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .sub-title {
        text-align: center;
        font-size: 16px;
        color: #666;
        margin-bottom: 30px;
    }

    .footer {
        text-align: center;
        color: #888;
        font-size: 13px;
        margin-top: 40px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================
# HELPER FUNCTIONS
# =========================
def get_ffmpeg_path() -> str:
    """
    Mengambil path FFmpeg dari package imageio-ffmpeg.
    Jadi tidak perlu install FFmpeg manual ke Windows PATH.
    """
    return imageio_ffmpeg.get_ffmpeg_exe()


def check_ffmpeg_installed() -> bool:
    """
    Mengecek apakah FFmpeg dari imageio-ffmpeg tersedia.
    """
    try:
        ffmpeg_path = get_ffmpeg_path()

        result = subprocess.run(
            [ffmpeg_path, "-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        return result.returncode == 0

    except Exception:
        return False


def save_uploaded_file(uploaded_file) -> str:
    """
    Menyimpan file upload dari Streamlit ke file sementara.
    """
    suffix = os.path.splitext(uploaded_file.name)[1]

    if not suffix:
        suffix = ".audio"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        return tmp.name


def convert_to_mp3(input_path: str) -> str:
    """
    Convert audio ke MP3 menggunakan FFmpeg dari package imageio-ffmpeg.
    Tidak perlu install FFmpeg manual.
    """
    ffmpeg_path = get_ffmpeg_path()
    output_path = input_path + "_converted.mp3"

    command = [
        ffmpeg_path,
        "-y",
        "-i", input_path,
        "-vn",
        "-acodec", "libmp3lame",
        "-ar", "44100",
        "-ac", "2",
        "-b:a", "128k",
        output_path
    ]

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError(
            "Gagal convert audio ke MP3.\n\n"
            f"Detail error:\n{result.stderr}"
        )

    return output_path


def wait_for_file_active(client, uploaded_file, timeout_seconds=600):
    """
    Menunggu file Gemini selesai diproses.
    Untuk audio besar, status awal biasanya PROCESSING.
    """
    start_time = time.time()

    file = client.files.get(name=uploaded_file.name)

    while file.state.name == "PROCESSING":
        elapsed = time.time() - start_time

        if elapsed > timeout_seconds:
            raise TimeoutError(
                "File terlalu lama diproses oleh Gemini. "
                "Coba potong audio menjadi beberapa bagian lebih kecil."
            )

        time.sleep(3)
        file = client.files.get(name=uploaded_file.name)

    if file.state.name != "ACTIVE":
        raise ValueError(f"File gagal diproses. Status file: {file.state.name}")

    return file


def build_prompt(
    language: str,
    clean_text: bool,
    speaker_label: bool,
    timestamp_mode: bool
) -> str:
    """
    Membuat prompt transkripsi sesuai pilihan user.
    """
    speaker_rule = (
        "- If multiple speakers are detected, label them as Speaker 1, Speaker 2, and so on."
        if speaker_label
        else "- Do not add speaker labels unless they are clearly mentioned in the audio."
    )

    timestamp_rule = (
        "- Add timestamps every time the topic changes or every 1-2 minutes using format [00:00]."
        if timestamp_mode
        else "- Do not add timestamps."
    )

    if clean_text:
        prompt = f"""
Transcribe this audio into {language}.

Rules:
- Output only the transcript.
- Do not summarize.
- Do not skip any important speech.
- Use proper punctuation and paragraph breaks.
- Remove filler words such as "eee", "hmm", "anu", and repeated stutters.
- If there are unclear words, write [unclear].
{speaker_rule}
{timestamp_rule}
- Do not add explanation, notes, or comments outside the transcript.
"""
    else:
        prompt = f"""
Transcribe this audio into {language}.

Rules:
- Output only the transcript.
- Keep the speech as close as possible to the original.
- Do not summarize.
- Do not skip any important speech.
- Keep filler words and repeated words if they exist in the audio.
- If there are unclear words, write [unclear].
{speaker_rule}
{timestamp_rule}
- Do not add explanation, notes, or comments outside the transcript.
"""

    return prompt


def transcribe_audio(
    audio_path: str,
    language: str,
    clean_text: bool,
    speaker_label: bool,
    timestamp_mode: bool,
    model_name: str
) -> str:
    """
    Upload audio MP3 ke Gemini Files API, tunggu file aktif,
    lalu kirim ke model Gemini untuk transkripsi.
    """
    if not API_KEY:
        raise ValueError("GEMINI_API_KEY belum diatur di file .env")

    client = genai.Client(api_key=API_KEY)

    uploaded_file = client.files.upload(file=audio_path)
    uploaded_file = wait_for_file_active(client, uploaded_file)

    prompt = build_prompt(
        language=language,
        clean_text=clean_text,
        speaker_label=speaker_label,
        timestamp_mode=timestamp_mode
    )

    response = client.models.generate_content(
        model=model_name,
        contents=[
            uploaded_file,
            prompt
        ]
    )

    if not response.text:
        raise ValueError("Gemini tidak mengembalikan hasil transkrip.")

    return response.text


# =========================
# UI
# =========================
st.markdown(
    '<div class="main-title">🎙️ Audio to Text</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">Convert audio menjadi teks menggunakan Gemini API</div>',
    unsafe_allow_html=True
)


with st.sidebar:
    st.header("⚙️ Pengaturan")

    model_name = st.selectbox(
        "Model Gemini",
        [
            "gemini-2.5-flash",
            "gemini-2.5-pro"
        ],
        index=0
    )

    language = st.selectbox(
        "Bahasa transkrip",
        [
            "Indonesian",
            "English",
            "Mixed Indonesian, Javanese and English",
            "Sundanese",
            "Javanese"
        ],
        index=0
    )

    clean_text = st.toggle(
        "Rapikan teks otomatis",
        value=True
    )

    speaker_label = st.toggle(
        "Deteksi beberapa pembicara",
        value=True
    )

    timestamp_mode = st.toggle(
        "Tambahkan timestamp",
        value=False
    )

    auto_convert = st.toggle(
        "Convert otomatis ke MP3",
        value=True
    )

    st.caption(
        "Aktifkan convert otomatis jika formatnya terbaca sebagai MPEG."
    )


# =========================
# CHECK FFmpeg
# =========================
if auto_convert and not check_ffmpeg_installed():
    st.warning(
        "FFmpeg belum terdeteksi. Fitur convert otomatis ke MP3 tidak bisa digunakan. "
        "Install FFmpeg terlebih dahulu atau matikan opsi convert otomatis."
    )


# =========================
# UPLOAD AUDIO
# =========================
uploaded_audio = st.file_uploader(
    "Upload file audio",
    type=[
        "mp3",
        "wav",
        "m4a",
        "ogg",
        "flac",
        "mpeg",
        "mpga",
        "aac"
    ]
)


if uploaded_audio is not None:
    file_size_mb = uploaded_audio.size / (1024 * 1024)

    st.info(f"Ukuran file: {file_size_mb:.2f} MB")
    st.caption(f"Nama file: {uploaded_audio.name}")
    st.caption(f"Tipe file terbaca: {uploaded_audio.type}")

    if file_size_mb > 2000:
        st.error("Ukuran file terlalu besar. Maksimal sekitar 2 GB per file.")
        st.stop()

    st.audio(uploaded_audio)

    transcribe_button = st.button(
        "🚀 Convert ke Text",
        use_container_width=True
    )

    if transcribe_button:
        audio_path = None
        converted_audio_path = None

        try:
            with st.spinner("Menyimpan file audio sementara..."):
                audio_path = save_uploaded_file(uploaded_audio)

            final_audio_path = audio_path

            if auto_convert:
                if not check_ffmpeg_installed():
                    raise RuntimeError(
                        "FFmpeg belum terinstall. "
                        "Install FFmpeg terlebih dahulu atau matikan opsi convert otomatis."
                    )

                with st.spinner("Mengubah audio ke MP3 agar lebih stabil diproses..."):
                    converted_audio_path = convert_to_mp3(audio_path)
                    final_audio_path = converted_audio_path

            with st.spinner("Mengupload audio ke Gemini dan membuat transkrip..."):
                transcript = transcribe_audio(
                    audio_path=final_audio_path,
                    language=language,
                    clean_text=clean_text,
                    speaker_label=speaker_label,
                    timestamp_mode=timestamp_mode,
                    model_name=model_name
                )

            st.success("Transkrip berhasil dibuat!")

            st.subheader("📝 Hasil Transkrip")

            st.text_area(
                "Transcript",
                value=transcript,
                height=450
            )

            filename = f"transcript_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

            st.download_button(
                "⬇️ Download Transkrip TXT",
                data=transcript,
                file_name=filename,
                mime="text/plain",
                use_container_width=True
            )

        except Exception as e:
            st.error(f"Terjadi error: {e}")

        finally:
            if audio_path and os.path.exists(audio_path):
                os.remove(audio_path)

            if converted_audio_path and os.path.exists(converted_audio_path):
                os.remove(converted_audio_path)

else:
    st.info("Silakan upload audio terlebih dahulu.")


st.markdown(
    '<div class="footer">Built with Streamlit + Gemini API</div>',
    unsafe_allow_html=True
)