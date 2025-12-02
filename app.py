import os
import cv2
import base64
import openai as kolosal
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import PyPDF2
from PIL import Image
import io
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- Konfigurasi ---
app = Flask(__name__, static_folder='public', static_url_path='')
CORS(app)

# KONFIGURASI API KOLOSAL AI
# Render.com akan membaca dari environment variables
API_KEY = os.getenv("KOLOSAL_API_KEY")
KOLOSAL_BASE_URL = "https://api.kolosal.ai/v1"
MODEL_NAME = "Claude Sonnet 4.5"

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'mp4', 'mov', 'avi'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size

# --- Inisialisasi Klien AI ---
ai_client = None

def initialize_ai_client():
    global ai_client
    if not API_KEY:
        print("WARNING: KOLOSAL_API_KEY tidak ditemukan di environment variables!")
        return False
    
    try:
        ai_client = kolosal.OpenAI(
            api_key=API_KEY,
            base_url=KOLOSAL_BASE_URL
        )
        print("✓ Klien Kolosal AI berhasil diinisialisasi.")
        return True
    except Exception as e:
        print(f"ERROR: Gagal menginisialisasi Klien AI: {e}")
        return False

# Inisialisasi saat startup
initialize_ai_client()

# --- Helper Functions ---

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def encode_image_to_base64(image_path):
    """Mengubah file gambar menjadi string base64"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def extract_text_from_pdf(file_path):
    """Mengekstrak teks dari file PDF"""
    text = ""
    try:
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text() or ""
                text += page_text + "\n"
    except Exception as e:
        print(f"PDF Error: {e}")
        text = f"[Error membaca PDF: {e}]"
    return text

def process_video_frames(video_path):
    """
    Mengambil 3 frame dari video (awal, tengah, akhir) untuk analisis AI.
    """
    frames_base64 = []
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        return []

    # Ambil frame di titik 10%, 50%, dan 90% durasi
    points = [int(total_frames * 0.1), int(total_frames * 0.5), int(total_frames * 0.9)]

    for p in points:
        cap.set(cv2.CAP_PROP_POS_FRAMES, p)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (640, 360))
            _, buffer = cv2.imencode('.jpg', frame)
            frames_base64.append(base64.b64encode(buffer).decode('utf-8'))

    cap.release()
    return frames_base64

# --- Integrasi AI ---

def call_ai_api(user_text, context_data):
    """
    Mengirim request ke API Kolosal (OpenAI Compatible).
    """
    if ai_client is None:
        return "⚠️ Maaf, sistem AI belum terkonfigurasi dengan benar. Pastikan API Key telah diatur di environment variables."

    # System Instruction
    system_instruction = "Anda adalah Asisten Keuangan UMKM ahli. Analisis dokumen, gambar struk, atau video kondisi toko yang diberikan. Berikan saran praktis, hemat, dan ramah. Respon dalam Bahasa Indonesia."

    # Prepare user content
    user_content = [{"type": "text", "text": user_text}]

    # Add Context Data
    if context_data['type'] == 'text':
        user_content.append({"type": "text", "text": f"\n\nISI DOKUMEN:\n{context_data['content']}"})

    elif context_data['type'] == 'image':
        user_content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{context_data['content']}"
            }
        })

    elif context_data['type'] == 'video_frames':
        user_content.append({"type": "text", "text": "Berikut adalah beberapa frame dari video yang diunggah user:"})
        for frame in context_data['content']:
            user_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{frame}"
                }
            })

    try:
        response = ai_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_content}
            ],
            max_tokens=2048,
            temperature=0.7
        )
        return response.choices[0].message.content

    except Exception as e:
        print(f"API Error: {e}")
        return f"⚠️ Maaf, terjadi kesalahan saat menghubungi AI: {str(e)}"

# --- Routes ---

@app.route('/')
def index():
    """Serve frontend dari folder public"""
    return send_from_directory('public', 'index.html')

@app.route('/health')
def health():
    """Health check endpoint untuk Render.com"""
    return jsonify({
        "status": "healthy",
        "api_configured": ai_client is not None
    })

@app.route('/api/chat', methods=['POST'])
def chat():
    user_message = request.form.get('message', '')
    context_data = {'type': 'none', 'content': ''}

    if 'file' in request.files:
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            ext = filename.rsplit('.', 1)[1].lower()

            # Proses berdasarkan tipe file
            if ext == 'pdf' or ext == 'txt':
                context_data['type'] = 'text'
                if ext == 'pdf':
                    context_data['content'] = extract_text_from_pdf(file_path)
                else:
                    try:
                        context_data['content'] = open(file_path, 'r', encoding='utf-8').read()
                    except Exception as e:
                        context_data['content'] = f"[Gagal membaca teks file: {e}]"

            elif ext in ['jpg', 'jpeg', 'png']:
                context_data['type'] = 'image'
                context_data['content'] = encode_image_to_base64(file_path)

            elif ext in ['mp4', 'mov', 'avi']:
                frames = process_video_frames(file_path)
                if frames:
                    context_data['type'] = 'video_frames'
                    context_data['content'] = frames
                else:
                    return jsonify({"reply": "❌ Gagal membaca video."})

            # Hapus file setelah diproses untuk hemat storage
            try:
                os.remove(file_path)
            except:
                pass

    ai_reply = call_ai_api(user_message, context_data)

    return jsonify({
        "reply": ai_reply
    })

if __name__ == '__main__':
    # Gunakan port dari environment variable (untuk Render.com)
    port = int(os.getenv('PORT', 8000))
    # Di production (Render), debug harus False
    debug = os.getenv('FLASK_ENV') != 'production'
    
    app.run(debug=debug, host='0.0.0.0', port=port)