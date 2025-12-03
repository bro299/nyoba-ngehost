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
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"Error encoding image: {e}")
        return None

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

def process_video_frames(video_path, max_frames=3):
    """
    Mengambil frame dari video untuk analisis AI.
    max_frames: jumlah maksimal frame yang diambil
    """
    frames_base64 = []
    cap = None
    
    try:
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print("Error: Tidak bisa membuka video")
            return []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            print("Error: Video tidak memiliki frame")
            return []

        # Ambil frame di titik yang merata
        points = [int(total_frames * (i + 1) / (max_frames + 1)) for i in range(max_frames)]

        for p in points:
            cap.set(cv2.CAP_PROP_POS_FRAMES, p)
            ret, frame = cap.read()
            if ret:
                # Resize untuk menghemat bandwidth
                frame = cv2.resize(frame, (640, 360))
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                frames_base64.append(base64.b64encode(buffer).decode('utf-8'))
            
            # Break jika sudah cukup frame
            if len(frames_base64) >= max_frames:
                break

    except Exception as e:
        print(f"Error processing video: {e}")
    finally:
        if cap is not None:
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
        if context_data['content']:
            user_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{context_data['content']}"
                }
            })

    elif context_data['type'] == 'video_frames':
        if context_data['content']:
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
    try:
        return send_from_directory('public', 'index.html')
    except Exception as e:
        return jsonify({
            "error": "Frontend tidak ditemukan",
            "message": "Pastikan folder 'public' dan file 'index.html' ada"
        }), 404

@app.route('/health')
def health():
    """Health check endpoint untuk Render.com"""
    return jsonify({
        "status": "healthy",
        "api_configured": ai_client is not None,
        "api_key_set": API_KEY is not None
    })

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        user_message = request.form.get('message', '')
        
        if not user_message:
            return jsonify({"error": "Pesan tidak boleh kosong"}), 400
        
        context_data = {'type': 'none', 'content': ''}

        if 'file' in request.files:
            file = request.files['file']
            if file and file.filename and allowed_file(file.filename):
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
                            with open(file_path, 'r', encoding='utf-8') as f:
                                context_data['content'] = f.read()
                        except Exception as e:
                            context_data['content'] = f"[Gagal membaca teks file: {e}]"

                elif ext in ['jpg', 'jpeg', 'png']:
                    context_data['type'] = 'image'
                    encoded = encode_image_to_base64(file_path)
                    if encoded:
                        context_data['content'] = encoded
                    else:
                        return jsonify({"error": "Gagal membaca gambar"}), 400

                elif ext in ['mp4', 'mov', 'avi']:
                    frames = process_video_frames(file_path, max_frames=3)
                    if frames:
                        context_data['type'] = 'video_frames'
                        context_data['content'] = frames
                    else:
                        return jsonify({"error": "❌ Gagal membaca video"}), 400

                # Hapus file setelah diproses untuk hemat storage
                try:
                    os.remove(file_path)
                except Exception as e:
                    print(f"Warning: Gagal menghapus file {file_path}: {e}")

        ai_reply = call_ai_api(user_message, context_data)

        return jsonify({
            "reply": ai_reply
        })
    
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        return jsonify({
            "error": "Terjadi kesalahan pada server",
            "message": str(e)
        }), 500

if __name__ == '__main__':
    # Gunakan port dari environment variable (untuk Render.com)
    port = int(os.getenv('PORT', 8000))
    # Di production (Render), debug harus False
    debug = os.getenv('FLASK_ENV') != 'production'
    
    app.run(debug=debug, host='0.0.0.0', port=port)
