"""
DeepFake Detection Flask Backend
Loads deepfake_ensemble.pkl, runs inference + Grad-CAM, streams analysis to frontend.
"""

import os, io, base64, pickle, time, uuid, json
from pathlib import Path

import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn as nn
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from flask import Flask, request, jsonify, render_template, Response, stream_with_context
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# ── Optional Ollama support ─────────────────────────────────────────────────
try:
    import requests as req
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100 MB
UPLOAD_FOLDER  = Path('static/uploads')
RESULTS_FOLDER = Path('static/results')
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)

DEVICE   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PKL_PATH = 'deepfake_ensemble.pkl'

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

# ────────────────────────────────────────────────────────────────────────────
# Model definitions (must match training code exactly)
# ────────────────────────────────────────────────────────────────────────────

class XceptionDetector(nn.Module):
    def __init__(self, pretrained=False, dropout=0.5):
        super().__init__()
        self.backbone   = timm.create_model('xception', pretrained=pretrained,
                                             num_classes=0, global_pool='avg')
        feats           = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feats, 512), nn.ReLU(True),
            nn.Dropout(dropout / 2),
            nn.Linear(512, 2)
        )
    def forward(self, x):
        return self.classifier(self.backbone(x))


class DCTBranch(nn.Module):
    def __init__(self, out_features=256):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((64, 64))
        self.net  = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(True),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64 * 16, out_features), nn.ReLU(True)
        )
    def dct_spectrum(self, x):
        gray = 0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]
        freq = torch.fft.rfft2(gray, norm='ortho')
        mag  = torch.log1p(freq.abs())
        return mag.unsqueeze(1).expand(-1, 3, -1, -1)
    def forward(self, x):
        return self.net(self.pool(self.dct_spectrum(x)))


class EfficientNetDCT(nn.Module):
    def __init__(self, pretrained=False, dropout=0.4):
        super().__init__()
        self.backbone   = timm.create_model('efficientnet_b4', pretrained=pretrained,
                                             num_classes=0, global_pool='avg')
        feat            = self.backbone.num_features
        self.dct        = DCTBranch(out_features=256)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat + 256, 512), nn.ReLU(True),
            nn.Dropout(dropout / 2),
            nn.Linear(512, 2)
        )
    def forward(self, x):
        return self.classifier(torch.cat([self.backbone(x), self.dct(x)], dim=1))


# ────────────────────────────────────────────────────────────────────────────
# Load ensemble from PKL
# ────────────────────────────────────────────────────────────────────────────

def load_ensemble():
    if not Path(PKL_PATH).exists():
        print(f"[WARNING] {PKL_PATH} not found — running in demo mode with random weights.")
        m_xcp = XceptionDetector(pretrained=False)
        m_eff = EfficientNetDCT(pretrained=False)
        return m_xcp, m_eff, 0.5, 0.5

    print(f"Loading ensemble from {PKL_PATH} ...")
    with open(PKL_PATH, 'rb') as f:
        pkg = pickle.load(f)

    m_xcp = XceptionDetector(pretrained=False)
    m_xcp.load_state_dict(pkg['xception_state'])

    m_eff = EfficientNetDCT(pretrained=False)
    m_eff.load_state_dict(pkg['efficientdct_state'])

    w_xcp = pkg.get('weight_xception', 0.5)
    w_eff = pkg.get('weight_efficientdct', 0.5)
    print(f"Loaded | w_xcp={w_xcp:.3f} w_eff={w_eff:.3f}")
    return m_xcp, m_eff, w_xcp, w_eff


MODEL_XCP, MODEL_EFF, W_XCP, W_EFF = load_ensemble()
MODEL_XCP.to(DEVICE).eval()
MODEL_EFF.to(DEVICE).eval()

CAM_XCP = GradCAMPlusPlus(model=MODEL_XCP, target_layers=[MODEL_XCP.backbone.block12.rep[-1]])
CAM_EFF = GradCAMPlusPlus(model=MODEL_EFF, target_layers=[MODEL_EFF.backbone.conv_head])

VAL_TFM = A.Compose([
    A.Resize(299, 299),
    A.Normalize(mean=MEAN, std=STD),
    ToTensorV2()
])


# ────────────────────────────────────────────────────────────────────────────
# Core analysis helpers
# ────────────────────────────────────────────────────────────────────────────

def img_to_b64(arr_rgb):
    buf = io.BytesIO()
    Image.fromarray(arr_rgb.astype(np.uint8)).save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode()


def analyse_frame(img_rgb):
    """Run ensemble + Grad-CAM on a single 299x299 RGB frame."""
    img_rs  = cv2.resize(img_rgb, (299, 299))
    img_f   = img_rs.astype(np.float32) / 255.0
    tensor  = VAL_TFM(image=img_rs)['image'].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        p_xcp = float(torch.softmax(MODEL_XCP(tensor), 1)[0, 1])
        p_eff = float(torch.softmax(MODEL_EFF(tensor), 1)[0, 1])

    p_ens   = W_XCP * p_xcp + W_EFF * p_eff
    label   = 'FAKE' if p_ens > 0.5 else 'REAL'

    # Grad-CAM for the predicted class
    cls = 1 if p_ens > 0.5 else 0
    cam_xcp_mask = CAM_XCP(input_tensor=tensor, targets=[ClassifierOutputTarget(cls)])[0]
    cam_eff_mask = CAM_EFF(input_tensor=tensor, targets=[ClassifierOutputTarget(cls)])[0]
    cam_avg_mask = 0.5 * cam_xcp_mask + 0.5 * cam_eff_mask

    overlay_xcp = show_cam_on_image(img_f, cam_xcp_mask, use_rgb=True,
                                    colormap=cv2.COLORMAP_JET, image_weight=0.5)
    overlay_eff = show_cam_on_image(img_f, cam_eff_mask, use_rgb=True,
                                    colormap=cv2.COLORMAP_INFERNO, image_weight=0.5)
    overlay_avg = show_cam_on_image(img_f, cam_avg_mask, use_rgb=True,
                                    colormap=cv2.COLORMAP_JET, image_weight=0.5)

    return {
        'label':        label,
        'fake_prob':    round(p_ens * 100, 1),
        'prob_xcp':     round(p_xcp * 100, 1),
        'prob_eff':     round(p_eff * 100, 1),
        'cam_energy':   float(cam_avg_mask.mean()),
        'original_b64': img_to_b64(img_rs),
        'overlay_b64':  img_to_b64(overlay_avg),
        'xcp_cam_b64':  img_to_b64(overlay_xcp),
        'eff_cam_b64':  img_to_b64(overlay_eff),
    }


def ask_ollama(orig_b64, cam_b64, label, confidence):
    if not OLLAMA_AVAILABLE:
        return "Ollama not available in this environment."
    prompt = (
        f"You are a forensic deepfake analyst. The ensemble AI model classified this face as "
        f"{label} with {confidence:.1f}% confidence.\n"
        f"Image 1 = original face. Image 2 = Grad-CAM++ heatmap (red=high model attention, blue=low).\n"
        f"Explain concisely in 4 points:\n"
        f"1. Which facial regions are highlighted in the heatmap\n"
        f"2. What specific visual artifacts suggest manipulation in those regions\n"
        f"3. Common deepfake tells present: blending edges, skin texture, lighting, eye reflections\n"
        f"4. Plain-English verdict for a non-technical user"
    )
    try:
        r = req.post('http://localhost:11434/api/generate', json={
            'model': 'llava', 'prompt': prompt,
            'images': [orig_b64, cam_b64], 'stream': False
        }, timeout=90)
        return r.json().get('response', 'No response from Ollama.')
    except Exception as e:
        return f"Ollama unavailable: {e}"


# ────────────────────────────────────────────────────────────────────────────
# Routes
# ────────────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyse', methods=['POST'])
def analyse():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    ext  = Path(file.filename).suffix.lower()
    uid  = uuid.uuid4().hex[:8]
    path = UPLOAD_FOLDER / f"{uid}{ext}"
    file.save(path)

    is_video  = ext in {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
    n_frames  = int(request.form.get('n_frames', 8))
    use_llava = request.form.get('use_llava', 'false') == 'true'

    def stream_analysis():
        try:
            # ── Extract frames ─────────────────────────────────────────────
            if is_video:
                cap     = cv2.VideoCapture(str(path))
                total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                indices = [int(i * total / n_frames) for i in range(n_frames)]
                frames  = []
                fi      = 0
                while True:
                    ret, fr = cap.read()
                    if not ret: break
                    if fi in indices:
                        frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
                    fi += 1
                cap.release()
            else:
                img    = np.array(Image.open(path).convert('RGB'))
                frames = [img]

            total_frames = len(frames)
            yield f"data: {json.dumps({'type':'progress','msg':f'Analysing {total_frames} frame(s)...','pct':5})}\n\n"

            # ── Analyse each frame ─────────────────────────────────────────
            results = []
            for i, fr in enumerate(frames):
                res = analyse_frame(fr)
                results.append(res)
                pct = int(10 + (i + 1) / total_frames * 70)
                yield f"data: {json.dumps({'type':'frame','frame_idx':i,'total':total_frames,'result':res,'pct':pct})}\n\n"

            # ── Aggregate verdict ──────────────────────────────────────────
            fake_votes = sum(1 for r in results if r['label'] == 'FAKE')
            mean_prob  = round(sum(r['fake_prob'] for r in results) / len(results), 1)
            final      = 'FAKE' if fake_votes > len(results) / 2 else 'REAL'

            # Pick best frame (highest cam_energy)
            best = max(results, key=lambda r: r['cam_energy'])

            yield f"data: {json.dumps({'type':'progress','msg':'Computing ensemble verdict...','pct':82})}\n\n"

            # ── Ollama explanation ─────────────────────────────────────────
            explanation = ""
            if use_llava:
                yield f"data: {json.dumps({'type':'progress','msg':'Asking LLaVA to explain...','pct':88})}\n\n"
                explanation = ask_ollama(best['original_b64'], best['overlay_b64'], final, mean_prob)

            # ── Save result image ──────────────────────────────────────────
            result_filename = f"{uid}_result.png"
            overlay_arr     = np.frombuffer(base64.b64decode(best['overlay_b64']), dtype=np.uint8)
            # decode from base64 PNG
            buf = io.BytesIO(base64.b64decode(best['overlay_b64']))
            Image.open(buf).save(RESULTS_FOLDER / result_filename)

            final_payload = {
                'type':        'done',
                'verdict':     final,
                'fake_prob':   mean_prob,
                'fake_votes':  fake_votes,
                'total_frames': total_frames,
                'best_frame':  best,
                'all_probs':   [r['fake_prob'] for r in results],
                'explanation': explanation,
                'result_img':  f'/static/results/{result_filename}',
                'pct': 100
            }
            yield f"data: {json.dumps(final_payload)}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type':'error','msg':str(e)})}\n\n"
        finally:
            try:
                path.unlink(missing_ok=True)
            except Exception:
                pass

    return Response(stream_with_context(stream_analysis()),
                    mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


@app.route('/health')
def health():
    return jsonify({
        'status': 'ok',
        'device': str(DEVICE),
        'pkl_loaded': Path(PKL_PATH).exists(),
        'ollama': OLLAMA_AVAILABLE
    })


if __name__ == '__main__':
    print(f"Device: {DEVICE}")
    print(f"PKL loaded: {Path(PKL_PATH).exists()}")
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)
