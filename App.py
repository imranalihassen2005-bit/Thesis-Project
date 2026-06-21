"""
TB-CAD Clinical Suite — Main Flask Application
AI-Based Computer-Aided Diagnosis System for Tuberculosis Detection
"""

import os
import time
import json
import base64
import numpy as np
from datetime import datetime
from flask import (Flask, render_template, request, jsonify, 
                   send_from_directory, url_for)

# ── App Configuration ──
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
GRADCAM_FOLDER = os.path.join(BASE_DIR, 'static', 'gradcam')
MODEL_DIR = os.path.join(BASE_DIR, 'Models')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(GRADCAM_FOLDER, exist_ok=True)

app = Flask(__name__,
            template_folder=os.path.join(BASE_DIR, 'templates'),
            static_folder=os.path.join(BASE_DIR, 'static'))
app.config['MAX_CONTENT_LENGTH'] = 25 * 1024 * 1024  # 25 MB

# ── Class Labels ──
CLASS_NAMES = ['Normal', 'Pneumonia', 'Tuberculosis']

# ── Load Model ──
# Suppress TF warnings (must be before import)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
print("[*] Loading AI model...")
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

model = None

try:
    model_path = os.path.join(MODEL_DIR, 'multiclass_model_100_epochs.keras')
    model = tf.keras.models.load_model(model_path)
    print(f"[OK] Loaded model from {model_path}")
except Exception as e:
    print(f"[ERROR] Could not load model: {e}")

# ── Import project modules ──
from database import (init_db, insert_scan, get_recent_scans, get_scan,
                       get_total_scans, get_most_frequent_prediction,
                       get_prediction_distribution, get_confidence_distribution,
                       get_model_metrics, insert_note, get_notes,
                       insert_chat, get_chats, get_latest_scan)
from gradcam import (generate_gradcam, create_gradcam_overlay, 
                      preprocess_image, preprocess_base64_frame)
from assistant import generate_response, get_severity

# Initialize DB
init_db()
print("[OK] Database initialized")


# ════════════════════════════════════════
#  PAGE ROUTES
# ════════════════════════════════════════

@app.route('/')
def home():
    """Home Dashboard."""
    return render_template('index.html')


@app.route('/chest-xray')
def chest_xray():
    """Chest X-Ray Analysis Workbench."""
    return render_template('chest_xray.html')


@app.route('/visual-analytics')
def visual_analytics():
    """Visual Analytics Dashboard."""
    return render_template('visual_analytics.html')


@app.route('/ai-assistant')
def ai_assistant():
    """AI Assistant Chat Interface."""
    return render_template('ai_assistant.html')


@app.route('/clinical-recommendations')
def clinical_recommendations():
    """Clinical Recommendations Page."""
    return render_template('clinical_recommendations.html')


@app.route('/model-insights')
def model_insights():
    """Model Insights & Architecture Page."""
    return render_template('model_insights.html')


# ════════════════════════════════════════
#  API ROUTES
# ════════════════════════════════════════

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """
    Upload an image, run model prediction, generate Grad-CAM, save to DB.
    Returns prediction results as JSON.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Validate file type
    allowed_ext = {'png', 'jpg', 'jpeg'}
    ext = file.filename.rsplit('.', 1)[-1].lower() if '.' in file.filename else ''
    if ext not in allowed_ext:
        return jsonify({'error': f'Invalid file type. Allowed: {", ".join(allowed_ext)}'}), 400
    
    if model is None:
        return jsonify({'error': 'Model not loaded. Check server logs.'}), 500

    start_time = time.time()
    
    # Save uploaded file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    safe_name = f"scan_{timestamp}.{ext}"
    save_path = os.path.join(UPLOAD_FOLDER, safe_name)
    file.save(save_path)
    
    # Preprocess
    img_array = preprocess_image(save_path)
    
    # Predict
    predictions = model.predict(img_array, verbose=0)
    pred_probs = predictions[0]
    pred_index = int(np.argmax(pred_probs))
    pred_class = CLASS_NAMES[pred_index]
    confidence = float(pred_probs[pred_index]) * 100
    
    all_confidences = {
        CLASS_NAMES[i]: float(pred_probs[i]) * 100 
        for i in range(len(CLASS_NAMES))
    }
    
    # Generate Grad-CAM
    heatmap_filename = None
    if pred_class in ['Tuberculosis', 'Pneumonia']:
        try:
            heatmap = generate_gradcam(model, img_array, pred_index)
            heatmap_filename = f"gradcam_{timestamp}.png"
            heatmap_path = os.path.join(GRADCAM_FOLDER, heatmap_filename)
            create_gradcam_overlay(save_path, heatmap, heatmap_path)
        except Exception as e:
            print(f"[WARN] Grad-CAM generation failed: {e}")
            heatmap_filename = None
    
    processing_time = round(time.time() - start_time, 2)
    
    # Save to database
    scan_id = insert_scan(
        image_name=safe_name,
        prediction=pred_class,
        confidence=round(confidence, 2),
        all_confidences=all_confidences,
        heatmap_path=heatmap_filename,
        processing_time=processing_time
    )
    
    # Determine severity
    severity = get_severity(pred_class, confidence)
    
    return jsonify({
        'scan_id': scan_id,
        'prediction': pred_class,
        'confidence': round(confidence, 2),
        'all_confidences': all_confidences,
        'severity': severity,
        'heatmap_path': f'/static/gradcam/{heatmap_filename}' if heatmap_filename else None,
        'image_path': f'/static/uploads/{safe_name}',
        'processing_time': processing_time
    })


@app.route('/api/webcam-predict', methods=['POST'])
def api_webcam_predict():
    """
    Accept a base64-encoded webcam frame, predict, and return result.
    Optimized for speed — no Grad-CAM by default.
    """
    data = request.get_json()
    if not data or 'frame' not in data:
        return jsonify({'error': 'No frame data provided'}), 400
    
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    start_time = time.time()
    
    try:
        img_array = preprocess_base64_frame(data['frame'])
        predictions = model.predict(img_array, verbose=0)
        pred_probs = predictions[0]
        pred_index = int(np.argmax(pred_probs))
        pred_class = CLASS_NAMES[pred_index]
        confidence = float(pred_probs[pred_index]) * 100
        
        all_confidences = {
            CLASS_NAMES[i]: float(pred_probs[i]) * 100 
            for i in range(len(CLASS_NAMES))
        }
        
        processing_time = round(time.time() - start_time, 2)
        
        # Generate Grad-CAM if requested
        heatmap_base64 = None
        if data.get('include_gradcam') and pred_class in ['Tuberculosis', 'Pneumonia']:
            try:
                heatmap = generate_gradcam(model, img_array, pred_index)
                import cv2
                from PIL import Image
                import io
                
                # Decode the original frame to overlay
                b64_data = data['frame']
                if ',' in b64_data:
                    b64_data = b64_data.split(',')[1]
                img_bytes = base64.b64decode(b64_data)
                orig_img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
                orig_array = np.array(orig_img.resize((224, 224)))
                orig_bgr = cv2.cvtColor(orig_array, cv2.COLOR_RGB2BGR)
                
                heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                overlay = cv2.addWeighted(orig_bgr, 0.6, heatmap_colored, 0.4, 0)
                
                _, buffer = cv2.imencode('.jpg', overlay)
                heatmap_base64 = base64.b64encode(buffer).decode('utf-8')
            except Exception as e:
                print(f"[WARN] Webcam Grad-CAM failed: {e}")
        
        return jsonify({
            'prediction': pred_class,
            'confidence': round(confidence, 2),
            'all_confidences': all_confidences,
            'severity': get_severity(pred_class, confidence),
            'processing_time': processing_time,
            'heatmap_base64': heatmap_base64
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/stats')
def api_stats():
    """Get dashboard statistics."""
    total = get_total_scans()
    most_frequent = get_most_frequent_prediction()
    recent = get_recent_scans(5)
    distribution = get_prediction_distribution()
    metrics = get_model_metrics('multiclass_model_100_epochs.keras')
    
    return jsonify({
        'total_scans': total,
        'most_frequent': most_frequent,
        'recent_scans': recent,
        'distribution': distribution,
        'model_metrics': {
            'accuracy': metrics['accuracy'] if metrics else 87.60,
            'f1_score': metrics['f1_score'] if metrics else 87.19,
            'tb_precision': metrics['tb_precision'] if metrics else 97.75,
            'tb_recall': metrics['tb_recall'] if metrics else 69.23
        }
    })


@app.route('/api/scans')
def api_scans():
    """Get paginated scan history."""
    limit = request.args.get('limit', 20, type=int)
    scans = get_recent_scans(limit)
    return jsonify({'scans': scans})


@app.route('/api/scans/<int:scan_id>')
def api_scan_detail(scan_id):
    """Get a single scan by ID."""
    scan = get_scan(scan_id)
    if scan:
        if scan.get('all_confidences'):
            scan['all_confidences'] = json.loads(scan['all_confidences'])
        return jsonify(scan)
    return jsonify({'error': 'Scan not found'}), 404


@app.route('/api/latest-prediction')
def api_latest_prediction():
    """Get the most recent prediction for the assistant context."""
    latest = get_latest_scan()
    if latest:
        latest['severity'] = get_severity(latest['prediction'], latest['confidence'])
        return jsonify(latest)
    return jsonify({'message': 'No scans yet'}), 404


@app.route('/api/chat', methods=['POST'])
def api_chat():
    """Rule-based AI assistant chat."""
    data = request.get_json()
    if not data or 'question' not in data:
        return jsonify({'error': 'No question provided'}), 400
    
    question = data['question']
    scan_id = data.get('scan_id')
    
    # Get prediction context
    context = None
    if scan_id:
        scan = get_scan(scan_id)
        if scan:
            if scan.get('all_confidences'):
                scan['all_confidences'] = json.loads(scan['all_confidences'])
            context = {
                'prediction': scan['prediction'],
                'confidence': scan['confidence'],
                'all_confidences': scan.get('all_confidences', {}),
                'heatmap_path': scan.get('heatmap_path'),
                'image_name': scan['image_name'],
                'scan_id': scan['id']
            }
    else:
        # Use latest scan
        latest = get_latest_scan()
        if latest:
            context = {
                'prediction': latest['prediction'],
                'confidence': latest['confidence'],
                'all_confidences': latest.get('all_confidences', {}),
                'heatmap_path': latest.get('heatmap_path'),
                'image_name': latest['image_name'],
                'scan_id': latest['id']
            }
    
    response = generate_response(question, context)
    
    # Save chat
    insert_chat(
        scan_id=context['scan_id'] if context else None,
        question=question,
        response=response
    )
    
    return jsonify({
        'response': response,
        'context': {
            'prediction': context['prediction'] if context else None,
            'confidence': context['confidence'] if context else None
        } if context else None
    })


@app.route('/api/chats')
def api_chats():
    """Get chat history."""
    scan_id = request.args.get('scan_id', type=int)
    limit = request.args.get('limit', 20, type=int)
    chats = get_chats(scan_id, limit)
    return jsonify({'chats': chats})


@app.route('/api/notes', methods=['POST'])
def api_notes_create():
    """Save a clinical note."""
    data = request.get_json()
    if not data or 'scan_id' not in data or 'note_text' not in data:
        return jsonify({'error': 'scan_id and note_text are required'}), 400
    
    note_id = insert_note(data['scan_id'], data['note_text'])
    return jsonify({'note_id': note_id, 'message': 'Note saved'})


@app.route('/api/notes/<int:scan_id>')
def api_notes_get(scan_id):
    """Get notes for a scan."""
    notes = get_notes(scan_id)
    return jsonify({'notes': notes})


@app.route('/api/analytics')
def api_analytics():
    """Get all analytics data for charts."""
    distribution = get_prediction_distribution()
    confidence_dist = get_confidence_distribution()
    all_metrics = get_model_metrics()
    
    return jsonify({
        'prediction_distribution': distribution,
        'confidence_distribution': confidence_dist,
        'model_metrics': all_metrics
    })


@app.route('/api/model-metrics')
def api_model_metrics_route():
    """Get model metrics for the 100-epoch model."""
    metrics = get_model_metrics('multiclass_model_100_epochs.keras')
    return jsonify({'metrics': [metrics] if metrics else []})


@app.route('/api/all-model-metrics')
def api_all_model_metrics():
    """Get metrics for ALL models (10-epoch, 100-epoch, DenseNet121)."""
    all_metrics = get_model_metrics()  # None returns all
    return jsonify({'models': all_metrics if all_metrics else []})


# ════════════════════════════════════════
#  RUN SERVER
# ════════════════════════════════════════

if __name__ == '__main__':
    print("\n" + "="*60)
    print("  TB-CAD Clinical Suite")
    print("  AI-Based Tuberculosis Detection System")
    print("  EfficientNetB0 + Grad-CAM (100-epoch model)")
    print("="*60)
    print(f"  Model: {'[OK]' if model else '[X]'} multiclass_model_100_epochs.keras")
    print(f"  Database: tb_cad.db")
    print(f"  Starting server on http://127.0.0.1:5000")
    print("="*60 + "\n")
    
    app.run(debug=True, host='127.0.0.1', port=5000, use_reloader=False)
