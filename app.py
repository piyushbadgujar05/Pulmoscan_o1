import os
import io
import json
import base64
from datetime import datetime
from flask import Flask, request, jsonify, render_template, send_file, redirect, url_for, session
from flask_cors import CORS
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import logging
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as ReportLabImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.piecharts import Pie
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics import renderPDF

# Load environment variables from .env file (for local development)
from dotenv import load_dotenv
load_dotenv()

# Firebase Admin SDK
import firebase_admin
from firebase_admin import credentials, firestore, storage, auth as firebase_auth

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Flask app initialization
from flask import Flask, send_from_directory

app = Flask(
    __name__,
    template_folder="frontend",  # HTML templates are in frontend/
    static_folder="frontend",     # Static files also in frontend/
    static_url_path=""
)


# Security: Use environment variable for secret key
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')

# CORS configuration
CORS(app, resources={
    r"/api/*": {
        "origins": "*",  # In production, restrict to your domain
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# Firebase initialization with environment variable
FIREBASE_INITIALIZED = False

try:
    # Try to load Firebase credentials from environment variable first
    firebase_json_str = os.environ.get("FIREBASE_ADMIN_JSON")
    
    if firebase_json_str:
        # Production: Load from environment variable
        firebase_json = json.loads(firebase_json_str)
        cred = credentials.Certificate(firebase_json)
        logger.info("✅ Using Firebase credentials from environment variable")
    else:
        # Development: Load from file
        firebase_cred_path = "pulmoscan-a2b88-firebase-adminsdk-fbsvc-ed30f0c618.json"
        if os.path.exists(firebase_cred_path):
            cred = credentials.Certificate(firebase_cred_path)
            logger.info("✅ Using Firebase credentials from local file")
        else:
            raise FileNotFoundError("Firebase credentials not found")
    
    firebase_admin.initialize_app(cred, {
        "storageBucket": "pulmoscan-a2b88.appspot.com"
    })
    firestore_db = firestore.client()
    storage_bucket = storage.bucket()
    FIREBASE_INITIALIZED = True
    logger.info("✅ Firebase initialized successfully")
    
except Exception as e:
    logger.error(f"❌ Firebase initialization failed: {e}")
    FIREBASE_INITIALIZED = False


def login_required(f):
    def decorated_function(*args, **kwargs):
        if 'user' not in session:
            return redirect('/login')
        return f(*args, **kwargs)
    decorated_function.__name__ = f.__name__
    return decorated_function


class LungCancerModel:
    def __init__(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        self.model = self.load_model(model_path)
        self.class_names = ['normal', 'benign', 'malignant']
        self.transform = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def load_model(self, model_path):
        """Load the trained model"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Create model architecture
            model = models.resnet50(pretrained=False)
            model.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(model.fc.in_features, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, 3)
            )
            
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(self.device)
            model.eval()
            logger.info("Model loaded successfully")
            return model
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def predict(self, image_bytes):
        """Predict from image bytes"""
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                prediction_idx = torch.argmax(outputs, dim=1).item()
                confidence = probabilities[0][prediction_idx].item()
            
            prediction = self.class_names[prediction_idx]
            
            # Get medical recommendation
            recommendation = self.get_medical_recommendation(prediction, confidence)
            
            return {
                'success': True,
                'prediction': prediction,
                'confidence': float(confidence),
                'probabilities': {
                    cls: float(prob) for cls, prob in zip(self.class_names, probabilities[0])
                },
                'recommendation': recommendation,
                'risk_level': self.get_risk_level(prediction),
                'timestamp': datetime.now().isoformat(),
                'image_data': base64.b64encode(image_bytes).decode('utf-8') if len(image_bytes) < 5000000 else None
            }
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_medical_recommendation(self, prediction, confidence):
        recommendations = {
            'normal': [
                f"No signs of cancer detected (Confidence: {confidence:.1%})",
                "Continue regular annual checkups",
                "Maintain healthy lifestyle habits",
                "No immediate follow-up required"
            ],
            'benign': [
                f"BENIGN CANCER DETECTED (Confidence: {confidence:.1%})",
                "URGENT: Consult with an oncologist immediately",
                "Schedule biopsy for confirmation",
                "Begin treatment planning immediately"
            ],
            'malignant': [
                f"MALIGNANT CANCER DETECTED (Confidence: {confidence:.1%})",
                "URGENT: Consult with an oncologist immediately",
                "Schedule biopsy for confirmation",
                "Begin treatment planning immediately"
            ]
        }
        return recommendations.get(prediction, ["Please consult with a healthcare professional."])
    
    def get_risk_level(self, prediction):
        risk_levels = {
            'normal': 'low',
            'benign': 'high',  # Changed from 'medium' to 'high'
            'malignant': 'high'
        }
        return risk_levels.get(prediction, 'unknown')

# --------------------------------------------------------------------------------
# Model Download Helper Functions
# --------------------------------------------------------------------------------
def download_from_google_drive(url, destination):
    """Download file from Google Drive with virus scan bypass"""
    import requests
    
    try:
        session = requests.Session()
        
        # First request to get confirmation token
        response = session.get(url, stream=True, allow_redirects=True)
        
        # Look for virus scan warning and confirmation token
        token = None
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                token = value
                break
        
        # If token found, make confirmed download request
        if token:
            params = {'confirm': token}
            response = session.get(url, params=params, stream=True, allow_redirects=True)
        
        # Save file
        logger.info(f"Downloading model to {destination}...")
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(chunk_size=32768):
                if chunk:
                    f.write(chunk)
        
        logger.info(f"✅ Model downloaded successfully ({os.path.getsize(destination) / 1024 / 1024:.1f} MB)")
        return True
        
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return False

def ensure_model_exists(model_path, model_url):
    """Ensure model file exists, download if necessary"""
    
    # Check if model already exists
    if os.path.exists(model_path):
        file_size = os.path.getsize(model_path) / 1024 / 1024
        logger.info(f"✅ Model file found: {model_path} ({file_size:.1f} MB)")
        return True
    
    # If no URL provided, can't download
    if not model_url:
        logger.error(f"❌ Model file not found at {model_path} and MODEL_URL not provided")
        return False
    
    # Download the model
    logger.info(f"📥 Downloading model from {model_url[:50]}...")
    
    # Create directory if needed
    model_dir = os.path.dirname(model_path)
    if model_dir and not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)
    
    return download_from_google_drive(model_url, model_path)

# Initialize model
MODEL_LOADED = False

try:
    # Get model path and URL from environment
    MODEL_PATH = os.environ.get('MODEL_PATH', 'lung_cancer_model.pth')
    MODEL_URL = os.environ.get('MODEL_URL')
    
    logger.info(f"Model Path: {MODEL_PATH}")
    logger.info(f"Model URL: {'Set' if MODEL_URL else 'Not set'}")
    
    # Ensure model file exists (download if needed)
    if ensure_model_exists(MODEL_PATH, MODEL_URL):
        model = LungCancerModel(MODEL_PATH)
        MODEL_LOADED = True
        logger.info("✅ Lung Cancer Detection System Ready!")
    else:
        logger.error("❌ Failed to obtain model file")
        
except Exception as e:
    MODEL_LOADED = False
    logger.error(f"❌ Failed to load model: {e}")

# --------------------------------------------------------------------------------
# Firebase Helper Functions
# --------------------------------------------------------------------------------
def upload_to_firebase_storage(file_bytes, filename, user_id):
    """Upload image to Firebase Storage with user folder"""
    try:
        blob_path = f"users/{user_id}/ct_scans/{filename}"
        blob = storage_bucket.blob(blob_path)
        blob.upload_from_string(file_bytes, content_type="image/jpeg")
        blob.make_public()
        return blob.public_url
    except Exception as e:
        logger.error(f"Storage upload error: {e}")
        return None

def save_to_firestore(analysis_data, user_id):
    """Save analysis results to Firestore"""
    try:
        doc_ref = firestore_db.collection("users").document(user_id).collection("analysis_results").document()
        analysis_data["id"] = doc_ref.id
        analysis_data["user_id"] = user_id
        analysis_data["created_at"] = datetime.now()
        
        doc_ref.set(analysis_data)
        return doc_ref.id
    except Exception as e:
        logger.error(f"Firestore save error: {e}")
        return None

# --------------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------------

@app.route('/')
def home():
    """Redirect to login or main app based on authentication"""
    if 'user' in session:
        return render_template('index.html')
    return redirect('/login')

@app.route('/login')
def login_page():
    """Login page"""
    if 'user' in session:
        return redirect('/')
    return render_template('login.html')

@app.route('/signup')
def signup_page():
    """Signup page"""
    if 'user' in session:
        return redirect('/')
    return render_template('signup.html')

@app.route('/api/auth/login', methods=['POST'])
def api_login():
    """API endpoint for login"""
    try:
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')
        
        # For demo purposes - in production, use Firebase Auth SDK on frontend
        # This is a simplified version that checks against Firestore
        users_ref = firestore_db.collection("users")
        query = users_ref.where("email", "==", email).limit(1).get()
        
        if not query:
            logger.warning(f"Login failed: User {email} not found")
            return jsonify({
                'success': False,
                'error': 'Invalid credentials'
            }), 401
        
        user_data = query[0].to_dict()
        user_id = query[0].id
        
        logger.info(f"User found: {email}, ID: {user_id}")
        # DEBUG: Print password lengths (don't print actual passwords in prod logs usually, but for local debug it's okay-ish, or just length)
        logger.info(f"Stored pwd: {user_data.get('password')}, Provided pwd: {password}")

        if user_data.get('password') != password:  
            logger.warning("Login failed: Password mismatch")
            return jsonify({
                'success': False,
                'error': 'Invalid credentials'
            }), 401
        
        session['user'] = {
            'uid': user_id,
            'email': user_data['email'],
            'name': user_data.get('name', user_data['email'].split('@')[0])
        }
        
        return jsonify({
            'success': True,
            'user': session['user']
        })
        
    except Exception as e:
        logger.error(f"Login error: {e}")
        return jsonify({
            'success': False,
            'error': 'Invalid credentials'
        }), 401

@app.route('/api/auth/signup', methods=['POST'])
def api_signup():
    """API endpoint for signup"""
    try:
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')
        name = data.get('name')
        
        # Check if user already exists
        users_ref = firestore_db.collection("users")
        query = users_ref.where("email", "==", email).limit(1).get()
        
        if query:
            return jsonify({
                'success': False,
                'error': 'User already exists with this email'
            }), 400
        
        # Store user data in Firestore
        user_data = {
            'email': email,
            'password': password, 
            'name': name,
            'created_at': datetime.now(),
            'role': 'medical_professional'
        }
        
        doc_ref = firestore_db.collection("users").document()
        doc_ref.set(user_data)
        
        # Login user
        session['user'] = {
            'uid': doc_ref.id,
            'email': email,
            'name': name
        }
        
        return jsonify({
            'success': True,
            'user': session['user']
        })
        
    except Exception as e:
        logger.error(f"Signup error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/api/auth/logout', methods=['POST'])
def api_logout():
    """Logout user"""
    session.pop('user', None)
    return jsonify({'success': True})

@app.route('/api/auth/user')
def get_user():
    """Get current user info"""
    user = session.get('user')
    if user:
        return jsonify({'success': True, 'user': user})
    return jsonify({'success': False, 'user': None})



@app.route('/dashboard')
@login_required
def dashboard():
    """Main application dashboard"""
    return render_template('index.html')

@app.route('/api/health')
@login_required
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': MODEL_LOADED,
        'device': str(model.device) if MODEL_LOADED else 'none',
        'user': session.get('user')
    })

@app.route('/api/predict', methods=['POST'])
@login_required
def predict():
    """Protected prediction endpoint"""
    if not MODEL_LOADED:
        return jsonify({
            'success': False,
            'error': 'Model not loaded. Please check server configuration.'
        }), 500
    
    if 'file' not in request.files:
        return jsonify({
            'success': False,
            'error': 'No file uploaded'
        }), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({
            'success': False,
            'error': 'No file selected'
        }), 400
    
 
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        return jsonify({
            'success': False,
            'error': 'Invalid file type. Please upload an image (PNG, JPG, JPEG, BMP, TIFF)'
        }), 400
    
    try:
        user_id = session['user']['uid']
        image_bytes = file.read()
        
        # Run prediction
        result = model.predict(image_bytes)
        if not result['success']:
            return jsonify(result), 500
        
        # Store in Firebase
        filename = f"{datetime.now().timestamp()}_{file.filename}"
        image_url = upload_to_firebase_storage(image_bytes, filename, user_id)
        
        if image_url:
            result['image_url'] = image_url
            result['user_id'] = user_id
            
            # Save to Firestore
            firestore_id = save_to_firestore(result, user_id)
            result['firestore_id'] = firestore_id
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Prediction endpoint error: {e}")
        return jsonify({
            'success': False,
            'error': f'Prediction failed: {str(e)}'
        }), 500

@app.route('/api/generate-report', methods=['POST'])
@login_required
def generate_report():
    """Generate and download PDF report"""
    try:
        data = request.get_json()
        analysis_data = data.get('analysis_data')
        patient_info = data.get('patient_info', {})
        
        if not analysis_data:
            return jsonify({'success': False, 'error': 'No analysis data provided'}), 400
        
        # Generate PDF
        pdf_buffer = generate_pdf_report(analysis_data, patient_info)
        
        # Create filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"lung_cancer_report_{timestamp}.pdf"
        
        return send_file(
            pdf_buffer,
            as_attachment=True,
            download_name=filename,
            mimetype='application/pdf'
        )
        
    except Exception as e:
        logger.error(f"Report generation error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/user/history')
@login_required
def get_user_history():
    """Get user's analysis history"""
    try:
        user_id = session['user']['uid']
        docs = firestore_db.collection("users").document(user_id).collection("analysis_results").order_by("created_at", direction=firestore.Query.DESCENDING).limit(20).stream()
        
        history = []
        for doc in docs:
            data = doc.to_dict()
            data['id'] = doc.id
            # Convert timestamp for JSON serialization
            if 'created_at' in data:
                data['created_at'] = data['created_at'].isoformat()
            history.append(data)
        
        return jsonify({'success': True, 'history': history})
        
    except Exception as e:
        logger.error(f"History fetch error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


def generate_pdf_report(analysis_data, patient_info=None):
    """Generate PDF medical report"""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=0.5*inch)
    
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        textColor=colors.HexColor('#4facfe'),
        alignment=1
    )
    
    story.append(Paragraph("PulmoScan AI - Medical Report", title_style))
    story.append(Spacer(1, 20))
    
    # Patient Information
    if patient_info:
        story.append(Paragraph("Patient Information", styles['Heading2']))
        patient_data = [
            ['Patient ID', patient_info.get('patient_id', 'N/A')],
            ['Age', patient_info.get('age', 'N/A')],
            ['Gender', patient_info.get('gender', 'N/A')],
            ['Referring Physician', patient_info.get('physician', 'N/A')]
        ]
        patient_table = Table(patient_data, colWidths=[2*inch, 3*inch])
        patient_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4facfe')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(patient_table)
        story.append(Spacer(1, 20))
    
    # Diagnosis Results
    story.append(Paragraph("Diagnosis Results", styles['Heading2']))
    
    prediction = analysis_data.get('prediction', 'Unknown')
    # Update prediction display for benign cases
    if prediction == 'benign':
        prediction_display = 'Benign Cancer'
    elif prediction == 'malignant':
        prediction_display = 'Malignant Cancer'
    else:
        prediction_display = prediction.title()
    
    confidence = analysis_data.get('confidence', 0) * 100

    result_data = [
        ['Prediction', prediction_display],
        ['Confidence', f"{confidence:.1f}%"],
        ['Risk Level', analysis_data.get('risk_level', 'Unknown').title()],
        ['Analysis Date', datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
    ]
    
    result_table = Table(result_data, colWidths=[2*inch, 3*inch])
    result_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#00f2fe')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(result_table)
    story.append(Spacer(1, 20))
    
    # Probability Distribution
    story.append(Paragraph("Probability Distribution", styles['Heading2']))
    probabilities = analysis_data.get('probabilities', {})
    
    prob_data = [['Class', 'Probability']]
    for cls, prob in probabilities.items():
        # Update class name display for benign
        display_cls = 'Benign Cancer' if cls == 'benign' else 'Malignant Cancer' if cls == 'malignant' else cls.title()
        prob_data.append([display_cls, f"{prob*100:.1f}%"])
    
    prob_table = Table(prob_data, colWidths=[2*inch, 3*inch])
    prob_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#764ba2')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.whitesmoke),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(prob_table)
    story.append(Spacer(1, 20))
    
    # Medical Recommendations
    story.append(Paragraph("Medical Recommendations", styles['Heading2']))
    recommendations = analysis_data.get('recommendation', [])
    
    for i, rec in enumerate(recommendations, 1):
        story.append(Paragraph(f"{i}. {rec}", styles['Normal']))
    
    story.append(Spacer(1, 20))
    
    # Disclaimer
    disclaimer_style = ParagraphStyle(
        'Disclaimer',
        parent=styles['Normal'],
        fontSize=8,
        textColor=colors.grey,
        alignment=1
    )
    story.append(Paragraph("This report is generated by AI and should be reviewed by a qualified medical professional. PulmoScan AI is not responsible for diagnostic decisions made based on this report.", disclaimer_style))
    
    doc.build(story)
    buffer.seek(0)
    return buffer

@app.route('/api/sample-analysis')
def sample_analysis():
    """Return sample data for frontend testing"""
    return jsonify({
        'success': True,
        'prediction': 'benign',
        'confidence': 0.894,
        'probabilities': {
            'normal': 0.894,
            'benign': 0.085,
            'malignant': 0.021
        },
        'recommendation': [
            "BENIGN CANCER DETECTED (Confidence: 89.4%)",
            "URGENT: Consult with an oncologist immediately",
            "Schedule biopsy for confirmation",
            "Begin treatment planning immediately"
        ],
        'risk_level': 'high',
        'timestamp': datetime.now().isoformat()
    })


if __name__ == '__main__':
    # Get configuration from environment
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') != 'production'
    
    app.run(
        debug=debug,
        host='0.0.0.0',
        port=port
    )