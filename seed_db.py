import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Use the correct credential file found in the directory
FIREBASE_CRED_PATH = "pulmoscan-a2b88-firebase-adminsdk-fbsvc-ed30f0c618.json"

def seed_demo_user():
    try:
        # Initialize Firebase
        cred = credentials.Certificate(FIREBASE_CRED_PATH)
        firebase_admin.initialize_app(cred)
        db = firestore.client()
        logger.info("Firebase initialized.")

        email = 'demo@pulmoscan.ai'
        
        # Check if user exists
        users_ref = db.collection("users")
        query = users_ref.where("email", "==", email).get()

        if query:
            logger.info(f"User {email} already exists. Skipping creation.")
            return

        # Create demo user
        user_data = {
            'email': email,
            'password': 'demo123',
            'name': 'Demo User',
            'created_at': datetime.now(),
            'role': 'medical_professional'
        }

        # Add to 'users' collection
        # We let Firestore auto-generate the ID, or we could specify one. 
        # app.py uses doc_ref.id so auto-generated id is fine.
        db.collection("users").add(user_data)
        logger.info(f"Successfully created user: {email}")

    except Exception as e:
        logger.error(f"Failed to seed database: {e}")

if __name__ == "__main__":
    seed_demo_user()
