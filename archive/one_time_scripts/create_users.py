"""
Script to create beta users for the Sports Prediction API
Run this to create your 5 friends/family accounts
"""
from sqlalchemy.orm import Session
from api.database import SessionLocal, init_db
from api.auth import create_user

def setup_beta_users():
    """Create 5 beta test users"""
    # Initialize database
    init_db()

    # Create database session
    db = SessionLocal()

    beta_users = [
        {
            "username": "user1",
            "email": "user1@example.com",
            "password": "password123",
            "full_name": "Beta User 1",
            "is_admin": False
        },
        {
            "username": "user2",
            "email": "user2@example.com",
            "password": "password123",
            "full_name": "Beta User 2",
            "is_admin": False
        },
        {
            "username": "user3",
            "email": "user3@example.com",
            "password": "password123",
            "full_name": "Beta User 3",
            "is_admin": False
        },
        {
            "username": "user4",
            "email": "user4@example.com",
            "password": "password123",
            "full_name": "Beta User 4",
            "is_admin": False
        },
        {
            "username": "user5",
            "email": "user5@example.com",
            "password": "password123",
            "full_name": "Beta User 5",
            "is_admin": False
        },
        {
            "username": "admin",
            "email": "admin@example.com",
            "password": "admin123",
            "full_name": "Admin User",
            "is_admin": True
        }
    ]

    print("Creating beta users...")
    print("=" * 60)

    for user_data in beta_users:
        try:
            user = create_user(
                db=db,
                email=user_data["email"],
                username=user_data["username"],
                password=user_data["password"],
                full_name=user_data["full_name"],
                is_admin=user_data.get("is_admin", False)
            )
            print(f"[OK] Created user: {user.username} ({user.email})")
            print(f"  Password: {user_data['password']}")
            print()
        except ValueError as e:
            print(f"[FAIL] Failed to create {user_data['username']}: {e}")
            print()

    print("=" * 60)
    print("Beta users created successfully!")
    print()
    print("You can now customize these users by editing this script:")
    print("  - Change usernames to real names")
    print("  - Change emails to their real emails")
    print("  - Change passwords to secure ones")
    print()
    print("To test the API:")
    print("  1. Start the API: py -m uvicorn api.main:app --reload")
    print("  2. Visit: http://localhost:8000/docs")
    print("  3. Login with any user above")

    db.close()


if __name__ == "__main__":
    setup_beta_users()
