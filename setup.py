"""
Setup and initialization script for CFB Data Scraper
Helps users get started quickly
"""
import os
import sys
import subprocess
from cfb_nfl_database import FootballDatabase


def check_dependencies():
    """Check if required packages are installed"""
    print("Checking dependencies...")
    required_packages = ['requests', 'schedule']
    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✓ {package} is installed")
        except ImportError:
            print(f"  ✗ {package} is NOT installed")
            missing_packages.append(package)

    if missing_packages:
        print("\nMissing packages detected!")
        print("Install them with: pip install -r requirements.txt")
        return False

    print("\n✓ All dependencies are installed!\n")
    return True


def initialize_database():
    """Initialize the database"""
    print("Initializing database...")

    if os.path.exists('cfb_games.db'):
        response = input("Database already exists. Reinitialize? (y/N): ")
        if response.lower() != 'y':
            print("Skipping database initialization.")
            return True

    try:
        db = FootballDatabase()
        db.connect()
        db.initialize_schema()
        db.close()
        print("✓ Database initialized successfully!\n")
        return True
    except Exception as e:
        print(f"✗ Error initializing database: {e}\n")
        return False


def initial_data_fetch():
    """Prompt user to fetch initial data"""
    print("Would you like to fetch some initial data?")
    print("Options:")
    print("  1. Fetch current week")
    print("  2. Fetch recent games (last 7 days)")
    print("  3. Fetch full current season")
    print("  4. Fetch historical data (specify years)")
    print("  5. Skip for now")

    choice = input("\nEnter choice (1-5): ").strip()

    if choice == '1':
        print("\nFetching current week...")
        subprocess.run([sys.executable, 'scheduler.py', 'current'])
    elif choice == '2':
        print("\nFetching recent games...")
        subprocess.run([sys.executable, 'scheduler.py', 'recent'])
    elif choice == '3':
        print("\nFetching full current season (this may take a while)...")
        subprocess.run([sys.executable, 'scheduler.py', 'full'])
    elif choice == '4':
        start_year = input("Enter start year (e.g., 2020): ").strip()
        end_year = input("Enter end year (e.g., 2024): ").strip()
        print(f"\nFetching historical data from {start_year} to {end_year}...")
        print("This may take a considerable amount of time...")
        subprocess.run([sys.executable, 'scheduler.py', 'historical', start_year, end_year])
    else:
        print("\nSkipping initial data fetch.")

    print("\n✓ Setup complete!")


def show_next_steps():
    """Show user what to do next"""
    print("\n" + "="*80)
    print("SETUP COMPLETE!")
    print("="*80)
    print("\nNext steps:")
    print("\n1. Fetch data manually:")
    print("   python scheduler.py current          # Fetch current week")
    print("   python scheduler.py recent           # Fetch last 7 days")
    print("   python scheduler.py full             # Fetch full current season")
    print("   python scheduler.py historical 2020 2024  # Fetch historical data")

    print("\n2. Run automated scheduled updates:")
    print("   python scheduler.py schedule")

    print("\n3. Query the database:")
    print("   python query_examples.py             # See example queries")

    print("\n4. View database:")
    print("   - Use any SQLite browser (DB Browser for SQLite, etc.)")
    print("   - Database file: cfb_games.db")

    print("\n5. Read the documentation:")
    print("   - Check README.md for detailed usage instructions")

    print("\n" + "="*80 + "\n")


def main():
    """Main setup process"""
    print("\n" + "="*80)
    print("NCAA D1 FBS College Football Data Scraper - Setup")
    print("="*80 + "\n")

    # Check dependencies
    if not check_dependencies():
        print("Please install required packages and run setup again.")
        return

    # Initialize database
    if not initialize_database():
        print("Database initialization failed. Please check the error and try again.")
        return

    # Prompt for initial data fetch
    fetch_now = input("Would you like to fetch some data now? (y/N): ").strip().lower()
    if fetch_now == 'y':
        initial_data_fetch()

    # Show next steps
    show_next_steps()


if __name__ == '__main__':
    main()
