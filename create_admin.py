"""
create_admin.py
Command-line script to create an admin account in customer_system.db.

Usage:
    python create_admin.py

The script will prompt for a username and password interactively.
The password is hidden while you type (uses getpass).
Passwords are hashed with bcrypt — the same method used for normal users.

This is the ONLY way to create an admin account. The web registration
form always creates normal users; there is no admin option there.
"""

import getpass
from db import init_db, register_admin


def _prompt_password(prompt: str) -> str:
    """
    Try getpass first (hides input). If the terminal doesn't support it
    (e.g. Windows IDE terminals raise KeyboardInterrupt), fall back to
    plain input() with a visible warning.
    """
    try:
        return getpass.getpass(prompt)
    except (KeyboardInterrupt, Exception):
        print()  # newline after the interrupted prompt
        print("  [Note: password will be visible — terminal does not support hidden input]")
        return input(prompt)


def main():
    print("=" * 50)
    print("  Persona — Create Admin Account")
    print("=" * 50)
    print()

    # Ensure the DB and tables exist before we try to insert.
    init_db()

    username = input("Enter admin username: ").strip()
    if not username:
        print("ERROR: Username cannot be empty.")
        return

    password = _prompt_password("Enter admin password: ")
    if not password:
        print("ERROR: Password cannot be empty.")
        return

    confirm = _prompt_password("Confirm admin password: ")
    if password != confirm:
        print("ERROR: Passwords do not match.")
        return

    ok, message = register_admin(username, password)
    if ok:
        print()
        print(f"SUCCESS: {message}")
        print(f"You can now log in at http://localhost:5000 with username '{username}'.")
    else:
        print()
        print(f"ERROR: {message}")


if __name__ == "__main__":
    main()
