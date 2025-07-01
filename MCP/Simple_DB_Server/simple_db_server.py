# simple_db_server.py
import asyncio
import sqlite3
import json
from typing import List, Dict, Any
from mcp.server.fastmcp import FastMCP

# Create the MCP server instance
mcp = FastMCP("simple-db-server")

# Database path
DB_PATH = "users.db"

def init_database():
    """Initialize the database with sample data"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Insert sample data
    sample_users = [
        ("Alice Johnson", "alice@example.com"),
        ("Bob Smith", "bob@example.com"),
        ("Carol Davis", "carol@example.com")
    ]
    
    cursor.executemany(
        "INSERT OR IGNORE INTO users (name, email) VALUES (?, ?)",
        sample_users
    )
    
    conn.commit()
    conn.close()

# Initialize database when server starts
init_database()

@mcp.resource("sqlite://users")
def get_users_resource() -> str:
    """Resource to access all user records in the database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("SELECT id, name, email, created_at FROM users")
    users = cursor.fetchall()
    
    conn.close()
    
    # Convert to list of dictionaries
    user_list = [
        {
            "id": user[0],
            "name": user[1],
            "email": user[2],
            "created_at": user[3]
        }
        for user in users
    ]
    
    return json.dumps(user_list, indent=2)

@mcp.tool()
def query_users(filter: str = "", limit: int = 10) -> str:
    """Query users from the database with optional filtering
    
    Args:
        filter: Optional search term to filter users by name or email
        limit: Maximum number of results to return (default: 10)
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    if filter:
        cursor.execute(
            "SELECT id, name, email, created_at FROM users WHERE name LIKE ? OR email LIKE ? LIMIT ?",
            (f"%{filter}%", f"%{filter}%", limit)
        )
    else:
        cursor.execute("SELECT id, name, email, created_at FROM users LIMIT ?", (limit,))
    
    users = cursor.fetchall()
    conn.close()
    
    user_list = [
        {
            "id": user[0],
            "name": user[1],
            "email": user[2],
            "created_at": user[3]
        }
        for user in users
    ]
    
    return json.dumps(user_list, indent=2)

@mcp.tool()
def add_user(name: str, email: str) -> str:
    """Add a new user to the database
    
    Args:
        name: User's full name
        email: User's email address
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute(
            "INSERT INTO users (name, email) VALUES (?, ?)",
            (name, email)
        )
        
        user_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return f"Successfully added user '{name}' with ID {user_id}"
        
    except sqlite3.IntegrityError as e:
        return f"Error: {str(e)} (Email address may already exist)"

if __name__ == "__main__":
    # Run the server
    mcp.run()