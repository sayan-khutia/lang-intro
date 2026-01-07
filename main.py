"""
FastAPI server for Fitness Coach chatbot.
Provides user authentication and chat endpoints with streaming support.
"""

from dotenv import load_dotenv
load_dotenv()

import os
import sqlite3
from datetime import datetime, timedelta
from typing import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
import hashlib
import secrets
import jwt

from fitness_coach import stream_chat_with_coach, get_user_chat_history

# Configuration
DATABASE_PATH = os.getenv("DATABASE_PATH", "fitness_coach.db")
JWT_SECRET = os.getenv("JWT_SECRET", secrets.token_hex(32))
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24

security = HTTPBearer()


# ============== Database Setup ==============

def get_db_connection():
    """Create a database connection."""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_database():
    """Initialize the SQLite database with required tables."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Users table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Chat history table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    """)
    
    # Create index for faster queries
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_chat_history_user_id 
        ON chat_history (user_id)
    """)
    
    conn.commit()
    conn.close()


# ============== Pydantic Models ==============

class UserRegister(BaseModel):
    email: EmailStr
    password: str


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: int
    email: str


class ChatMessage(BaseModel):
    message: str


class ChatHistoryItem(BaseModel):
    id: int
    role: str
    content: str
    created_at: str


class ChatHistoryResponse(BaseModel):
    history: list[ChatHistoryItem]
    total: int


# ============== Authentication Helpers ==============

def hash_password(password: str) -> str:
    """Hash password using SHA-256 with salt."""
    salt = secrets.token_hex(16)
    password_hash = hashlib.sha256((password + salt).encode()).hexdigest()
    return f"{salt}:{password_hash}"


def verify_password(password: str, stored_hash: str) -> bool:
    """Verify password against stored hash."""
    try:
        salt, password_hash = stored_hash.split(":")
        return hashlib.sha256((password + salt).encode()).hexdigest() == password_hash
    except ValueError:
        return False


def create_jwt_token(user_id: int, email: str) -> str:
    """Create a JWT token for the user."""
    expiration = datetime.utcnow() + timedelta(hours=JWT_EXPIRATION_HOURS)
    payload = {
        "user_id": user_id,
        "email": email,
        "exp": expiration,
        "iat": datetime.utcnow()
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_jwt_token(token: str) -> dict:
    """Decode and verify JWT token."""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired"
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    """Dependency to get the current authenticated user."""
    token = credentials.credentials
    payload = decode_jwt_token(token)
    return {"user_id": payload["user_id"], "email": payload["email"]}


# ============== Database Operations ==============

def create_user(email: str, password: str) -> dict:
    """Create a new user in the database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        password_hash = hash_password(password)
        cursor.execute(
            "INSERT INTO users (email, password_hash) VALUES (?, ?)",
            (email, password_hash)
        )
        conn.commit()
        user_id = cursor.lastrowid
        return {"id": user_id, "email": email}
    except sqlite3.IntegrityError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    finally:
        conn.close()


def get_user_by_email(email: str) -> dict | None:
    """Get user by email."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE email = ?", (email,))
    row = cursor.fetchone()
    conn.close()
    
    if row:
        return dict(row)
    return None


def save_chat_message(user_id: int, role: str, content: str):
    """Save a chat message to the database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO chat_history (user_id, role, content) VALUES (?, ?, ?)",
        (user_id, role, content)
    )
    conn.commit()
    conn.close()


def get_chat_history(user_id: int, limit: int = 50, offset: int = 0) -> tuple[list[dict], int]:
    """Get chat history for a user."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Get total count
    cursor.execute(
        "SELECT COUNT(*) FROM chat_history WHERE user_id = ?",
        (user_id,)
    )
    total = cursor.fetchone()[0]
    
    # Get paginated history
    cursor.execute("""
        SELECT id, role, content, created_at 
        FROM chat_history 
        WHERE user_id = ? 
        ORDER BY created_at ASC 
        LIMIT ? OFFSET ?
    """, (user_id, limit, offset))
    
    rows = cursor.fetchall()
    conn.close()
    
    history = [dict(row) for row in rows]
    return history, total


# ============== FastAPI App ==============

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database on startup."""
    init_database()
    yield


app = FastAPI(
    title="Fitness Coach API",
    description="AI-powered fitness coaching chatbot with user authentication",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============== Auth Endpoints ==============

@app.post("/auth/register", response_model=TokenResponse, tags=["Authentication"])
async def register(user_data: UserRegister):
    """Register a new user."""
    if len(user_data.password) < 6:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password must be at least 6 characters"
        )
    
    user = create_user(user_data.email, user_data.password)
    token = create_jwt_token(user["id"], user["email"])
    
    return TokenResponse(
        access_token=token,
        user_id=user["id"],
        email=user["email"]
    )


@app.post("/auth/login", response_model=TokenResponse, tags=["Authentication"])
async def login(user_data: UserLogin):
    """Login with email and password."""
    user = get_user_by_email(user_data.email)
    
    if not user or not verify_password(user_data.password, user["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )
    
    token = create_jwt_token(user["id"], user["email"])
    
    return TokenResponse(
        access_token=token,
        user_id=user["id"],
        email=user["email"]
    )


# ============== Chat Endpoints ==============

@app.post("/chat/message", tags=["Chat"])
async def send_message(
    chat_message: ChatMessage,
    current_user: dict = Depends(get_current_user)
):
    """
    Send a message to the fitness coach and receive a streaming response.
    
    The response is streamed as Server-Sent Events (SSE).
    """
    user_id = current_user["user_id"]
    user_email = current_user["email"]
    message = chat_message.message.strip()
    
    if not message:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Message cannot be empty"
        )
    
    # Save user message to database
    save_chat_message(user_id, "user", message)
    
    async def generate_response() -> AsyncGenerator[str, None]:
        """Generate streaming response from fitness coach."""
        full_response = []
        
        try:
            async for chunk in stream_chat_with_coach(str(user_id), message):
                full_response.append(chunk)
                # Stream as Server-Sent Events format
                yield f"data: {chunk}\n\n"
            
            # Save complete AI response to database
            complete_response = "".join(full_response)
            save_chat_message(user_id, "assistant", complete_response)
            
            # Send end signal
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            save_chat_message(user_id, "assistant", error_msg)
            yield f"data: {error_msg}\n\n"
            yield "data: [DONE]\n\n"
    
    return StreamingResponse(
        generate_response(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@app.get("/chat/history", response_model=ChatHistoryResponse, tags=["Chat"])
async def fetch_chat_history(
    limit: int = 50,
    offset: int = 0,
    current_user: dict = Depends(get_current_user)
):
    """
    Fetch chat history for the authenticated user.
    
    - **limit**: Maximum number of messages to return (default: 50)
    - **offset**: Number of messages to skip (for pagination)
    """
    user_id = current_user["user_id"]
    
    if limit < 1 or limit > 100:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Limit must be between 1 and 100"
        )
    
    if offset < 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Offset cannot be negative"
        )
    
    history, total = get_chat_history(user_id, limit, offset)
    
    return ChatHistoryResponse(
        history=[ChatHistoryItem(**item) for item in history],
        total=total
    )


# ============== Health Check ==============

@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
