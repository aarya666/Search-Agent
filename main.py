# main.py
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy import Column, Integer, String, Boolean, DateTime, create_engine
from sqlalchemy.orm import sessionmaker, declarative_base, Session
from passlib.context import CryptContext
from datetime import datetime, timedelta
from jose import jwt, JWTError
from pydantic import BaseModel, EmailStr
import os
from fastapi import UploadFile, File, Form
import shutil
import tempfile
import asyncio

from multi_engine_search import (
    run_job_search_async,
    search_with_filters_only_async,
)
from fastapi.middleware.cors import CORSMiddleware
# =========================
# CONFIG
# =========================

SECRET_KEY = "CHANGE_THIS_SECRET_KEY"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24

DATABASE_URL = "sqlite:///./app.db"

engine = create_engine(
    DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(bind=engine, autoflush=False)
Base = declarative_base()

# =========================
# MODELS
# =========================

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    is_admin = Column(Boolean, default=False)
    last_login_at = Column(DateTime, nullable=True)
    last_active_at = Column(DateTime, nullable=True)

Base.metadata.create_all(bind=engine)

# =========================
# SCHEMAS
# =========================

class SignupPayload(BaseModel):
    email: EmailStr
    password: str

class LoginPayload(BaseModel):
    email: EmailStr
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str
    user: dict

# =========================
# SECURITY
# =========================

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def hash_password(password: str) -> str:
    return pwd_context.hash(password[:72])

def verify_password(password: str, hashed: str) -> bool:
    return pwd_context.verify(password[:72], hashed)

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Not authenticated",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user = db.query(User).filter(User.id == int(user_id)).first()
    if not user:
        raise credentials_exception

    user.last_active_at = datetime.utcnow()
    db.commit()

    return user

def require_admin(user: User = Depends(get_current_user)):
    if not user.is_admin:
        raise HTTPException(status_code=403, detail="Admin only")
    return user

# =========================
# FASTAPI APP
# =========================

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for dev only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# =========================
# AUTH ROUTES
# =========================

@app.post("/auth/signup", response_model=TokenResponse)
def auth_signup(payload: SignupPayload, db: Session = Depends(get_db)):
    existing = db.query(User).filter(User.email == payload.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered.")

    now = datetime.utcnow()

    # ✅ Only first user becomes admin
    is_first_user = db.query(User).count() == 0

    user = User(
        email=payload.email,
        hashed_password=hash_password(payload.password),
        is_admin=is_first_user,
        last_login_at=now,
        last_active_at=now,
    )

    db.add(user)
    db.commit()
    db.refresh(user)

    token = create_access_token({"sub": str(user.id)})

    return {
        "access_token": token,
        "token_type": "bearer",
        "user": {
            "id": user.id,
            "email": user.email,
            "is_admin": user.is_admin,
        },
    }


@app.post("/auth/login", response_model=TokenResponse)
def auth_login(payload: LoginPayload, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == payload.email).first()

    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    if not verify_password(payload.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    now = datetime.utcnow()
    user.last_login_at = now
    user.last_active_at = now
    db.commit()

    token = create_access_token({"sub": str(user.id)})

    return {
        "access_token": token,
        "token_type": "bearer",
        "user": {
            "id": user.id,
            "email": user.email,
            "is_admin": user.is_admin,
        },
    }


@app.get("/auth/me")
def auth_me(current_user: User = Depends(get_current_user)):
    return {
        "id": current_user.id,
        "email": current_user.email,
        "is_admin": current_user.is_admin,
    }


@app.get("/admin/stats")
def admin_stats(
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    total = db.query(User).count()
    active = (
        db.query(User)
        .filter(User.last_active_at >= datetime.utcnow() - timedelta(minutes=10))
        .count()
    )

    return {
        "total_users": total,
        "active_users_last_10_min": active,
    }

# =========================
# FILE UPLOAD ROUTE
# =========================

@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    tmpdir = tempfile.gettempdir()
    filepath = os.path.join(tmpdir, file.filename)

    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {"filename": file.filename}


# =========================
# RESUME-BASED SEARCH
# =========================

@app.post("/start_search")
async def start_search(
    filename: str = Form(...),
    max_results: int = Form(10),
    remote_modes: str = Form(""),
    cities: str = Form(""),
    roles: str = Form(""),
    job_type: str = Form(""),
):
    try:
        result = await run_job_search_async(
            filename=filename,
            max_results=max_results,
            remote_modes=remote_modes,
            cities=cities,
            roles=roles,
            job_type=job_type,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =========================
# FILTER-ONLY SEARCH
# =========================

@app.post("/search_filters_only")
async def search_filters_only(
    max_results: int = Form(10),
    remote_modes: str = Form(""),
    cities: str = Form(""),
    roles: str = Form(""),
    job_type: str = Form(""),
):
    try:
        result = await search_with_filters_only_async(
            max_results=max_results,
            remote_modes=remote_modes,
            cities=cities,
            roles=roles,
            job_type=job_type,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))