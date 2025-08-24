import os
import uuid
import asyncio
import re
from datetime import datetime
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, field_validator
import motor.motor_asyncio
from pymongo import IndexModel
import logging
import json
from dotenv import load_dotenv

# Import LLM integration
import litellm

# Load environment variables
load_dotenv()

# Initialize FastAPI
app = FastAPI(
    title="BytePrep API",
    description="AI-Powered Coding Interview Preparation Platform",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database setup
MONGO_URL = os.getenv("MONGO_URL")
DB_NAME = os.getenv("DB_NAME", "byteprep")

client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URL)
db = client[DB_NAME]

# Security
security = HTTPBearer()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic Models
class CodeEvaluationRequest(BaseModel):
    code: str = Field(..., description="Code to evaluate")
    language: str = Field(..., description="Programming language")
    problem_description: str = Field(..., description="Problem statement")
    difficulty: str = Field(default="medium", description="Problem difficulty")
    
    @field_validator('language')
    def validate_language(cls, v):
        supported = ['python', 'javascript', 'java', 'cpp', 'c', 'go']
        if v.lower() not in supported:
            raise ValueError(f'Language {v} not supported')
        return v.lower()

class FeedbackItem(BaseModel):
    category: str
    message: str
    severity: str
    line_number: Optional[int] = None
    suggestion: Optional[str] = None

class EvaluationResult(BaseModel):
    evaluation_id: str
    timestamp: datetime
    overall_score: float = Field(..., ge=0, le=100)
    starcoder_feedback: Dict[str, Any]
    gemini_feedback: Dict[str, Any]
    feedback_items: List[FeedbackItem]
    suggestions: List[str]
    execution_time: float

class Problem(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    description: str
    difficulty: str
    category: str
    examples: List[Dict[str, Any]] = []
    test_cases: List[Dict[str, Any]] = []
    created_at: datetime = Field(default_factory=datetime.now)

class UserProgress(BaseModel):
    user_id: str
    problems_solved: List[str] = []
    total_evaluations: int = 0
    average_score: float = 0.0
    strengths: List[str] = []
    weaknesses: List[str] = []
    last_activity: datetime = Field(default_factory=datetime.now)

# LLM Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    litellm.api_key = GEMINI_API_KEY

# Services
class CodeEvaluationService:
    def __init__(self):
        # System messages for different AI roles
        self.starcoder_system_message = """You are a code analysis specialist focusing on performance optimization, code efficiency, and technical implementation details. 
Your response must be a JSON object with the following keys: "analysis", "efficiency_score", "time_complexity", "space_complexity", "optimizations", "quality_score".
- "analysis": A detailed feedback string.
- "efficiency_score": An integer between 0 and 100.
- "time_complexity": A string (e.g., "O(n)").
- "space_complexity": A string (e.g., "O(1)").
- "optimizations": A list of strings.
- "quality_score": An integer between 0 and 100.
"""
        self.gemini_system_message = """You are an expert programming tutor specializing in code evaluation and feedback for coding interviews. 
Your response must be a JSON object with the following keys: "feedback", "correctness_score", "readability_score", "interview_score", "suggestions".
- "feedback": A detailed feedback string.
- "correctness_score": An integer between 0 and 100.
- "readability_score": An integer between 0 and 100.
- "interview_score": An integer between 0 and 100.
- "suggestions": A list of strings.
"""
    
    async def evaluate_with_starcoder(self, code: str, language: str, problem: str) -> Dict[str, Any]:
        """Evaluate code using StarCoder2-style analysis"""
        
        try:
            prompt = f"Problem: {problem}\n\nLanguage: {language}\n\nCode:\n```\n{code}\n```"
            messages = [
                {"role": "system", "content": self.starcoder_system_message},
                {"role": "user", "content": prompt}
            ]
            
            response = await litellm.acompletion(
                model="gemini/gemini-pro", 
                messages=messages,
                response_format={ "type": "json_object" }
            )
            
            response_content = response.choices[0].message.content
            return json.loads(response_content)
        except Exception as e:
            logger.error(f"StarCoder evaluation failed: {e}")
            return {
                "analysis": "Code analysis system temporarily unavailable", 
                "efficiency_score": 50, 
                "quality_score": 50,
                "time_complexity": "O(?)",
                "space_complexity": "O(?)",
                "optimizations": []
            }
    
    async def evaluate_with_gemini(self, code: str, language: str, problem: str) -> Dict[str, Any]:
        """Evaluate code using Gemini Pro"""
        
        try:
            prompt = f"Problem: {problem}\n\nLanguage: {language}\n\nCode:\n```\n{code}\n```"
            messages = [
                {"role": "system", "content": self.gemini_system_message},
                {"role": "user", "content": prompt}
            ]
            
            response = await litellm.acompletion(
                model="gemini/gemini-pro", 
                messages=messages,
                response_format={ "type": "json_object" }
            )
            
            response_content = response.choices[0].message.content
            return json.loads(response_content)
        except Exception as e:
            logger.error(f"Gemini evaluation failed: {e}")
            return {
                "feedback": "Interview evaluation system temporarily unavailable", 
                "correctness_score": 50, 
                "interview_score": 50,
                "readability_score": 50,
                "suggestions": []
            }
    
    async def evaluate_code(self, request: CodeEvaluationRequest) -> EvaluationResult:
        """Comprehensive dual-AI code evaluation"""
        start_time = datetime.now()
        
        # Parallel evaluation with both AI systems
        starcoder_task = asyncio.create_task(
            self.evaluate_with_starcoder(request.code, request.language, request.problem_description)
        )
        gemini_task = asyncio.create_task(
            self.evaluate_with_gemini(request.code, request.language, request.problem_description)
        )
        
        starcoder_result, gemini_result = await asyncio.gather(starcoder_task, gemini_task)
        
        # Combine feedback
        feedback_items = []
        suggestions = []
        
        # Add StarCoder feedback items
        for opt in starcoder_result.get("optimizations", []):
            feedback_items.append(FeedbackItem(
                category="performance",
                message=opt,
                severity="info"
            ))
            
        # Add Gemini feedback items  
        for sug in gemini_result.get("suggestions", []):
            feedback_items.append(FeedbackItem(
                category="readability", 
                message=sug,
                severity="info"
            ))
            
        suggestions.extend(starcoder_result.get("optimizations", []))
        suggestions.extend(gemini_result.get("suggestions", []))
        
        # Calculate overall score
        efficiency_score = starcoder_result.get("efficiency_score", 50)
        correctness_score = gemini_result.get("correctness_score", 50)
        overall_score = (efficiency_score * 0.4 + correctness_score * 0.6)
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        result = EvaluationResult(
            evaluation_id=str(uuid.uuid4()),
            timestamp=start_time,
            overall_score=overall_score,
            starcoder_feedback=starcoder_result,
            gemini_feedback=gemini_result,
            feedback_items=feedback_items,
            suggestions=suggestions[:10],  # Limit suggestions
            execution_time=execution_time
        )
        
        # Store evaluation in database
        await db.evaluations.insert_one(result.dict())
        
        return result

# Initialize services
evaluation_service = CodeEvaluationService()

# API Endpoints
@app.post("/api/evaluate", response_model=EvaluationResult)
async def evaluate_code(request: CodeEvaluationRequest):
    """Evaluate code using dual AI feedback system"""
    try:
        result = await evaluation_service.evaluate_code(request)
        return result
    except Exception as e:
        logger.error(f"Code evaluation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")

@app.get("/api/problems")
async def get_problems(category: Optional[str] = None, difficulty: Optional[str] = None):
    """Get coding problems for practice"""
    try:
        filter_dict = {}
        if category:
            filter_dict["category"] = category
        if difficulty:
            filter_dict["difficulty"] = difficulty
            
        problems = []
        async for problem in db.problems.find(filter_dict).limit(20):
            problem["_id"] = str(problem["_id"])
            problems.append(problem)
            
        return {"problems": problems}
    except Exception as e:
        logger.error(f"Failed to fetch problems: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch problems")

@app.post("/api/problems")
async def create_problem(problem: Problem):
    """Create a new coding problem"""
    try:
        problem_dict = problem.dict()
        result = await db.problems.insert_one(problem_dict)
        problem_dict["_id"] = str(result.inserted_id)
        return problem_dict
    except Exception as e:
        logger.error(f"Failed to create problem: {e}")
        raise HTTPException(status_code=500, detail="Failed to create problem")

@app.get("/api/user/{user_id}/progress")
async def get_user_progress(user_id: str):
    """Get user progress and statistics"""
    try:
        progress = await db.user_progress.find_one({"user_id": user_id})
        if not progress:
            # Create new progress record
            progress = UserProgress(user_id=user_id).dict()
            await db.user_progress.insert_one(progress)
        else:
            progress["_id"] = str(progress["_id"])
        return progress
    except Exception as e:
        logger.error(f"Failed to fetch user progress: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch user progress")

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "ai_systems": {
            "gemini": "ready",
            "starcoder": "ready"
        }
    }

# Initialize database indexes
@app.on_event("startup")
async def startup_event():
    """Initialize database and indexes"""
    try:
        # Create indexes
        await db.problems.create_index([("category", 1), ("difficulty", 1)])
        await db.evaluations.create_index([("timestamp", -1)])
        await db.user_progress.create_index([("user_id", 1)])
        
        # Insert sample problems if collection is empty
        if await db.problems.count_documents({}) == 0:
            sample_problems = [
                {
                    "id": str(uuid.uuid4()),
                    "title": "Two Sum",
                    "description": "Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.",
                    "difficulty": "easy",
                    "category": "arrays",
                    "examples": [
                        {"input": "nums = [2,7,11,15], target = 9", "output": "[0,1]"}
                    ],
                    "created_at": datetime.now()
                },
                {
                    "id": str(uuid.uuid4()),
                    "title": "Reverse Linked List",
                    "description": "Given the head of a singly linked list, reverse the list, and return the reversed list.",
                    "difficulty": "easy",
                    "category": "linked-lists", 
                    "examples": [
                        {"input": "[1,2,3,4,5]", "output": "[5,4,3,2,1]"}
                    ],
                    "created_at": datetime.now()
                },
                {
                    "id": str(uuid.uuid4()),
                    "title": "Binary Tree Maximum Path Sum",
                    "description": "A path in a binary tree is a sequence of nodes where each pair of adjacent nodes in the sequence has an edge connecting them. Find the maximum path sum.",
                    "difficulty": "hard",
                    "category": "trees",
                    "examples": [
                        {"input": "[1,2,3]", "output": "6"}
                    ],
                    "created_at": datetime.now()
                }
            ]
            await db.problems.insert_many(sample_problems)
        
        logger.info("BytePrep API started successfully")
    except Exception as e:
        logger.error(f"Startup failed: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
