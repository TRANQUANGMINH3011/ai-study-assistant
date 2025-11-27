"""
AI Study Assistant - Lambda Handlers
Xử lý tất cả các API endpoints cho 10 Use Cases
"""

try:
    import unzip_requirements # type: ignore
except ImportError:
    pass

import json
import os
import base64
import logging
import boto3
import uuid
from typing import Dict, Any

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize AWS clients
s3_client = boto3.client('s3')
dynamodb = boto3.resource('dynamodb')

# Environment variables
BUCKET_NAME = os.environ.get('BUCKET_NAME', 'ai-study-assistant-files-dev')
DOCUMENTS_TABLE = os.environ.get('DOCUMENTS_TABLE', 'ai-study-assistant-documents-dev')
FLASHCARDS_TABLE = os.environ.get('FLASHCARDS_TABLE', 'ai-study-assistant-flashcards-dev')
QUIZZES_TABLE = os.environ.get('QUIZZES_TABLE', 'ai-study-assistant-quizzes-dev')
SUMMARIES_TABLE = os.environ.get('SUMMARIES_TABLE', 'ai-study-assistant-summaries-dev')
CHAT_HISTORY_TABLE = os.environ.get('CHAT_HISTORY_TABLE', 'ai-study-assistant-chat-history-dev')
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', '')

from datetime import datetime

# =====================================================
# UTILITY FUNCTIONS
# =====================================================

def generate_id(prefix: str = "") -> str:
    """Generate unique ID"""
    unique_id = str(uuid.uuid4())
    return f"{prefix}{unique_id}" if prefix else unique_id

def get_timestamp() -> str:
    """Get current timestamp"""
    return datetime.utcnow().isoformat() + 'Z'

def create_response(status_code: int, body: Any, headers: Dict[str, str] = None) -> Dict[str, Any]:
    """Create API Gateway response"""
    default_headers = {
        'Content-Type': 'application/json',
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Headers': 'Content-Type,Authorization',
        'Access-Control-Allow-Methods': 'GET,POST,PUT,DELETE,OPTIONS'
    }
    if headers:
        default_headers.update(headers)
    
    return {
        'statusCode': status_code,
        'headers': default_headers,
        'body': json.dumps(body, ensure_ascii=False, default=str)
    }

def parse_body(event: Dict[str, Any]) -> Dict[str, Any]:
    """Parse request body"""
    body = event.get('body', '{}')
    if isinstance(body, str):
        return json.loads(body) if body else {}
    return body or {}

def get_user_id_from_event(event: Dict[str, Any]) -> str:
    """Get user ID from Cognito authorizer"""
    try:
        claims = event.get('requestContext', {}).get('authorizer', {}).get('claims', {})
        return claims.get('sub', claims.get('cognito:username', 'anonymous'))
    except Exception:
        return 'anonymous'

# =====================================================
# PDF TEXT EXTRACTION
# =====================================================

def extract_text_from_pdf(pdf_content: bytes, max_chars: int = 100000) -> str:
    """Extract text from PDF"""
    try:
        import PyPDF2
        from io import BytesIO
        
        pdf_file = BytesIO(pdf_content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        
        extracted_text = []
        total_chars = 0
        
        for page in pdf_reader.pages:
            text = page.extract_text() or ""
            if total_chars + len(text) > max_chars:
                remaining = max_chars - total_chars
                extracted_text.append(text[:remaining])
                break
            extracted_text.append(text)
            total_chars += len(text)
        
        return '\n'.join(extracted_text)
    except Exception as e:
        logger.error(f"PDF extraction failed: {e}")
        return ""

def extract_text_from_file(file_content: bytes, file_name: str, max_chars: int = 100000) -> str:
    """Extract text from file"""
    ext = file_name.lower().split('.')[-1]
    if ext == 'pdf':
        return extract_text_from_pdf(file_content, max_chars)
    elif ext in ['docx', 'doc']:
        return extract_text_from_docx(file_content, max_chars)
    elif ext in ['txt', 'text']:
        text = file_content.decode('utf-8')
        return text[:max_chars] if len(text) > max_chars else text
    return ""

def extract_text_from_docx(file_content: bytes, max_chars: int = 100000) -> str:
    """Extract text from DOCX file"""
    try:
        from docx import Document
        import io
        
        doc = Document(io.BytesIO(file_content))
        paragraphs = []
        for para in doc.paragraphs:
            if para.text.strip():
                paragraphs.append(para.text)
        
        text = '\n'.join(paragraphs)
        return text[:max_chars] if len(text) > max_chars else text
    except Exception as e:
        logger.error(f"DOCX extraction failed: {e}")
        return ""

# =====================================================
# GEMINI AI SERVICE (using requests - free API)
# =====================================================

def call_gpt(prompt: str, system_prompt: str = None, max_tokens: int = 4000, temperature: float = 0.7) -> str:
    """Call Google Gemini API using requests (free, lightweight)"""
    import requests
    
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not set")
    
    # Combine system prompt and user prompt for Gemini
    full_prompt = prompt
    if system_prompt:
        full_prompt = f"{system_prompt}\n\n{prompt}"
    
    headers = {
        "Content-Type": "application/json"
    }
    
    payload = {
        "contents": [{
            "parts": [{"text": full_prompt}]
        }],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens
        }
    }
    
    # Use gemini-2.0-flash (available and free)
    response = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}",
        headers=headers,
        json=payload,
        timeout=60
    )
    
    if response.status_code != 200:
        error_detail = response.json().get('error', {}).get('message', response.text)
        raise Exception(f"Gemini API error: {error_detail}")
    
    return response.json()['candidates'][0]['content']['parts'][0]['text']

def call_gpt_json(prompt: str, system_prompt: str = None, max_tokens: int = 4000) -> Any:
    """Call GPT and parse JSON response"""
    result = call_gpt(prompt, system_prompt, max_tokens, 0.7)
    result = result.strip()
    
    # Remove markdown code blocks
    if result.startswith('```json'):
        result = result[7:]
    elif result.startswith('```'):
        result = result[3:]
    if result.endswith('```'):
        result = result[:-3]
    
    return json.loads(result.strip())

# =====================================================
# PROMPTS
# =====================================================

SUMMARIZE_PROMPT = """Bạn là một trợ lý học tập chuyên nghiệp. Hãy tóm tắt tài liệu sau:
- Ngắn gọn nhưng đầy đủ các ý chính
- Có cấu trúc rõ ràng
- Giữ lại các khái niệm quan trọng

Tài liệu:
{content}

Ngôn ngữ: {language}
"""

FLASHCARDS_PROMPT = """Từ nội dung sau, tạo {num_cards} flashcards với format JSON:
[{{"front": "Câu hỏi/Thuật ngữ", "back": "Câu trả lời/Định nghĩa"}}]

CHỈ TRẢ VỀ JSON ARRAY.

Nội dung: {content}
Ngôn ngữ: {language}
"""

QUIZ_PROMPT = """Tạo CHÍNH XÁC {num_questions} câu hỏi trắc nghiệm từ nội dung sau.

YÊU CẦU BẮT BUỘC:
- Số lượng câu hỏi: ĐÚNG {num_questions} câu (không hơn không kém)
- Độ khó: {difficulty}
- Mỗi câu có 4 đáp án A, B, C, D

Format JSON (CHỈ trả về JSON, không giải thích):
[{{
  "question": "Nội dung câu hỏi?",
  "options": ["Đáp án A", "Đáp án B", "Đáp án C", "Đáp án D"],
  "correctAnswer": 0,
  "explanation": "Giải thích ngắn gọn"
}}]

LƯU Ý QUAN TRỌNG:
- options PHẢI là ARRAY chứa đúng 4 chuỗi
- correctAnswer là INDEX (0, 1, 2, hoặc 3)
- Trả về ĐÚNG {num_questions} object trong array

Nội dung tài liệu:
{content}

Ngôn ngữ: {language}
"""

CHAT_SYSTEM_PROMPT = """Bạn là trợ lý học tập. Trả lời câu hỏi dựa trên tài liệu sau:

{document_content}

Chỉ trả lời dựa trên thông tin trong tài liệu."""

EXPLAIN_PROMPT = """Giải thích khái niệm/đoạn văn sau:

Nội dung cần giải thích: {text}

Ngôn ngữ: {language}
"""

# =====================================================
# DYNAMODB OPERATIONS
# =====================================================

def save_document(doc_data: Dict) -> Dict:
    """Save document metadata"""
    table = dynamodb.Table(DOCUMENTS_TABLE)
    document = {
        'documentId': doc_data.get('documentId', generate_id('doc-')),
        'userId': doc_data['userId'],
        'fileName': doc_data['fileName'],
        's3Key': doc_data['s3Key'],
        'fileSize': doc_data.get('fileSize', 0),
        'fileType': doc_data.get('fileType', 'pdf'),
        'extractedText': doc_data.get('extractedText', ''),
        'createdAt': get_timestamp(),
        'status': 'active'
    }
    table.put_item(Item=document)
    return document

def get_document_by_id(document_id: str) -> Dict:
    """Get document by ID"""
    table = dynamodb.Table(DOCUMENTS_TABLE)
    response = table.get_item(Key={'documentId': document_id})
    return response.get('Item')

def get_documents_by_user(user_id: str) -> list:
    """Get all documents for user"""
    table = dynamodb.Table(DOCUMENTS_TABLE)
    response = table.query(
        IndexName='userId-index',
        KeyConditionExpression='userId = :uid',
        ExpressionAttributeValues={':uid': user_id}
    )
    return response.get('Items', [])

def delete_document_by_id(document_id: str) -> bool:
    """Delete document"""
    table = dynamodb.Table(DOCUMENTS_TABLE)
    doc = get_document_by_id(document_id)
    if doc:
        try:
            s3_client.delete_object(Bucket=BUCKET_NAME, Key=doc['s3Key'])
        except Exception as e:
            logger.warning(f"S3 delete failed: {e}")
        table.delete_item(Key={'documentId': document_id})
        return True
    return False

def save_summary(data: Dict) -> Dict:
    """Save summary"""
    table = dynamodb.Table(SUMMARIES_TABLE)
    summary = {
        'summaryId': data.get('summaryId', generate_id('sum-')),
        'documentId': data['documentId'],
        'userId': data['userId'],
        'content': data['content'],
        'language': data.get('language', 'Vietnamese'),
        'createdAt': get_timestamp()
    }
    table.put_item(Item=summary)
    return summary

def get_summary_by_document(document_id: str) -> Dict:
    """Get summary by document ID"""
    table = dynamodb.Table(SUMMARIES_TABLE)
    response = table.query(
        IndexName='documentId-index',
        KeyConditionExpression='documentId = :docId',
        ExpressionAttributeValues={':docId': document_id}
    )
    items = response.get('Items', [])
    return items[0] if items else None

def save_flashcard_set(data: Dict) -> Dict:
    """Save flashcard set"""
    table = dynamodb.Table(FLASHCARDS_TABLE)
    flashcard_set = {
        'flashcardSetId': data.get('flashcardSetId', generate_id('fc-')),
        'documentId': data['documentId'],
        'userId': data['userId'],
        'title': data.get('title', 'Flashcard Set'),
        'cards': data['cards'],
        'cardCount': len(data['cards']),
        'language': data.get('language', 'Vietnamese'),
        'createdAt': get_timestamp()
    }
    table.put_item(Item=flashcard_set)
    return flashcard_set

def get_flashcards_by_document(document_id: str) -> list:
    """Get flashcards by document"""
    table = dynamodb.Table(FLASHCARDS_TABLE)
    response = table.query(
        IndexName='documentId-index',
        KeyConditionExpression='documentId = :docId',
        ExpressionAttributeValues={':docId': document_id}
    )
    return response.get('Items', [])

def get_flashcards_by_user(user_id: str) -> list:
    """Get all flashcards for user"""
    table = dynamodb.Table(FLASHCARDS_TABLE)
    response = table.query(
        IndexName='userId-index',
        KeyConditionExpression='userId = :uid',
        ExpressionAttributeValues={':uid': user_id}
    )
    return response.get('Items', [])

def save_quiz(data: Dict) -> Dict:
    """Save quiz"""
    table = dynamodb.Table(QUIZZES_TABLE)
    quiz = {
        'quizId': data.get('quizId', generate_id('quiz-')),
        'documentId': data['documentId'],
        'userId': data['userId'],
        'title': data.get('title', 'Quiz'),
        'questions': data['questions'],
        'questionCount': len(data['questions']),
        'language': data.get('language', 'Vietnamese'),
        'attempts': [],
        'createdAt': get_timestamp()
    }
    table.put_item(Item=quiz)
    return quiz

def get_quiz_by_id(quiz_id: str) -> Dict:
    """Get quiz by ID"""
    table = dynamodb.Table(QUIZZES_TABLE)
    response = table.get_item(Key={'quizId': quiz_id})
    return response.get('Item')

def get_quizzes_by_user(user_id: str) -> list:
    """Get all quizzes for user"""
    table = dynamodb.Table(QUIZZES_TABLE)
    response = table.query(
        IndexName='userId-index',
        KeyConditionExpression='userId = :uid',
        ExpressionAttributeValues={':uid': user_id}
    )
    return response.get('Items', [])

def save_quiz_attempt(quiz_id: str, attempt_data: Dict) -> Dict:
    """Save quiz attempt"""
    table = dynamodb.Table(QUIZZES_TABLE)
    attempt = {
        'attemptId': generate_id('att-'),
        'answers': attempt_data['answers'],
        'score': attempt_data['score'],
        'totalQuestions': attempt_data['totalQuestions'],
        'percentage': attempt_data['percentage'],
        'completedAt': get_timestamp()
    }
    table.update_item(
        Key={'quizId': quiz_id},
        UpdateExpression='SET attempts = list_append(if_not_exists(attempts, :empty), :attempt)',
        ExpressionAttributeValues={':empty': [], ':attempt': [attempt]}
    )
    return attempt

def save_chat_message(data: Dict) -> Dict:
    """Save chat message"""
    table = dynamodb.Table(CHAT_HISTORY_TABLE)
    chat = {
        'chatId': data.get('chatId', generate_id('chat-')),
        'documentId': data['documentId'],
        'userId': data['userId'],
        'userMessage': data['userMessage'],
        'assistantResponse': data['assistantResponse'],
        'createdAt': get_timestamp()
    }
    table.put_item(Item=chat)
    return chat

def get_chat_history_by_document(document_id: str, user_id: str, limit: int = 20) -> list:
    """Get chat history"""
    table = dynamodb.Table(CHAT_HISTORY_TABLE)
    response = table.query(
        IndexName='documentId-index',
        KeyConditionExpression='documentId = :docId',
        ExpressionAttributeValues={':docId': document_id},
        ScanIndexForward=False,
        Limit=limit
    )
    items = [item for item in response.get('Items', []) if item.get('userId') == user_id]
    return sorted(items, key=lambda x: x.get('createdAt', ''))

# =====================================================
# HEALTH CHECK
# =====================================================

def health_check(event, context):
    """Health check endpoint - GET /health"""
    return create_response(200, {
        "status": "healthy",
        "service": "AI Study Assistant",
        "version": "1.0.0"
    })

# =====================================================
# USE CASE 2: UPLOAD DOCUMENT
# =====================================================

def upload_document(event, context):
    """Upload document - POST /documents/upload"""
    try:
        user_id = get_user_id_from_event(event)
        body = parse_body(event)
        
        # Support 2 ways: base64 file OR plain text content
        file_content_b64 = body.get('file')
        plain_content = body.get('content')
        file_name = body.get('fileName') or body.get('filename', 'document.txt')
        file_type = body.get('fileType') or body.get('contentType', 'text/plain')
        
        if file_content_b64:
            # Way 1: Base64 encoded file
            file_content = base64.b64decode(file_content_b64)
            extracted_text = extract_text_from_file(file_content, file_name)
        elif plain_content:
            # Way 2: Plain text content
            file_content = plain_content.encode('utf-8')
            extracted_text = plain_content
        else:
            return create_response(400, {"error": "Either 'file' (base64) or 'content' (text) is required"})
        
        file_size = len(file_content)
        
        # Upload to S3
        s3_key = f"users/{user_id}/documents/{generate_id()}/{file_name}"
        s3_client.put_object(
            Bucket=BUCKET_NAME,
            Key=s3_key,
            Body=file_content,
            ContentType=file_type
        )
        
        # Save to DynamoDB
        document = save_document({
            'userId': user_id,
            'fileName': file_name,
            's3Key': s3_key,
            'fileSize': file_size,
            'fileType': file_name.split('.')[-1].lower(),
            'extractedText': extracted_text[:50000]
        })
        
        response_doc = {k: v for k, v in document.items() if k != 'extractedText'}
        response_doc['textExtracted'] = len(extracted_text) > 0
        
        return create_response(201, {
            "message": "Document uploaded successfully",
            "document": response_doc
        })
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        return create_response(500, {"error": str(e)})

def get_upload_url(event, context):
    """Get presigned URL - POST /documents/upload-url"""
    try:
        user_id = get_user_id_from_event(event)
        body = parse_body(event)
        
        file_name = body.get('fileName', 'document.pdf')
        content_type = body.get('fileType', 'application/pdf')
        
        s3_key = f"users/{user_id}/documents/{generate_id()}/{file_name}"
        
        url = s3_client.generate_presigned_url(
            'put_object',
            Params={'Bucket': BUCKET_NAME, 'Key': s3_key, 'ContentType': content_type},
            ExpiresIn=3600
        )
        
        return create_response(200, {
            "uploadUrl": url,
            "s3Key": s3_key,
            "expiresIn": 3600
        })
    except Exception as e:
        logger.error(f"Get upload URL failed: {e}")
        return create_response(500, {"error": str(e)})

# =====================================================
# USE CASE 3: SUMMARIZE DOCUMENT
# =====================================================

def summarize_document(event, context):
    """Summarize document - POST /documents/{documentId}/summarize"""
    try:
        user_id = get_user_id_from_event(event)
        document_id = event['pathParameters']['documentId']
        body = parse_body(event)
        
        language = body.get('language', 'Vietnamese')
        
        document = get_document_by_id(document_id)
        if not document:
            return create_response(404, {"error": "Document not found"})
        
        if document.get('userId') != user_id:
            return create_response(403, {"error": "Access denied"})
        
        extracted_text = document.get('extractedText', '')
        if not extracted_text:
            response = s3_client.get_object(Bucket=BUCKET_NAME, Key=document['s3Key'])
            file_content = response['Body'].read()
            extracted_text = extract_text_from_file(file_content, document['fileName'])
        
        if not extracted_text:
            return create_response(400, {"error": "No text content available"})
        
        # Generate summary
        prompt = SUMMARIZE_PROMPT.format(content=extracted_text[:15000], language=language)
        summary_content = call_gpt(prompt, max_tokens=2000, temperature=0.5)
        
        summary = save_summary({
            'documentId': document_id,
            'userId': user_id,
            'content': summary_content,
            'language': language
        })
        
        return create_response(200, {
            "message": "Summary generated successfully",
            "summary": summary
        })
    except Exception as e:
        logger.error(f"Summarize failed: {e}")
        return create_response(500, {"error": str(e)})

def get_summary(event, context):
    """Get summary - GET /documents/{documentId}/summary"""
    try:
        user_id = get_user_id_from_event(event)
        document_id = event['pathParameters']['documentId']
        
        document = get_document_by_id(document_id)
        if not document:
            return create_response(404, {"error": "Document not found"})
        
        if document.get('userId') != user_id:
            return create_response(403, {"error": "Access denied"})
        
        summary = get_summary_by_document(document_id)
        if not summary:
            return create_response(404, {"error": "Summary not found"})
        
        return create_response(200, {"summary": summary})
    except Exception as e:
        logger.error(f"Get summary failed: {e}")
        return create_response(500, {"error": str(e)})

# =====================================================
# USE CASE 4 & 5: FLASHCARDS
# =====================================================

def create_flashcards(event, context):
    """Create flashcards - POST /documents/{documentId}/flashcards"""
    try:
        user_id = get_user_id_from_event(event)
        document_id = event['pathParameters']['documentId']
        body = parse_body(event)
        
        # Support both parameter names
        num_cards = body.get('count') or body.get('numCards') or 20
        language = body.get('language', 'Vietnamese')
        title = body.get('title', 'Flashcard Set')
        
        document = get_document_by_id(document_id)
        if not document:
            return create_response(404, {"error": "Document not found"})
        
        if document.get('userId') != user_id:
            return create_response(403, {"error": "Access denied"})
        
        extracted_text = document.get('extractedText', '')
        if not extracted_text:
            return create_response(400, {"error": "No text content available"})
        
        # Generate flashcards
        prompt = FLASHCARDS_PROMPT.format(
            content=extracted_text[:15000],
            num_cards=num_cards,
            language=language
        )
        cards = call_gpt_json(prompt, max_tokens=4000)
        
        flashcard_set = save_flashcard_set({
            'documentId': document_id,
            'userId': user_id,
            'title': title,
            'cards': cards,
            'language': language
        })
        
        return create_response(201, {
            "message": "Flashcards created successfully",
            "flashcardSet": flashcard_set,
            "flashcards": cards  # Also return cards directly for frontend
        })
    except Exception as e:
        logger.error(f"Create flashcards failed: {e}")
        return create_response(500, {"error": str(e)})

def get_flashcards(event, context):
    """Get flashcards - GET /documents/{documentId}/flashcards"""
    try:
        user_id = get_user_id_from_event(event)
        document_id = event['pathParameters']['documentId']
        
        document = get_document_by_id(document_id)
        if not document:
            return create_response(404, {"error": "Document not found"})
        
        if document.get('userId') != user_id:
            return create_response(403, {"error": "Access denied"})
        
        flashcard_sets = get_flashcards_by_document(document_id)
        return create_response(200, {"flashcardSets": flashcard_sets})
    except Exception as e:
        logger.error(f"Get flashcards failed: {e}")
        return create_response(500, {"error": str(e)})

def list_flashcards(event, context):
    """List all flashcards - GET /flashcards"""
    try:
        user_id = get_user_id_from_event(event)
        flashcard_sets = get_flashcards_by_user(user_id)
        return create_response(200, {
            "flashcardSets": flashcard_sets,
            "count": len(flashcard_sets)
        })
    except Exception as e:
        logger.error(f"List flashcards failed: {e}")
        return create_response(500, {"error": str(e)})

# =====================================================
# USE CASE 6 & 7: QUIZ
# =====================================================

def create_quiz(event, context):
    """Create quiz - POST /documents/{documentId}/quiz"""
    try:
        user_id = get_user_id_from_event(event)
        document_id = event['pathParameters']['documentId']
        body = parse_body(event)
        
        # Support both parameter names
        num_questions = body.get('questionCount') or body.get('numQuestions') or 10
        language = body.get('language', 'Vietnamese')
        title = body.get('title', 'Quiz')
        difficulty = body.get('difficulty', 'medium')
        
        document = get_document_by_id(document_id)
        if not document:
            return create_response(404, {"error": "Document not found"})
        
        if document.get('userId') != user_id:
            return create_response(403, {"error": "Access denied"})
        
        extracted_text = document.get('extractedText', '')
        if not extracted_text:
            return create_response(400, {"error": "No text content available"})
        
        # Generate quiz
        prompt = QUIZ_PROMPT.format(
            content=extracted_text[:15000],
            num_questions=num_questions,
            language=language,
            difficulty=difficulty
        )
        questions = call_gpt_json(prompt, max_tokens=4000)
        
        # Ensure we have exactly the requested number of questions
        if isinstance(questions, list) and len(questions) > num_questions:
            questions = questions[:num_questions]
        
        quiz = save_quiz({
            'documentId': document_id,
            'userId': user_id,
            'title': title,
            'questions': questions,
            'language': language
        })
        
        return create_response(201, {
            "message": "Quiz created successfully",
            "quiz": quiz
        })
    except Exception as e:
        logger.error(f"Create quiz failed: {e}")
        return create_response(500, {"error": str(e)})

def get_quiz(event, context):
    """Get quiz - GET /quizzes/{quizId}"""
    try:
        user_id = get_user_id_from_event(event)
        quiz_id = event['pathParameters']['quizId']
        
        quiz = get_quiz_by_id(quiz_id)
        if not quiz:
            return create_response(404, {"error": "Quiz not found"})
        
        if quiz.get('userId') != user_id:
            return create_response(403, {"error": "Access denied"})
        
        # Hide correct answers
        questions_for_user = []
        for q in quiz.get('questions', []):
            questions_for_user.append({
                'question': q['question'],
                'options': q['options']
            })
        
        return create_response(200, {
            "quiz": {
                'quizId': quiz['quizId'],
                'title': quiz.get('title', 'Quiz'),
                'questions': questions_for_user,
                'questionCount': len(questions_for_user)
            }
        })
    except Exception as e:
        logger.error(f"Get quiz failed: {e}")
        return create_response(500, {"error": str(e)})

def submit_quiz(event, context):
    """Submit quiz - POST /quizzes/{quizId}/submit"""
    try:
        user_id = get_user_id_from_event(event)
        quiz_id = event['pathParameters']['quizId']
        body = parse_body(event)
        
        answers = body.get('answers', {})
        
        quiz = get_quiz_by_id(quiz_id)
        if not quiz:
            return create_response(404, {"error": "Quiz not found"})
        
        if quiz.get('userId') != user_id:
            return create_response(403, {"error": "Access denied"})
        
        # Grade quiz
        questions = quiz['questions']
        total = len(questions)
        correct = 0
        results = []
        
        for idx, question in enumerate(questions):
            user_answer = answers.get(str(idx))
            correct_answer = question['correct_answer']
            is_correct = user_answer == correct_answer
            
            if is_correct:
                correct += 1
            
            results.append({
                'question_index': idx,
                'user_answer': user_answer,
                'correct_answer': correct_answer,
                'is_correct': is_correct,
                'explanation': question.get('explanation', '')
            })
        
        percentage = round((correct / total) * 100, 1) if total > 0 else 0
        
        # Save attempt
        attempt = save_quiz_attempt(quiz_id, {
            'answers': answers,
            'score': correct,
            'totalQuestions': total,
            'percentage': percentage
        })
        
        return create_response(200, {
            "message": "Quiz submitted",
            "result": {
                'score': correct,
                'totalQuestions': total,
                'percentage': percentage,
                'results': results,
                'passed': percentage >= 60
            }
        })
    except Exception as e:
        logger.error(f"Submit quiz failed: {e}")
        return create_response(500, {"error": str(e)})

def list_quizzes(event, context):
    """List quizzes - GET /quizzes"""
    try:
        user_id = get_user_id_from_event(event)
        quizzes = get_quizzes_by_user(user_id)
        
        quizzes_summary = []
        for q in quizzes:
            quizzes_summary.append({
                'quizId': q['quizId'],
                'documentId': q['documentId'],
                'title': q.get('title', 'Quiz'),
                'questionCount': q.get('questionCount', 0),
                'attempts': len(q.get('attempts', [])),
                'createdAt': q.get('createdAt')
            })
        
        return create_response(200, {
            "quizzes": quizzes_summary,
            "count": len(quizzes_summary)
        })
    except Exception as e:
        logger.error(f"List quizzes failed: {e}")
        return create_response(500, {"error": str(e)})

# =====================================================
# USE CASE 8: CHAT WITH DOCUMENT
# =====================================================

def chat_with_document(event, context):
    """Chat with document - POST /documents/{documentId}/chat"""
    try:
        user_id = get_user_id_from_event(event)
        document_id = event['pathParameters']['documentId']
        body = parse_body(event)
        
        # Support both 'question' and 'message' fields
        question = body.get('question') or body.get('message', '')
        language = body.get('language', 'Vietnamese')
        
        if not question:
            return create_response(400, {"error": "Question/message is required"})
        
        document = get_document_by_id(document_id)
        if not document:
            return create_response(404, {"error": "Document not found"})
        
        if document.get('userId') != user_id:
            return create_response(403, {"error": "Access denied"})
        
        extracted_text = document.get('extractedText', '')
        if not extracted_text:
            return create_response(400, {"error": "No text content available"})
        
        # Generate response
        system_prompt = CHAT_SYSTEM_PROMPT.format(document_content=extracted_text[:10000])
        user_prompt = f"Câu hỏi: {question}\n\nTrả lời bằng {language}:"
        
        response = call_gpt(user_prompt, system_prompt, max_tokens=2000)
        
        # Save chat
        chat = save_chat_message({
            'documentId': document_id,
            'userId': user_id,
            'userMessage': question,
            'assistantResponse': response
        })
        
        return create_response(200, {
            "question": question,
            "answer": response,
            "chatId": chat['chatId']
        })
    except Exception as e:
        logger.error(f"Chat failed: {e}")
        return create_response(500, {"error": str(e)})

def get_chat_history(event, context):
    """Get chat history - GET /documents/{documentId}/chat/history"""
    try:
        user_id = get_user_id_from_event(event)
        document_id = event['pathParameters']['documentId']
        
        document = get_document_by_id(document_id)
        if not document:
            return create_response(404, {"error": "Document not found"})
        
        if document.get('userId') != user_id:
            return create_response(403, {"error": "Access denied"})
        
        history = get_chat_history_by_document(document_id, user_id)
        
        return create_response(200, {
            "chatHistory": history,
            "count": len(history)
        })
    except Exception as e:
        logger.error(f"Get chat history failed: {e}")
        return create_response(500, {"error": str(e)})

# =====================================================
# USE CASE 9: EXPLAIN CONCEPT (ELI5)
# =====================================================

def explain_concept(event, context):
    """Explain concept - POST /explain"""
    try:
        body = parse_body(event)
        
        # Support both 'text' and 'concept' fields
        text = body.get('text') or body.get('concept', '')
        language = body.get('language', 'Vietnamese')
        level = body.get('level', 'eli5')
        
        if not text:
            return create_response(400, {"error": "Text or concept is required"})
        
        if len(text) > 5000:
            return create_response(400, {"error": "Text too long"})
        
        # Adjust prompt based on level
        level_instructions = {
            'eli5': """Giải thích như đang nói với TRẺ 5 TUỔI:
- Dùng từ ngữ cực kỳ đơn giản
- Dùng ví dụ quen thuộc (bánh kẹo, đồ chơi, hoạt hình)
- Tránh hoàn toàn thuật ngữ kỹ thuật
- Câu ngắn, vui vẻ""",
            
            'beginner': """Giải thích cho NGƯỜI MỚI BẮT ĐẦU:
- Dùng ngôn ngữ đơn giản, dễ hiểu
- Giải thích các thuật ngữ cơ bản
- Đưa ví dụ thực tế dễ liên hệ
- Không giả định kiến thức nền""",
            
            'intermediate': """Giải thích ở mức TRUNG CẤP:
- Giả định người đọc có kiến thức cơ bản
- Đi sâu hơn vào chi tiết
- Sử dụng thuật ngữ chuyên ngành nhưng giải thích rõ
- Đưa ví dụ thực tế trong ngành""",
            
            'expert': """Giải thích ở mức CHUYÊN GIA:
- Giả định người đọc có nền tảng chuyên môn
- Sử dụng thuật ngữ kỹ thuật chính xác
- Phân tích chi tiết và chuyên sâu
- Thảo luận ưu/nhược điểm, so sánh với các khái niệm liên quan
- Có thể đề cập đến các nghiên cứu, tiêu chuẩn ngành"""
        }
        
        level_instruction = level_instructions.get(level, level_instructions['eli5'])
        
        # Generate explanation
        prompt = f"""{level_instruction}

{EXPLAIN_PROMPT.format(text=text, language=language)}"""
        explanation = call_gpt(prompt, max_tokens=2000, temperature=0.7)
        
        return create_response(200, {
            "originalText": text,
            "level": level,
            "explanation": explanation
        })
    except Exception as e:
        logger.error(f"Explain failed: {e}")
        return create_response(500, {"error": str(e)})

# =====================================================
# USE CASE 10: DOCUMENT MANAGEMENT
# =====================================================

def list_documents(event, context):
    """List documents - GET /documents"""
    try:
        user_id = get_user_id_from_event(event)
        documents = get_documents_by_user(user_id)
        
        docs_summary = []
        for doc in documents:
            docs_summary.append({
                'documentId': doc['documentId'],
                'fileName': doc['fileName'],
                'fileSize': doc.get('fileSize', 0),
                'fileType': doc.get('fileType', 'pdf'),
                'createdAt': doc.get('createdAt'),
                'status': doc.get('status', 'active'),
                'hasText': bool(doc.get('extractedText'))
            })
        
        return create_response(200, {
            "documents": docs_summary,
            "count": len(docs_summary)
        })
    except Exception as e:
        logger.error(f"List documents failed: {e}")
        return create_response(500, {"error": str(e)})

def get_document(event, context):
    """Get document - GET /documents/{documentId}"""
    try:
        user_id = get_user_id_from_event(event)
        document_id = event['pathParameters']['documentId']
        
        document = get_document_by_id(document_id)
        if not document:
            return create_response(404, {"error": "Document not found"})
        
        if document.get('userId') != user_id:
            return create_response(403, {"error": "Access denied"})
        
        # Get related data
        summary = get_summary_by_document(document_id)
        flashcards = get_flashcards_by_document(document_id)
        
        doc_response = {k: v for k, v in document.items() if k != 'extractedText'}
        doc_response['hasText'] = bool(document.get('extractedText'))
        
        return create_response(200, {
            "document": doc_response,
            "summary": summary,
            "flashcardSets": flashcards
        })
    except Exception as e:
        logger.error(f"Get document failed: {e}")
        return create_response(500, {"error": str(e)})

def delete_document(event, context):
    """Delete document - DELETE /documents/{documentId}"""
    try:
        user_id = get_user_id_from_event(event)
        document_id = event['pathParameters']['documentId']
        
        document = get_document_by_id(document_id)
        if not document:
            return create_response(404, {"error": "Document not found"})
        
        if document.get('userId') != user_id:
            return create_response(403, {"error": "Access denied"})
        
        delete_document_by_id(document_id)
        
        return create_response(200, {
            "message": "Document deleted successfully",
            "documentId": document_id
        })
    except Exception as e:
        logger.error(f"Delete document failed: {e}")
        return create_response(500, {"error": str(e)})

def create_podcast(event, context):
    """
    Hàm chính: Lấy PDF từ S3 -> Tạo Podcast -> Lưu Audio lên S3
    """
    try:
        body = json.loads(event['body'])
        s3_key = body.get('s3Key') # Đường dẫn file PDF trên S3
        
        if not s3_key:
            return {"statusCode": 400, "body": "Missing s3Key"}

        # 1. Tải file từ S3 về thư mục tạm /tmp/ của Lambda
        local_pdf_path = f"/tmp/{os.path.basename(s3_key)}"
        s3_client.download_file(BUCKET_NAME, s3_key, local_pdf_path)
        
        # 2. Cấu hình AI (Lấy API Key từ biến môi trường AWS)
        config = base_config.copy()
        
        # Nếu bạn dùng OpenAI (như file serverless.yml hiện tại)
        openai_key = os.environ.get("OPENAI_API_KEY")
        if openai_key:
            config["Big-Text-Model"]["provider"]["key"] = openai_key
            config["Small-Text-Model"]["provider"]["key"] = openai_key
            config["Text-To-Speech-Model"]["provider"]["key"] = openai_key
            
            config["Big-Text-Model"]["provider"]["name"] = "openai"
            config["Big-Text-Model"]["model"] = "gpt-4o-mini"
            config["Small-Text-Model"]["provider"]["name"] = "openai"
            config["Small-Text-Model"]["model"] = "gpt-4o-mini"
            config["Text-To-Speech-Model"]["provider"]["name"] = "openai"
            config["Text-To-Speech-Model"]["model"] = "tts-1"

        # 3. Gọi hàm xử lý AI (Output vào /tmp/)
        output_dir = f"/tmp/output_{uuid.uuid4()}"
        success, result_path = podcast_processor(
            pdf_path=local_pdf_path,
            config_path=None,
            format_config=config, # Truyền config trực tiếp
            output_dir=output_dir,
            format_type="summary"
        )
        
        if not success:
            return {"statusCode": 500, "body": f"AI Error: {result_path}"}

        # 4. Upload file kết quả (Audio) lên S3
        # Lấy file audio thật (thường nằm trong thư mục con hoặc tên là podcast.wav)
        # Logic của processor trả về đường dẫn file cuối cùng
        final_audio_key = s3_key.replace("uploads/", "generated/").replace(".pdf", ".wav")
        s3_client.upload_file(result_path, BUCKET_NAME, final_audio_key)
        
        return {
            "statusCode": 200,
            "body": json.dumps({
                "message": "Podcast created successfully!",
                "audioS3Key": final_audio_key
            })
        }

    except Exception as e:
        import traceback
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e), "trace": traceback.format_exc()})
        }