# 🎓 AI Study Assistant

> **Trợ lý học tập thông minh** - Sử dụng AI để tạo Flashcards, Quiz, Chat với tài liệu và giải thích khái niệm theo nhiều cấp độ.

![Neo-brutalism Design](https://img.shields.io/badge/Design-Neo--brutalism-yellow?style=for-the-badge)
![AWS Lambda](https://img.shields.io/badge/Backend-AWS%20Lambda-orange?style=for-the-badge)
![Gemini AI](https://img.shields.io/badge/AI-Google%20Gemini%202.0-blue?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.9+-green?style=for-the-badge)

---

## 📸 Screenshots

<div align="center">
  <img src="https://img.shields.io/badge/Status-Live-brightgreen?style=for-the-badge" alt="Status"/>
</div>

---

## ✨ Tính năng

| Tính năng | Mô tả |
|-----------|-------|
| 📤 **Upload tài liệu** | Hỗ trợ PDF, DOCX, TXT hoặc paste nội dung trực tiếp |
| 🎴 **Flashcards** | Tự động tạo flashcards với số lượng tùy chỉnh |
| ❓ **Quiz** | Tạo quiz trắc nghiệm với 3 mức độ: Dễ, Trung bình, Khó |
| 💬 **Chat RAG** | Hỏi đáp thông minh dựa trên nội dung tài liệu |
| 🧒 **ELI5** | Giải thích khái niệm theo 4 cấp độ: ELI5, Beginner, Intermediate, Expert |
| 📝 **Tóm tắt** | Tóm tắt nội dung tài liệu tự động |
| 📚 **Quản lý tài liệu** | Xem, xóa, quản lý tài liệu đã upload |

---

## 🏗️ Kiến trúc hệ thống

```
┌─────────────────┐     ┌─────────────────────────────────────────────────┐
│                 │     │              AWS Cloud                          │
│   Frontend      │     │  ┌─────────────────────────────────────────┐   │
│   (HTML/CSS)    │────▶│  │         API Gateway (REST)              │   │
│                 │     │  │         + CORS enabled                  │   │
└─────────────────┘     │  └────────────────┬────────────────────────┘   │
                        │                   │                             │
                        │  ┌────────────────▼────────────────────────┐   │
                        │  │          AWS Lambda (Python 3.9)        │   │
                        │  │          - 18 Functions                 │   │
                        │  │          - handler.py                   │   │
                        │  └────┬───────────┬───────────┬────────────┘   │
                        │       │           │           │                 │
                        │  ┌────▼────┐ ┌────▼────┐ ┌────▼────┐           │
                        │  │   S3   │ │DynamoDB │ │ Gemini  │           │
                        │  │ Bucket │ │ Tables  │ │  API    │           │
                        │  └────────┘ └─────────┘ └─────────┘           │
                        └─────────────────────────────────────────────────┘
```

### DynamoDB Tables
- `Documents` - Lưu trữ metadata tài liệu
- `Flashcards` - Lưu trữ flashcard sets
- `Quizzes` - Lưu trữ quiz và kết quả
- `Summaries` - Lưu trữ tóm tắt
- `ChatHistory` - Lưu trữ lịch sử chat

---

## 🚀 Hướng dẫn cài đặt

### Yêu cầu hệ thống

- **Node.js** >= 18.x
- **Python** >= 3.9
- **AWS CLI** đã cấu hình
- **Serverless Framework** 3.x

### Bước 1: Clone dự án

```bash
git clone https://github.com/TRANQUANGMINH3011/ai-study-assistant.git
cd ai-study-assistant
```

### Bước 2: Cài đặt dependencies

```bash
# Cài đặt Node.js dependencies
npm install

# Cài đặt Serverless Framework (nếu chưa có)
npm install -g serverless
```

### Bước 3: Lấy Gemini API Key (Miễn phí)

1. Truy cập: https://aistudio.google.com/app/apikey
2. Đăng nhập tài khoản Google
3. Click **"Create API Key"**
4. Copy API key

### Bước 4: Deploy Backend lên AWS

**Windows (PowerShell):**
```powershell
$env:GEMINI_API_KEY = "your-gemini-api-key-here"
serverless deploy --stage dev
```

**Linux/Mac:**
```bash
export GEMINI_API_KEY="your-gemini-api-key-here"
serverless deploy --stage dev
```

Sau khi deploy thành công, bạn sẽ thấy output như:
```
endpoints:
  POST - https://xxx.execute-api.ap-southeast-1.amazonaws.com/dev/documents/upload
  POST - https://xxx.execute-api.ap-southeast-1.amazonaws.com/dev/documents/{documentId}/flashcards
  ...
```

### Bước 5: Cập nhật API URL trong Frontend

Mở file `frontend/index.html`, tìm và thay đổi dòng:

```javascript
const API_BASE = 'https://YOUR-API-ID.execute-api.ap-southeast-1.amazonaws.com/dev';
```

Thay `YOUR-API-ID` bằng API ID từ output ở bước 4.

### Bước 6: Chạy Frontend

#### Cách 1: Mở trực tiếp file HTML
```bash
# Windows
start frontend/index.html

# Mac
open frontend/index.html

# Linux
xdg-open frontend/index.html
```

#### Cách 2: Dùng Live Server (VS Code)
1. Cài extension **Live Server** trong VS Code
2. Mở file `frontend/index.html`
3. Click **"Go Live"** ở góc dưới phải

#### Cách 3: Dùng Python HTTP Server
```bash
cd frontend
python -m http.server 8080
```
Mở trình duyệt: http://localhost:8080

#### Cách 4: Dùng Node.js serve
```bash
npx serve frontend
```

---

## 📁 Cấu trúc dự án

```
ai-study-assistant/
├── handler.py              # Lambda handlers - 18 functions (~1100 lines)
├── serverless.yml          # AWS infrastructure config
├── requirements.txt        # Python: requests, PyPDF2, python-docx
├── package.json            # Node.js dependencies
├── README.md               # Tài liệu này
├── frontend/
│   └── index.html          # Giao diện Neo-brutalism (Single file ~1500 lines)
└── local/                  # Code tham khảo từ Local NotebookLM
    ├── config.py
    ├── processor.py
    └── app/
        ├── prompts.py      # Các prompt mẫu
        └── step1-4.py      # Processing steps
```

---

## 🔌 API Endpoints

**Base URL:** `https://0q3mju5aqe.execute-api.ap-southeast-1.amazonaws.com/dev`

### Tài liệu
| Method | Endpoint | Mô tả |
|--------|----------|-------|
| `POST` | `/documents/upload` | Upload tài liệu (PDF/DOCX/TXT hoặc text) |
| `GET` | `/documents` | Lấy danh sách tài liệu |
| `GET` | `/documents/{id}` | Lấy chi tiết tài liệu |
| `DELETE` | `/documents/{id}` | Xóa tài liệu |

### AI Features
| Method | Endpoint | Body | Mô tả |
|--------|----------|------|-------|
| `POST` | `/documents/{id}/flashcards` | `{count: 10}` | Tạo flashcards |
| `GET` | `/documents/{id}/flashcards` | - | Lấy flashcards đã tạo |
| `POST` | `/documents/{id}/quiz` | `{questionCount: 5, difficulty: "medium"}` | Tạo quiz |
| `POST` | `/documents/{id}/chat` | `{question: "..."}` | Chat với tài liệu |
| `POST` | `/documents/{id}/summarize` | `{language: "Vietnamese"}` | Tóm tắt |
| `POST` | `/explain` | `{text: "...", level: "eli5"}` | Giải thích (4 levels) |
| `GET` | `/health` | - | Health check |

---

## 📝 Ví dụ sử dụng API (Postman/cURL)

### 1. Upload tài liệu
```bash
curl -X POST https://your-api/dev/documents/upload \
  -H "Content-Type: application/json" \
  -d '{
    "filename": "machine-learning.txt",
    "content": "Machine Learning là một nhánh của trí tuệ nhân tạo...",
    "contentType": "text/plain"
  }'
```

### 2. Tạo Flashcards
```bash
curl -X POST https://your-api/dev/documents/{documentId}/flashcards \
  -H "Content-Type: application/json" \
  -d '{"count": 10}'
```

**Response:**
```json
{
  "message": "Flashcards created successfully",
  "flashcards": [
    {"front": "Machine Learning là gì?", "back": "ML là nhánh của AI..."},
    ...
  ]
}
```

### 3. Tạo Quiz
```bash
curl -X POST https://your-api/dev/documents/{documentId}/quiz \
  -H "Content-Type: application/json" \
  -d '{"questionCount": 5, "difficulty": "medium"}'
```

**Response:**
```json
{
  "quiz": {
    "questions": [
      {
        "question": "Machine Learning là gì?",
        "options": ["Đáp án A", "Đáp án B", "Đáp án C", "Đáp án D"],
        "correctAnswer": 0,
        "explanation": "Giải thích..."
      }
    ]
  }
}
```

### 4. Chat với tài liệu
```bash
curl -X POST https://your-api/dev/documents/{documentId}/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "Machine Learning là gì?"}'
```

### 5. Giải thích ELI5 (4 cấp độ)
```bash
# Cấp độ: eli5, beginner, intermediate, expert
curl -X POST https://your-api/dev/explain \
  -H "Content-Type: application/json" \
  -d '{"text": "Blockchain", "level": "expert"}'
```

**Các cấp độ giải thích:**
| Level | Mô tả |
|-------|-------|
| `eli5` | Giải thích đơn giản như cho trẻ 5 tuổi |
| `beginner` | Giải thích cơ bản cho người mới bắt đầu |
| `intermediate` | Giải thích chi tiết với ví dụ |
| `expert` | Giải thích chuyên sâu, kỹ thuật |

---

## 🎨 Giao diện Neo-brutalism

Giao diện được thiết kế theo phong cách **Neo-brutalism** với các đặc điểm:

- ✅ **Bold borders** - Viền đậm 3-5px
- ✅ **Strong shadows** - Bóng đổ mạnh (4px 4px 0px black)
- ✅ **Vivid colors** - Màu sắc tươi sáng (Yellow, Pink, Blue, Green, Purple)
- ✅ **Chunky elements** - Các thành phần lớn, dễ tương tác
- ✅ **No rounded corners** - Không bo góc

---

## 🛠️ Xử lý lỗi thường gặp

### ❌ Lỗi "GEMINI_API_KEY is not set"
```bash
# Kiểm tra biến môi trường đã set chưa
echo $env:GEMINI_API_KEY  # Windows PowerShell
echo $GEMINI_API_KEY      # Linux/Mac
```

### ❌ Lỗi "Lambda package too large" (>250MB)
```bash
# Xóa cache và deploy lại
Remove-Item -Recurse -Force .serverless
serverless deploy --stage dev
```

### ❌ Lỗi CORS khi gọi API từ Frontend
Đảm bảo trong `serverless.yml` đã có `cors: true`:
```yaml
events:
  - http:
      path: documents/upload
      method: post
      cors: true  # Quan trọng!
```

### ❌ Lỗi "Model not found" với Gemini API
Sử dụng model có sẵn trong `handler.py`:
```python
"gemini-2.0-flash"  # ✅ Model đang hoạt động
```

---

## 🔧 Tùy chỉnh

### Đổi màu giao diện
Mở `frontend/index.html`, tìm phần `:root`:
```css
:root {
    --yellow: #FFE600;
    --pink: #FF6B9D;
    --blue: #00D4FF;
    --green: #7CFF6B;
    --purple: #B794F6;
}
```

### Đổi model AI
Mở `handler.py`, tìm URL Gemini API:
```python
f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
```

---

## 📄 License

MIT License - Tự do sử dụng và chỉnh sửa.

---

## 🤝 Đóng góp

1. Fork dự án
2. Tạo branch: `git checkout -b feature/tinh-nang-moi`
3. Commit: `git commit -m 'Thêm tính năng mới'`
4. Push: `git push origin feature/tinh-nang-moi`
5. Tạo Pull Request

---

## 📊 Tech Stack

| Layer | Technology |
|-------|------------|
| Frontend | HTML5, CSS3 (Neo-brutalism), Vanilla JS |
| Backend | AWS Lambda (Python 3.9) |
| API | AWS API Gateway (REST) |
| Database | AWS DynamoDB (5 tables) |
| Storage | AWS S3 |
| AI | Google Gemini 2.0 Flash |
| Infrastructure | Serverless Framework 3.x |

---

## 👨‍💻 Tác giả

**TRAN QUANG MINH** - B23DCCC112
**TA TIEN LOC** - B23DCCC100
**NGUYEN MINH HIEU** - B23DCCC066
**PHAM MINH HIEN** - B23DCCC059
**LE HOANG HAI** - B23DCCC058

---

## 📝 Changelog

### v1.0.0 (2024-11-27)
- ✅ Initial release
- ✅ Upload documents (PDF, DOCX, TXT, direct text)
- ✅ Create flashcards with custom count
- ✅ Create quiz with 3 difficulty levels
- ✅ Chat with documents (RAG)
- ✅ ELI5 explain with 4 levels
- ✅ Neo-brutalism UI design
- ✅ Gemini 2.0 Flash integration

---

⭐ **Nếu dự án hữu ích, hãy cho một star nhé!** ⭐
