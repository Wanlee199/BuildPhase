# Code Review: KTCN Backend API Project Structure

## 1. Yêu cầu (User Request)
Kiểm tra qua một chút về repo cấu trúc file và thư mục (`https://github.com/nguyenhuynhdt37/ktcn-backend-api`) xem có ổn không và cho đánh giá.

---

## 2. Kết quả Review Cấu trúc (Review Results)

Nhìn vào cây thư mục của project, dự án đang áp dụng kiến trúc **Modular Monolith (Monolith theo Domain)** kết hợp với chuẩn cấu trúc hiện đại của FastAPI. Đây là một cấu trúc rất chuyên nghiệp và bền vững cho các dự án cỡ vừa đến lớn.

### 🌟 Những điểm cộng lớn (Strengths)

*   **Chia module theo Domain (Domain-Driven Design - DDD lite):** 
    Thay vì nhóm tất cả router vào một folder `routers/`, tất cả models vào `models/` (kiểu Django/Flask cũ), dự án đã chia theo từng chức năng tại `app/modules/` (`article/`, `auth/`, `category/`, `staff/`...). 
    👉 **Lợi ích:** Tính đóng gói (encapsulation) cực cao. Khi cần sửa tính năng Article, developer chỉ cần vào đúng folder `app/modules/article/` để làm việc. Rất dễ để tách thành Microservices sau này nếu hệ thống phình to.
*   **Cấu trúc bên trong mỗi module cực kỳ chuẩn mực:**
    Mỗi module đều có sự tách bạch rõ ràng giữa các layer:
    *   `models.py`: Khai báo Database Schema (SQLAlchemy).
    *   `schemas/` (hoặc `schemas.py`): Khai báo Pydantic schemas để validate Request/Response.
    *   `service.py`: Nơi chứa toàn bộ Business Logic.
    *   `routers/`: (thường chia ra `admin.py`, `portal.py`) xử lý HTTP Request và gọi xuống service.
*   **Tách biệt Core, Common và Shared:**
    *   `app/core/`: Chứa các thành phần cốt lõi để chạy app (config, middleware, security, db engine, logger).
    *   `app/common/`: Chứa các Base Model, Base Repository dùng chung.
    *   `app/shared/`: Chứa các external services/helpers như AI, Redis, SEO helper. Sự phân định này rất logic.
*   **Quản lý Database và Migration tốt:**
    *   Tách biệt rõ ràng `migrations/` (dành cho Alembic tracking) và `database/scripts/`, `database/schema/` (dành cho raw SQL seeds, fake data). 
*   **Hệ thống Document (`docs/`) đồ sộ:** 
    Có hẳn một thư mục `docs/` chứa API specs, guide cho Frontend team (như `fe_client_language_guide.md`, `fe_admin_cookie_integration.md`). Đây là một best practice tuyệt vời khi làm việc nhóm (FE-BE collaboration).

---

### ⚠️ Những điểm cần cải thiện (Areas for Improvement)

Dù cấu trúc đã rất tốt, vẫn có một vài điểm có thể tối ưu thêm để repo gọn gàng và chuẩn mực hơn:

*   **Sự thiếu nhất quán ở thư mục Repositories:**
    Trong cấu trúc chuẩn Repository Pattern, Database logic nên nằm ở `repository.py`, còn Business logic nằm ở `service.py`. 
    Tuy nhiên, trong repo hiện tại, chỉ có một số module (như `article`, `language`) là có folder `repositories/` hoặc file `repository.py`. Còn lại hầu hết các module khác (như `menu`, `staff`, `category`) DB queries đang bị viết thẳng vào `service.py`. Nên chuẩn hóa 1 pattern duy nhất (tốt nhất là đẩy hết query xuống layer Repository) cho toàn bộ project.
*   **Thư mục `scratch/` đang nằm trong root của project:**
    Thư mục này dường như chứa các script test tạm thời, crawl dữ liệu rác, dump sql (`check_db_logs.py`, `explore_html.py`, `test_ai_logs.py`). Việc đẩy thư mục này lên git (đặc biệt là nhánh `main`) làm repo bị rác. 
    👉 **Giải pháp:** Nên thêm `scratch/` vào `.gitignore`. Nếu đó là các script crawl dữ liệu cần thiết, hãy đưa nó vào thư mục `scripts/` hoặc `tools/`.
*   **Cấu trúc thư mục `schemas` và `routers` (Dạng file vs Dạng folder):**
    Có sự không đồng nhất: một số module dùng file `schemas.py` / `routers.py` (vd: `consultation`, `dashboard`), nhưng một số module lại dùng folder `schemas/` / `routers/` (vd: `article`, `category`). 
    👉 **Giải pháp:** Nếu module nhỏ, hãy dùng file `.py`. Nếu module lớn, dùng folder với `__init__.py`. Tuy nhiên, nên quy định rõ ràng khi nào thì split ra folder để code base nhìn đồng bộ hơn.
*   **Các module đang bị "phẳng" (Flat modules):**
    Trong `app/modules/` đang có quá nhiều folder cấp 1 (hơn 20 folder). 
    👉 **Giải pháp (Tùy chọn):** Khi dự án lớn hơn, bạn có thể gom nhóm các domain liên quan lại. Ví dụ:
    *   `app/modules/hr/`: chứa `staff`, `department`, `position`
    *   `app/modules/content/`: chứa `article`, `category`, `tag`, `media`
    *   `app/modules/system/`: chứa `auth`, `notification`, `health`, `dashboard`

### 💡 Tổng kết
**Chấm điểm cấu trúc: 9/10.** 
Bạn đang sở hữu một cấu trúc source code rất "Production-ready", dễ scale, dễ đọc cho người mới onboard. Chỉ cần dọn dẹp thư mục `scratch/` và nhất quán lại việc sử dụng Repository Pattern ở tất cả các module là cấu trúc này sẽ trở nên hoàn hảo.

---

## 3. Đề xuất Cấu trúc Chuẩn (Standard Project Structure)

Dưới đây là tài liệu mô tả chi tiết Cấu trúc thư mục chuẩn (Standard Project Structure) cho dự án FastAPI theo kiến trúc Modular Monolith, được tối ưu và chuẩn hoá từ dự án.

### 3.1. Tổng quan cấu trúc thư mục gốc (Root Directory)

```text
ktcn-backend-api/
├── .env.example                # File mẫu biến môi trường
├── .gitignore                  # Bỏ qua git (venv, __pycache__, scratch, .env, *.db)
├── Dockerfile                  # Docker build image cho backend
├── docker-compose.yml          # Chạy production/staging stack
├── docker-compose.dev.yml      # Chạy local dev (Postgres, Redis, OmniRoute, v.v.)
├── pyproject.toml              # Quản lý dependencies / linter / formatter (Poetry/Ruff/Black)
├── requirements.txt            # Danh sách thư viện Python
├── README.md                   # Giới thiệu & hướng dẫn cài đặt dự án
├── alembic.ini                 # Config của Alembic migration
│
├── app/                        # 🚀 SOURCE CODE CHÍNH CỦA ỨNG DỤNG
│   ├── main.py                 # FastAPI app entrypoint
│   ├── core/                   # Cấu hình lõi (Database, Security, Config, Logger, Middleware)
│   ├── common/                 # Base classes dùng chung (BaseModel, BaseRepo, Global Exceptions)
│   ├── shared/                 # External integrations & helpers (AI, Redis, S3/Media, SEO)
│   └── modules/                # Các domain nghiệp vụ (Auth, Article, Staff, Category...)
│
├── database/                   # Khởi tạo DB, raw schema & scripts data ban đầu
│   ├── schema/                 # Raw SQL schemas / DDL
│   └── scripts/                # Scripts seed dữ liệu khởi tạo cơ bản
│
├── migrations/                 # Alembic migrations (quản lý lịch sử schema)
│   ├── env.py
│   ├── script.py.mako
│   └── versions/               # Các migration files đánh theo revision ID
│
├── tests/                      # Kiểm thử tự động (Pytest)
│   ├── conftest.py             # Fixtures dùng chung (db test session, client mock, user mock)
│   ├── unit/                   # Unit tests (test service/logic độc lập)
│   └── integration/            # Integration / API endpoint tests
│
├── scripts/                    # Scripts phục vụ vận hành, crawlers, batch jobs chính thức
│   ├── crawlers/               # Các tool crawl dữ liệu từ portal cũ
│   └── data/                   # Data json/csv trung gian phục vụ import/export
│
└── docs/                       # Tài liệu thiết kế hệ thống, API guide cho Frontend
    ├── admin/                  # Specs & hướng dẫn cho FE Admin
    └── portal/                 # Specs & hướng dẫn cho FE Portal
```

### 3.2. Chi tiết cấu trúc bên trong thư mục `app/`

#### `app/core/` (Cấu hình lõi & Hạ tầng)
Chứa mọi thứ cần thiết để FastAPI khởi động và chạy hạ tầng:
```text
app/core/
├── config.py             # Quản lý Settings bằng pydantic-settings (.env)
├── database.py           # Async SQLAlchemy Engine, SessionLocal, get_db dependency
├── security.py           # Hash password (bcrypt), JWT Encode/Decode, API key validator
├── logger.py             # Logger format (loguru / logging structlog)
├── exceptions.py         # Global App Exceptions & Exception Handlers
└── middleware/           # Các custom middleware
    ├── rate_limit.py     # Giới hạn request rate (Redis-based)
    ├── security.py       # Security headers (CORS, CSP, XSS protection)
    └── visitor_counter.py
```

#### `app/common/` (Nền tảng trừu tượng hóa)
Chứa các class nền tảng để kế thừa, tránh lặp lại code:
```text
app/common/
├── models/
│   └── base.py           # BaseSQLAlchemyModel (id, created_at, updated_at, is_deleted)
├── repositories/
│   └── base.py           # BaseRepository (CRUD cơ bản: get, list, create, update, delete)
├── schemas/
│   └── base.py           # BasePydanticSchema, PaginatedResponse, APIResponse wrapper
└── enums/                # Các enum toàn cục (LanguageCode, Status, UserRole)
```

#### `app/shared/` (Các dịch vụ ngoài & Tiện ích)
Chứa các module độc lập không thuộc về một business domain cụ thể:
```text
app/shared/
├── redis.py              # Redis client connection & caching helpers
├── pagination.py         # Phân trang thống nhất (page, page_size, total_pages)
├── sort_order.py         # Helper sắp xếp thứ tự hiển thị
├── ai/                   # Tích hợp AI/LLM
│   ├── base.py
│   ├── config.py
│   ├── providers/        # OmniRoute, OpenAI, Claude, v.v.
│   └── service.py
├── media/                # Upload file, CDN, S3, MinIO client
└── seo/                  # SEO metadata generator, sitemap builder
```

### 3.3. Cấu trúc chuẩn của một Domain Module (`app/modules/<domain>/`)

Mỗi tính năng/nghiệp vụ nằm hoàn toàn trong folder riêng và tuân thủ mô hình **4 lớp (Layers)**:

```text
app/modules/article/
├── __init__.py
├── models.py             # ORM Entity: Article, ArticleTranslation, Tag
├── schemas/              # Pydantic schemas (hoặc file schemas.py nếu module đơn giản)
│   ├── __init__.py
│   ├── admin.py          # Schemas cho Admin API (Request/Response chi tiết)
│   └── portal.py         # Schemas cho Portal/Public API (Tối ưu cho hiển thị)
├── repository.py         # Tương tác trực tiếp DB (hoặc folder repositories/)
├── service.py            # Business Logic chính: validation nghiệp vụ, transactions
├── dependencies.py       # Router dependencies đặc thù (permissions, get_current_article)
└── routers/              # API Endpoints
    ├── __init__.py       # Gom router của module lại
    ├── admin.py          # /api/v1/admin/articles/...
    └── portal.py         # /api/v1/portal/articles/...
```

#### Quy tắc luồng dữ liệu (Data Flow Standard):
```
Client Request 
     ⬇️
Router (Validate Request qua Schema, gọi Dependency kiểm tra quyền)
     ⬇️
Service (Thực hiện nghiệp vụ: validate business rule, điều phối repo, trigger background task)
     ⬇️
Repository (Thực hiện câu lệnh SQL / Query / Filtering trên SQLAlchemy)
     ⬇️
Database (PostgreSQL)
```

> **Nguyên tắc vàng:** 
> - **Router** KHÔNG được truy vấn DB trực tiếp.
> - **Service** KHÔNG xử lý trực tiếp HTTP Request/Response (không nhận `Request`, không trả `HTTPException`).
> - **Repository** KHÔNG chứa logic nghiệp vụ (chỉ nhận tham số và trả về ORM Model).
