# 👗 FitLens Backend (FastAPI + Supabase)

A backend service for **FitLens**, a personal fashion recommendation app.  
Users upload a photo + select a style (`casual` / `traditional`), and the API returns outfit recommendations with links & prices.

---

## 🚀 1) Setup

### Clone & enter project

```bash
git clone <your-repo-url>
cd fitlens-backend
```

### Create virtual environment

```bash
python -m venv .venv
# Mac/Linux:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate
```

### Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Configure environment variables

### Copy the example file and fill in your Supabase details:

```bash
cp .env.example .env
```

### .env.example: -

```bash
SUPABASE_URL=https://your-project-url.supabase.co
SUPABASE_SERVICE_ROLE_KEY=your-service-role-key
Get these from Supabase → Settings → API.
```

#### Database Setup (Supabase)

Run this SQL in the Supabase SQL Editor to create tables:

```bash
create table if not exists products (
  id text primary key,
  title text,
  store text,
  url text,
  image text,
  category text check (category in ('casual','traditional')),
  brand text,
  tags jsonb
);

create table if not exists prices (
  product_id text references products(id),
  currency text,
  mrp int,
  price int,
  in_stock boolean default true,
  sizes text[],
  updated_at timestamptz default now()
);

create table if not exists clicks (
  id uuid default gen_random_uuid() primary key,
  product_id text,
  session_id text,
  ts bigint
);

create table if not exists likes (
  id uuid default gen_random_uuid() primary key,
  product_id text,
  session_id text,
  ts bigint
);

```

### Run the Backend

```bash
uvicorn src.app.main:app --reload --host 127.0.0.1 --port 8000
```

The API will be available at:
http://127.0.0.1:8000

Interactive API Docs:
http://127.0.0.1:8000/docs
