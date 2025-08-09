from supabase import create_client, Client
from .config import SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY

def get_client() -> Client:
    # Single client: full access via service role (backend-only)
    return create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
