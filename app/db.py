import os, psycopg2, psycopg2.extras

DSN = os.getenv("DB_DSN", "postgresql://rag:ragpwd@localhost:5432/ragdb")

def ro_conn():
    conn = psycopg2.connect(DSN)
    conn.set_session(readonly=True, autocommit=True)
    return conn

def rw_conn():
    return psycopg2.connect(DSN)

def fetch_all(sql, params=()):
    with ro_conn() as c, c.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql, params)
        return cur.fetchall()

def execute(sql, params=()):
    with rw_conn() as c, c.cursor() as cur:
        cur.execute(sql, params); c.commit()
