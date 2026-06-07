import json
import os
import sqlite3
from datetime import datetime
from typing import Dict, Optional
from dotenv import load_dotenv

load_dotenv()

DB_PATH   = "logs/observability.db"
JSONL_PATH = "logs/calls.jsonl"


class ObservabilityLogger:

    def __init__(self):
        os.makedirs("logs", exist_ok=True)
        self._init_db()

    # ─────────────────────────────────────
    # DB SETUP
    # ─────────────────────────────────────

    def _init_db(self):
        """Create SQLite tables if not exists"""
        conn = self._conn()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS calls (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp     TEXT,
                model         TEXT,
                adapter_type  TEXT,
                category      TEXT,
                prompt_id     TEXT,
                prompt        TEXT,
                response      TEXT,
                tokens_input  INTEGER,
                tokens_output INTEGER,
                latency_ms    REAL,
                cost_usd      REAL,
                score         REAL,
                is_harmful    INTEGER
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id   TEXT,
                timestamp    TEXT,
                adapter_type TEXT,
                turn_count   INTEGER,
                total_tokens INTEGER,
                total_cost   REAL,
                total_latency_ms REAL
            )
        """)
        conn.commit()
        conn.close()

    def _conn(self):
        return sqlite3.connect(DB_PATH)

    # ─────────────────────────────────────
    # LOG A SINGLE LLM CALL
    # ─────────────────────────────────────

    def log_call(
        self,
        adapter_type:  str,
        prompt:        str,
        response:      str,
        tokens_input:  int,
        tokens_output: int,
        latency_ms:    float,
        cost_usd:      float,
        model_name:    str,
        category:      str  = "conversation",
        prompt_id:     str  = "",
        score:         float = None,
        is_harmful:    bool  = False,
    ):
        timestamp = datetime.utcnow().isoformat()

        # Save to SQLite
        conn = self._conn()
        conn.execute("""
            INSERT INTO calls (
                timestamp, model, adapter_type, category,
                prompt_id, prompt, response,
                tokens_input, tokens_output,
                latency_ms, cost_usd, score, is_harmful
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            timestamp, model_name, adapter_type, category,
            prompt_id, prompt[:500], response[:500],
            tokens_input, tokens_output,
            latency_ms, cost_usd,
            score, int(is_harmful)
        ))
        conn.commit()
        conn.close()

        # Save to JSONL
        record = {
            "timestamp":     timestamp,
            "model":         model_name,
            "adapter_type":  adapter_type,
            "category":      category,
            "prompt_id":     prompt_id,
            "prompt":        prompt[:500],
            "response":      response[:500],
            "tokens_input":  tokens_input,
            "tokens_output": tokens_output,
            "latency_ms":    latency_ms,
            "cost_usd":      cost_usd,
            "score":         score,
            "is_harmful":    is_harmful,
        }

        with open(JSONL_PATH, "a") as f:
            f.write(json.dumps(record) + "\n")

    # ─────────────────────────────────────
    # LOG A FULL CONVERSATION SESSION
    # ─────────────────────────────────────

    def log_session(
        self,
        session_id:       str,
        adapter_type:     str,
        turn_count:       int,
        total_tokens:     int,
        total_cost:       float,
        total_latency_ms: float,
    ):
        timestamp = datetime.utcnow().isoformat()
        conn = self._conn()
        conn.execute("""
            INSERT INTO sessions (
                session_id, timestamp, adapter_type,
                turn_count, total_tokens,
                total_cost, total_latency_ms
            ) VALUES (?,?,?,?,?,?,?)
        """, (
            session_id, timestamp, adapter_type,
            turn_count, total_tokens,
            total_cost, total_latency_ms
        ))
        conn.commit()
        conn.close()

    # ─────────────────────────────────────
    # QUERY / STATS
    # ─────────────────────────────────────

    def get_stats(self) -> Dict:
        """Summary statistics across all logged calls"""
        conn = self._conn()
        cursor = conn.cursor()

        stats = {}

        for adapter in ["hf", "groq"]:
            cursor.execute("""
                SELECT
                    COUNT(*)                    as total_calls,
                    AVG(latency_ms)             as avg_latency,
                    SUM(cost_usd)               as total_cost,
                    SUM(tokens_input + tokens_output) as total_tokens,
                    AVG(score)                  as avg_score,
                    SUM(is_harmful)             as harmful_count
                FROM calls
                WHERE adapter_type = ?
            """, (adapter,))

            row = cursor.fetchone()
            if row:
                stats[adapter] = {
                    "total_calls":   row[0],
                    "avg_latency_ms": round(row[1] or 0, 2),
                    "total_cost_usd": round(row[2] or 0, 6),
                    "total_tokens":   row[3] or 0,
                    "avg_score":      round(row[4] or 0, 2),
                    "harmful_blocked": row[5] or 0,
                }

        conn.close()
        return stats

    def get_cost_latency_table(self) -> list:
        """Cost + latency per model — for report"""
        conn   = self._conn()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT
                model,
                adapter_type,
                COUNT(*)            as calls,
                AVG(latency_ms)     as avg_latency,
                SUM(cost_usd)       as total_cost,
                AVG(cost_usd)       as avg_cost_per_call
            FROM calls
            GROUP BY model, adapter_type
        """)

        rows = cursor.fetchall()
        conn.close()

        return [
            {
                "model":             r[0],
                "adapter":           r[1],
                "calls":             r[2],
                "avg_latency_ms":    round(r[3] or 0, 2),
                "total_cost_usd":    round(r[4] or 0, 6),
                "avg_cost_per_call": round(r[5] or 0, 8),
            }
            for r in rows
        ]

    def print_stats(self):
        stats = self.get_stats()
        print("\n=== OBSERVABILITY STATS ===")
        for model, s in stats.items():
            print(f"\n  {model.upper()}:")
            for k, v in s.items():
                print(f"    {k:<22} → {v}")

        print("\n=== COST + LATENCY TABLE ===")
        table = self.get_cost_latency_table()
        for row in table:
            print(
                f"  {row['adapter']:<6} | "
                f"calls={row['calls']} | "
                f"avg_latency={row['avg_latency_ms']}ms | "
                f"avg_cost=${row['avg_cost_per_call']}"
            )