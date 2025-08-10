# app_select_assess_prompt_langchain.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import os, json

# ---- LLM is mandatory (no fallback) ----
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY environment variable is required (no fallback).")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5")

# LangChain + OpenAI
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser  # NOTE: correct import path

app = FastAPI(title="SELECT* Assessment & LLM Prompt (LangChain, no fallback)")

# ====== Input models (minimal & flexible) ======
class SelectItem(BaseModel):
    """
    A flexible schema for one SELECT analysis item.
    Send whatever fields you have; common examples are shown as optional.
    """
    table: Optional[str] = None
    used_fields: Optional[List[str]] = None       # fields actually used downstream
    star_used: Optional[bool] = None              # whether original code used SELECT * / SELECT SINGLE *
    where_fields: Optional[List[str]] = None      # fields used in WHERE
    into_target: Optional[str] = None             # var/table receiving the row(s)
    join_tables: Optional[List[str]] = None       # if joins exist
    snippet: Optional[str] = None                 # raw ABAP SELECT snippet
    line: Optional[int] = None                    # source line, if available
    notes: Optional[str] = None                   # free-form notes
    # any additional keys the caller provides will be accepted

class Unit(BaseModel):
    """
    Mirrors the earlier MATNR pattern: you can pass program/include metadata plus 'selects'.
    """
    pgm_name: Optional[str] = ""
    inc_name: Optional[str] = ""
    type: Optional[str] = ""          # e.g., "perform", "method", "raw_code"
    name: Optional[str] = ""          # unit name (FORM/METHOD/etc.)
    class_implementation: Optional[str] = ""
    start_line: Optional[int] = 0
    end_line: Optional[int] = 0
    code: Optional[str] = ""          # optional: raw ABAP block (helpful but not required)
    selects: Optional[List[SelectItem]] = Field(default=None)

# ====== Agentic planning-lite: summarize the 'selects' for the LLM ======
def summarize_selects(unit: Unit) -> Dict[str, Any]:
    selects = unit.selects or []
    total = len(selects)
    star_count = sum(1 for s in selects if (s.star_used is True))
    tables = {}
    fields_needed = set()
    for s in selects:
        if s.table:
            tables[s.table] = tables.get(s.table, 0) + 1
        if s.used_fields:
            for f in s.used_fields:
                fields_needed.add(f)
    return {
        "program": unit.pgm_name,
        "include": unit.inc_name,
        "unit_type": unit.type,
        "unit_name": unit.name,
        "range": {"start_line": unit.start_line or 0, "end_line": unit.end_line or 0},
        "stats": {
            "total_selects": total,
            "star_selects": star_count,
            "tables_frequency": tables,
            "unique_used_fields_count": len(fields_needed),
        }
    }

# ====== LangChain prompt & chain ======
SYSTEM_MSG = "You are a precise ABAP remediation planner that outputs strict JSON only."

USER_TEMPLATE = """
You are a senior ABAP reviewer and modernization planner.

We are assessing **SELECT usage**, especially SELECT * (or SELECT SINGLE *) and overfetching.
Your job for this unit is:
1) Create a concise, human-readable **assessment** paragraph for a report, based ONLY on the provided `selects` entries.
   - Highlight risks like SELECT * overfetch, unused columns, missing explicit field lists, unnecessary ORDER BY, etc.
   - Mention why it matters (performance, memory, network, and maintainability).
   - Keep it factual and brief.
2) Produce a **remediation LLM prompt** to be used later. The prompt must:
   - Reference the unit metadata (program/include/unit/lines).
   - Use only ECC-safe syntax (no 7.4+ features).
   - Instruct the LLM to replace `SELECT *` with an explicit field list derived from `used_fields`.
   - Preserve behavior; do not change business logic.
   - Require output JSON with: original_code, remediated_code, changes[] (line, before, after, reason).
   - If `used_fields` is empty or missing, instruct the LLM to propose a safe minimal set or mark as "needs-review".

Return ONLY strict JSON with keys:
{{
  "assessment": "<concise assessment>",
  "llm_prompt": "<prompt to use later>"
}}

Unit metadata:
- Program: {pgm_name}
- Include: {inc_name}
- Unit type: {unit_type}
- Unit name: {unit_name}
- Start line: {start_line}
- End line: {end_line}

ABAP code (optional; may be empty or truncated):
{code}
Planning summary (agentic):
{plan_json}

selects (JSON):
{selects_json}
""".strip()

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_MSG),
        ("user", USER_TEMPLATE),
    ]
)

llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0.0)
parser = JsonOutputParser()
chain = prompt | llm | parser

# ====== Core LLM call via LangChain ======
def llm_assess_and_prompt_for_selects(unit: Unit) -> Dict[str, str]:
    selects_json = json.dumps([s.model_dump() for s in (unit.selects or [])], ensure_ascii=False, indent=2)

    # truncate huge code blocks to save tokens
    code = unit.code or ""
    MAX = 20000
    if len(code) > MAX:
        code = code[:MAX] + "\n*TRUNCATED*"

    plan = summarize_selects(unit)
    plan_json = json.dumps(plan, ensure_ascii=False, indent=2)

    try:
        return chain.invoke(
            {
                "pgm_name": unit.pgm_name or "",
                "inc_name": unit.inc_name or "",
                "unit_type": unit.type or "",
                "unit_name": unit.name or "",
                "start_line": unit.start_line or 0,
                "end_line": unit.end_line or 0,
                "code": code,
                "plan_json": plan_json,
                "selects_json": selects_json,
            }
        )
    except Exception as e:
        # hard fail (no fallback)
        raise HTTPException(status_code=502, detail=f"LLM call failed: {e}")

# ====== API ======
@app.post("/assess-selects")
def assess_selects(units: List[Unit]) -> List[Dict[str, Any]]:
    """
    Input: array of units each potentially containing `selects` (list of SelectItem).
    Output: same array, but replacing `selects` with:
      - 'assessment' (string)
      - 'llm_prompt' (string)
    """
    out: List[Dict[str, Any]] = []
    for u in units:
        obj = u.model_dump()
        llm_out = llm_assess_and_prompt_for_selects(u)
        obj["assessment"] = llm_out.get("assessment", "")
        obj["llm_prompt"] = llm_out.get("llm_prompt", "")
        obj.pop("selects", None)  # remove as requested
        out.append(obj)
    return out

@app.get("/health")
def health():
    return {"ok": True, "model": OPENAI_MODEL}
