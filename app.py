# app_select_assess_prompt_strict.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Any
import os, json

# ---- LLM is mandatory (no fallback) ----
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY environment variable is required (no fallback).")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5")  # default to gpt-4o if not set

# LangChain + OpenAI
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser  # correct path in recent LangChain

app = FastAPI(title="SELECT Assessment & LLM Prompt (Strict Schema, LangChain, no fallback)")

# ====== Strict input models ======
class SelectItem(BaseModel):
    table: str
    target_type: str
    target_name: str
    used_fields: List[str]
    suggested_fields: List[str]
    suggested_statement: str

    @field_validator("used_fields", "suggested_fields")
    @classmethod
    def no_none_elems(cls, v: List[str]) -> List[str]:
        return [x for x in v if x is not None]

class Unit(BaseModel):
    pgm_name: str
    inc_name: str
    type: str
    name: str
    selects: List[SelectItem] = Field(default_factory=list)

# ====== Agentic-lite planner ======
def summarize_selects(unit: Unit) -> Dict[str, Any]:
    tables_count: Dict[str, int] = {}
    total = len(unit.selects)
    used_fields_union = set()
    suggested_fields_union = set()

    for s in unit.selects:
        tables_count[s.table] = tables_count.get(s.table, 0) + 1
        for f in s.used_fields:
            if f:
                used_fields_union.add(f.upper())
        for f in s.suggested_fields:
            if f:
                suggested_fields_union.add(f.upper())

    return {
        "program": unit.pgm_name,
        "include": unit.inc_name,
        "unit_type": unit.type,
        "unit_name": unit.name,
        "stats": {
            "total_selects": total,
            "tables_frequency": tables_count,
            "unique_used_fields": sorted(used_fields_union),
            "unique_suggested_fields": sorted(suggested_fields_union),
        }
    }

# ====== LangChain prompt & chain ======
SYSTEM_MSG = "You are a precise ABAP reviewer who outputs strict JSON only."

USER_TEMPLATE = """
You are assessing ABAP SELECT usage for an S4HANA System ( 7.4+ syntax). 
We provide structured entries under `selects` with:
- table, target_type, target_name
- used_fields (fields actually needed downstream)
- suggested_fields (what we believe the SELECT list should be)
- suggested_statement (a proposed non-* SELECT)

Your job:
1) Produce a concise human-readable **assessment** paragraph:
   - Summarize risks from SELECT * or overfetching vs. what `used_fields` indicate.
   - Note performance/memory/network/maintainability impacts.
   - Mention any mismatch between used_fields and suggested_fields.
   - Keep it factual and brief.

2) Produce an actionable **LLM remediation prompt** for later.
   The prompt must be:
   - To the point,concise and contain **no more than 5 numbered bullet points**
   - Reference metadata (program/include/unit).
   - Ask to replace SELECT * with explicit field list from `suggested_fields`.
   - Preserve behavior; S4HANA COmpatible ABAP only (abap 7.4+ Syntax).
   - Require output JSON with keys: original_code, remediated_code, changes[] (line/before/after/reason). 
   - If suggested_statement is provided, instruct using it as the base; if empty, request generation from suggested_fields.

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

# ====== Core LLM call ======
def llm_assess_and_prompt(unit: Unit) -> Dict[str, str]:
    plan = summarize_selects(unit)
    plan_json = json.dumps(plan, ensure_ascii=False, indent=2)
    selects_json = json.dumps([s.model_dump() for s in unit.selects], ensure_ascii=False, indent=2)

    try:
        return chain.invoke(
            {
                "pgm_name": unit.pgm_name,
                "inc_name": unit.inc_name,
                "unit_type": unit.type,
                "unit_name": unit.name,
                "plan_json": plan_json,
                "selects_json": selects_json,
            }
        )
    except Exception as e:
        # hard fail by design
        raise HTTPException(status_code=502, detail=f"LLM call failed: {e}")

# ====== API ======
@app.post("/assess-selects")
async def assess_selects(units: List[Unit]) -> List[Dict[str, Any]]:
    """
    Input: array of units with strict `selects` items.
    Output: same array, replacing `selects` with:
      - 'assessment' (string)
      - 'llm_prompt' (string)
    """
    out: List[Dict[str, Any]] = []
    for u in units:
        obj = u.model_dump()
        llm_out = llm_assess_and_prompt(u)
        obj["assessment"] = llm_out.get("assessment", "")
        obj["llm_prompt"] = llm_out.get("llm_prompt", "")
        obj.pop("selects", None)  # remove as requested
        out.append(obj)
    return out

@app.get("/health")
def health():
    return {"ok": True, "model": OPENAI_MODEL}
