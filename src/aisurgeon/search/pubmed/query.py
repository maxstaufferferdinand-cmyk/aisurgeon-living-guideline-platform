"""Deterministic PubMed query construction and validation."""

import hashlib
import re
from datetime import date

from aisurgeon.search.pubmed.models import PubMedQuery, SearchUnit

QUERY_BUILDER_VERSION = "pubmed_query_builder_v4"
HUMANS_FILTER = "NOT (animals[mh] NOT humans[mh])"
EVIDENCE_TYPE_FILTER = (
    '("Randomized Controlled Trial"[pt] OR "Meta-Analysis"[pt] OR "Systematic Review"[pt])'
)
EXCLUSION_FILTER = 'NOT ("Practice Guideline"[pt] OR "Guideline"[pt])'
FIELD_TERM_PATTERN = re.compile(
    r'(?P<prefix>^|[\s(])(?P<term>(?!"|AND\b|OR\b|NOT\b)[A-Za-z0-9][A-Za-z0-9*.-]*'
    r'(?:[ -][A-Za-z0-9*.-]+)*)(?P<field>\[(?:tiab|Title/Abstract|All Fields)\])',
    flags=re.IGNORECASE,
)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def sanitize_query_core(query: str) -> str:
    """Quote unquoted PubMed fielded phrases that NCBI parses as syntax errors."""

    def replace(match: re.Match[str]) -> str:
        term = match.group("term")
        if " " not in term and "-" not in term:
            return match.group(0)
        return f'{match.group("prefix")}"{term}"{match.group("field")}'

    return FIELD_TERM_PATTERN.sub(replace, query)


def validate_query_core(query: str) -> list[str]:
    errors: list[str] = []
    if not query.strip():
        errors.append("empty_query_core")
    depth = 0
    for character in query:
        depth += character == "("
        depth -= character == ")"
        if depth < 0:
            errors.append("unbalanced_parentheses")
            break
    if depth != 0 and "unbalanced_parentheses" not in errors:
        errors.append("unbalanced_parentheses")
    if re.search(r"\(\s*\)", query):
        errors.append("empty_boolean_group")
    if re.search(r"(^|\()\s*(AND|OR|NOT)\b|\b(AND|OR|NOT)\s*($|\))", query, re.I):
        errors.append("isolated_boolean_operator")
    if re.search(r"\b(AND|OR|NOT)\s+(AND|OR|NOT)\b", query, re.I):
        errors.append("adjacent_boolean_operators")
    if len(query) > 8000:
        errors.append("query_too_long")
    if re.search(
        r"\[\s*(?:pt|dp|date\s*-\s*publication)\s*\]|animals\[mh\]|humans\[mh\]|"
        r"\b(?:retmax|retstart|api_key)\b",
        query,
        re.IGNORECASE,
    ):
        errors.append("technical_filter_in_query_core")
    if re.search(
        r"\b(und|oder|bei|ohne|behandlung|patienten|patientinnen|erwachsene|kinder)\b",
        query,
        re.IGNORECASE,
    ):
        errors.append("possible_german_search_term")
    normalized_terms = [
        f"{term.casefold()}{field.casefold()}"
        for term, field in re.findall(r'"([^"]+)"\s*(\[[^\]]+\])?', query)
    ]
    if len(normalized_terms) - len(set(normalized_terms)) > 20:
        errors.append("obviously_redundant_terms")
    return list(dict.fromkeys(errors))


def validate_final_pubmed_query(query: str) -> list[str]:
    """Reject unsafe Boolean composition in a complete query or NCBI translation."""
    errors: list[str] = []
    if re.search(r"\b(?:AND|OR)\s+NOT\s*\(", query, re.IGNORECASE):
        errors.append("boolean_operator_before_negative_exclusion")
    if re.search(
        r"\bAND\s*\(\s*(?:\"animals\"\[MeSH Terms\]|animals\[mh\])\s+NOT\s+"
        r"(?:\"humans\"\[MeSH Terms\]|humans\[mh\])\s*\)",
        query,
        re.IGNORECASE,
    ):
        errors.append("animals_exclusion_used_as_positive_filter")
    return errors


def build_query(
    unit: SearchUnit,
    *,
    start_date: date,
    end_date: date,
    prompt_version: str,
    prompt_hash: str,
    model_config: dict[str, object],
    model_config_hash: str,
) -> PubMedQuery:
    core = sanitize_query_core((unit.query_core or "").strip())
    errors = validate_query_core(core)
    if errors:
        raise ValueError(f"Invalid query_core: {', '.join(errors)}")
    date_filter = (
        f'("{start_date:%Y/%m/%d}"[Date - Publication] : "{end_date:%Y/%m/%d}"[Date - Publication])'
    )
    final = (
        f"({core} AND {date_filter} AND {EVIDENCE_TYPE_FILTER}) {HUMANS_FILTER} {EXCLUSION_FILTER}"
    )
    final_errors = validate_final_pubmed_query(final)
    if final_errors:
        raise ValueError(f"Invalid final PubMed query: {', '.join(final_errors)}")
    query_id = f"{unit.source_id}_QUERY_{sha256_text(unit.search_unit_id + final)[:12]}"
    return PubMedQuery(
        source_id=unit.source_id,
        query_id=query_id,
        search_unit_id=unit.search_unit_id,
        linked_formal_item_ids=unit.linked_formal_item_ids,
        query_core=core,
        date_filter=date_filter,
        humans_filter=HUMANS_FILTER,
        evidence_type_filter=EVIDENCE_TYPE_FILTER,
        exclusion_filter=EXCLUSION_FILTER,
        final_pubmed_query=final,
        start_date=start_date.isoformat(),
        end_date=end_date.isoformat(),
        query_version=QUERY_BUILDER_VERSION,
        prompt_version=prompt_version,
        prompt_hash=prompt_hash,
        model_id=str(model_config["model_id"]),
        model_configuration=model_config,
        model_configuration_hash=model_config_hash,
        review_required=unit.review_required,
    )
