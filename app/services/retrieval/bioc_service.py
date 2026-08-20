from typing import Optional

import logfire

from app.config import settings
from app.services.retrieval.pubmed_service import _get

# Section types not worth embedding/citing: title and abstract are already
# captured by the ESearch/EFetch metadata fields, and refs/ack/competing-
# interest blocks add noise without evidentiary value.
_SKIP_SECTIONS = {"TITLE", "ABSTRACT", "REF", "ACK_FUND", "COMP_INT", "AUTH_CONT", "SUPPL"}


def fetch_full_text_sections(pmid: str) -> Optional[list[dict]]:
    """
    Fetch PMC Open Access full text for a PMID via NCBI's BioC-PMC API.
    Returns a list of {"section": str, "text": str} (consecutive same-section
    passages merged), or None if the article isn't open access / has no usable
    full text. Never raises: any non-200, timeout, or malformed body degrades
    to None, same as the rest of the NCBI client (see pubmed_service._get).
    """
    url = f"{settings.NCBI_BIOC_BASE_URL}/BioC_json/{pmid}/unicode"

    with logfire.span("BioC full text fetch", pmid=pmid):
        response = _get(url, {})
        if response is None:
            return None

        try:
            data = response.json()
        except ValueError as e:
            logfire.warning(f"BioC response parsing failed for PMID {pmid}: {e}")
            return None

        # Top-level response is a list of BioCCollection objects (usually one).
        collections = data if isinstance(data, list) else [data]

        sections: list[dict] = []
        for collection in collections:
            for document in collection.get("documents", []):
                for passage in document.get("passages", []):
                    text = (passage.get("text") or "").strip()
                    if not text:
                        continue
                    infons = passage.get("infons") or {}
                    section = (infons.get("section_type") or infons.get("type") or "Body").upper()
                    if section in _SKIP_SECTIONS:
                        continue
                    if sections and sections[-1]["section"] == section:
                        sections[-1]["text"] += "\n" + text
                    else:
                        sections.append({"section": section, "text": text})

        if not sections:
            return None

        logfire.info(f"BioC full text found for PMID {pmid}: {len(sections)} section(s)")
        return sections
