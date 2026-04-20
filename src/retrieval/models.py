from dataclasses import dataclass, field


@dataclass
class SearchResult:
    chunk_id: str
    content: str
    source: str
    doc_id: str
    score: float
    metadata: dict = field(default_factory=dict)
