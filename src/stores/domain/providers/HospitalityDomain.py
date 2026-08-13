from src.stores.domain.DomainInterface import DomainInterface

from src.generation.prompts import SYSTEM_PROMPT
from src.analysis.review_tools import analyze_reviews
from src.retrieval.rag_tool import create_search_tool
from src.config.enums import KnowledgeBaseCollection

class HospitalityDomain(DomainInterface):

    @property
    def system_prompt(self) -> str:
        return SYSTEM_PROMPT

    @property
    def knowledge_base(self) -> str:
        return KnowledgeBaseCollection.HOSPITALITY.value

    @property
    def tools(self) -> list:
        return [
            analyze_reviews,
            create_search_tool(self.knowledge_base),
        ]