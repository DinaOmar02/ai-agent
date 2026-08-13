from abc import ABC, abstractmethod


class DomainInterface(ABC):

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        pass

    @property
    @abstractmethod
    def tools(self) -> list:
        pass

    @property
    @abstractmethod
    def knowledge_base(self) -> str:
        pass