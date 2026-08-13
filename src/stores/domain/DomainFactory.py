from src.stores.domain.providers.HospitalityDomain import (
    HospitalityDomain
)


class DomainFactory:

    @staticmethod
    def create(domain: str):

        domains = {
            "hospitality": HospitalityDomain,
        }

        domain = domain.lower().strip()

        if domain not in domains:
            raise ValueError(
                f"Unsupported domain: {domain}"
            )

        return domains[domain]()