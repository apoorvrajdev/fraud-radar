"""Repository for Merchant queries."""
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.merchant import Merchant
from app.repositories.base import BaseRepository


class MerchantRepository(BaseRepository[Merchant]):
    """Data access methods for Merchant entities."""

    model = Merchant

    def list_by_category(
        self, db: Session, category: str, *, limit: int = 100
    ) -> list[Merchant]:
        """List merchants in a given category."""
        stmt = (
            select(Merchant)
            .where(Merchant.category == category)
            .limit(limit)
        )
        return list(db.execute(stmt).scalars().all())


merchant_repository = MerchantRepository()
