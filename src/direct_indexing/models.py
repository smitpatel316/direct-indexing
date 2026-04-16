"""
SQLAlchemy models for Direct Indexing persistence.

Tables:
- positions: current holdings with lot-level cost basis
- orders: complete order history
- tlh_ledger: TLH harvest records + wash sale tracking
- rebalance_log: rebalance cycle performance

Uses SQLite for local persistence. On startup, reconciles local state
with Alpaca's reported positions.
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Optional

from sqlalchemy import (
    create_engine, Column, String, Float, Integer, Date, DateTime,
    Enum as SAEnum, Index, UniqueConstraint,
)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import declarative_base, Session, sessionmaker
from sqlalchemy.sql import func

from .config import get_config


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

Base = declarative_base()
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class Position(Base):
    """Lot-level position record."""
    __tablename__ = "positions"

    lot_id = Column(String(64), primary_key=True)
    ticker = Column(String(16), nullable=False, index=True)
    shares = Column(Float, nullable=False)
    cost_basis_per_share = Column(Float, nullable=False)
    acquisition_date = Column(Date, nullable=False)
    # Current state
    current_shares = Column(Float, nullable=False)  # may differ after partial sells
    # Metadata
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        Index("ix_positions_ticker_lot", "ticker", "lot_id"),
    )

    def __repr__(self) -> str:
        return (
            f"<Position lot={self.lot_id} ticker={self.ticker} "
            f"shares={self.shares} avg=${self.cost_basis_per_share:.2f}>"
        )

    @property
    def total_cost(self) -> float:
        return self.shares * self.cost_basis_per_share


class Order(Base):
    """Complete order history."""
    __tablename__ = "orders"

    order_id = Column(String(64), primary_key=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    ticker = Column(String(16), nullable=False, index=True)
    side = Column(String(8), nullable=False)  # buy | sell
    qty = Column(Float, nullable=False)
    fill_price = Column(Float, nullable=True)
    order_type = Column(String(16), nullable=False)  # market | limit | stop
    status = Column(String(16), nullable=False)  # submitted | filled | cancelled | rejected
    reason = Column(String(32), nullable=False, default="")  # rebalance | tlh_sell | tlh_buy | reconstitution
    created_at = Column(DateTime, server_default=func.now())

    __table_args__ = (
        Index("ix_orders_ticker_timestamp", "ticker", "timestamp"),
        Index("ix_orders_timestamp", "timestamp"),
    )

    def __repr__(self) -> str:
        return (
            f"<Order {self.order_id} {self.side.upper()} {self.qty} {self.ticker} "
            f"@ ${self.fill_price or 0:.2f} ({self.status})>"
        )


class TLHLedger(Base):
    """TLH harvest record with wash sale tracking."""
    __tablename__ = "tlh_ledger"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(16), nullable=False, index=True)
    sell_date = Column(Date, nullable=False, index=True)
    loss_amount = Column(Float, nullable=False)  # positive = loss harvested
    substitute_ticker = Column(String(16), nullable=True)
    eligible_rebuy_date = Column(Date, nullable=False, index=True)  # sell_date + 31 days
    is_active = Column(Integer, nullable=False, default=1)  # 1=active, 0=closed
    rebuy_ticker = Column(String(16), nullable=True)  # what we bought back (may differ from original)
    rebuy_date = Column(Date, nullable=True)
    created_at = Column(DateTime, server_default=func.now())

    __table_args__ = (
        Index("ix_tlh_ticker_sell", "ticker", "sell_date"),
        Index("ix_tlh_eligible_rebuy", "eligible_rebuy_date", "is_active"),
    )

    def __repr__(self) -> str:
        return (
            f"<TLHLedger {self.ticker} sell={self.sell_date} "
            f"loss=${self.loss_amount:.2f} eligible={self.eligible_rebuy_date}>"
        )

    def is_restricted(self, as_of: date) -> bool:
        """Check if ticker is still in wash sale window."""
        return self.is_active and as_of < self.eligible_rebuy_date


class RebalanceLog(Base):
    """Rebalance cycle performance log."""
    __tablename__ = "rebalance_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(Date, nullable=False, unique=True, index=True)
    pre_tracking_error = Column(Float, nullable=True)  # in percent
    post_tracking_error = Column(Float, nullable=True)  # in percent
    num_trades = Column(Integer, nullable=False, default=0)
    num_tlh_trades = Column(Integer, nullable=False, default=0)
    total_tlh_harvested = Column(Float, nullable=False, default=0.0)
    total_trade_value = Column(Float, nullable=False, default=0.0)  # dollar value traded
    portfolio_value = Column(Float, nullable=False)
    reason = Column(String(32), nullable=False, default="scheduled")
    created_at = Column(DateTime, server_default=func.now())

    __table_args__ = (
        Index("ix_rebal_date", "date"),
    )

    def __repr__(self) -> str:
        return (
            f"<RebalanceLog {self.date} trades={self.num_trades} "
            f"tlh=${self.total_tlh_harvested:.2f} pv=${self.portfolio_value:.2f}>"
        )


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

class Database:
    """
    SQLite database manager for direct indexing.

    Handles:
    - Session management
    - Schema creation (auto-migrate)
    - Startup reconciliation with Alpaca
    """

    _instance: Optional[Database] = None

    def __init__(self, db_path: Optional[Path] = None):
        if db_path is None:
            config = get_config()
            data_dir = Path(config.data_dir) if config.data_dir else Path("data")
            db_path = data_dir / "direct_indexing.db"
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._engine: Optional[Engine] = None
        self._session_factory: Optional[sessionmaker] = None

    @property
    def engine(self) -> Engine:
        if self._engine is None:
            self._engine = create_engine(
                f"sqlite:///{self.db_path}",
                echo=False,
                connect_args={"check_same_thread": False},
            )
        return self._engine

    @property
    def session(self) -> Session:
        if self._session_factory is None:
            self._session_factory = sessionmaker(bind=self.engine)
        return self._session_factory()

    def init(self) -> None:
        """Create all tables (auto-migrate)."""
        Base.metadata.create_all(self.engine)
        logger.info(f"Database initialized at {self.db_path}")

    # -------------------------------------------------------------------------
    # Reconciliation
    # -------------------------------------------------------------------------

    def reconcile_with_alpaca(
        self,
        alpaca_client,
        prices: dict[str, float],
    ) -> dict:
        """
        Reconcile local position state with Alpaca's reported positions.

        On startup:
        1. Load local lots from DB
        2. Fetch Alpaca positions
        3. For each Alpaca position:
           - If we have matching lots: update current_shares
           - If we don't have it: create new lot (Bootstrap)
           - If we have it but Alpaca doesn't: lot is fully sold (mark closed)
        4. Log discrepancies
        """
        from .alpaca_client import AlpacaClient

        session = self.session
        changes = {"new_lots": 0, "updated_lots": 0, "closed_lots": 0, "discrepancies": []}

        # Fetch Alpaca positions
        alpaca_positions = alpaca_client.get_positions()
        alpaca_by_ticker = {p.symbol: p for p in alpaca_positions}

        # Load our lots
        local_lots = session.query(Position).filter(Position.current_shares > 0).all()
        local_by_ticker: dict[str, list[Position]] = {}
        for lot in local_lots:
            local_by_ticker.setdefault(lot.ticker, []).append(lot)

        # Check each Alpaca position
        for ticker, alpaca_pos in alpaca_by_ticker.items():
            local_lots = local_by_ticker.get(ticker, [])

            if not local_lots:
                # New position not in our DB — create bootstrap lot
                new_lot = Position(
                    lot_id=f"bootstrap-{ticker}-{date.today().isoformat()}",
                    ticker=ticker,
                    shares=float(alpaca_pos.qty),
                    cost_basis_per_share=float(alpaca_pos.avg_entry_price),
                    acquisition_date=date.today(),
                    current_shares=float(alpaca_pos.qty),
                )
                session.add(new_lot)
                changes["new_lots"] += 1
                logger.info(f"New bootstrap lot created: {ticker} {alpaca_pos.qty} shares")
                continue

            # Update existing lot's current_shares
            total_local = sum(lot.shares for lot in local_lots)
            if abs(total_local - float(alpaca_pos.qty)) > 0.001:
                changes["discrepancies"].append({
                    "ticker": ticker,
                    "alpaca_qty": float(alpaca_pos.qty),
                    "local_qty": total_local,
                })
                # Reconcile: scale our lots proportionally
                scale = float(alpaca_pos.qty) / total_local
                for lot in local_lots:
                    lot.current_shares = lot.shares * scale
                    changes["updated_lots"] += 1

        # Check for positions in our DB but not in Alpaca (sold/closed)
        for ticker, lots in local_by_ticker.items():
            if ticker not in alpaca_by_ticker:
                for lot in lots:
                    if lot.current_shares > 0:
                        lot.current_shares = 0
                        changes["closed_lots"] += 1
                        logger.info(f"Lot closed (not in Alpaca): {ticker} {lot.lot_id}")

        session.commit()
        return changes

    # -------------------------------------------------------------------------
    # Position CRUD
    # -------------------------------------------------------------------------

    def upsert_lot(self, lot: Position) -> None:
        session = self.session
        session.merge(lot)
        session.commit()

    def get_lots_by_ticker(self, ticker: str) -> list[Position]:
        session = self.session
        return (
            session.query(Position)
            .filter(Position.ticker == ticker, Position.current_shares > 0)
            .order_by(Position.acquisition_date)
            .all()
        )

    def get_all_open_lots(self) -> list[Position]:
        session = self.session
        return (
            session.query(Position)
            .filter(Position.current_shares > 0)
            .all()
        )

    # -------------------------------------------------------------------------
    # Order CRUD
    # -------------------------------------------------------------------------

    def add_order(self, order: Order) -> None:
        session = self.session
        session.add(order)
        session.commit()

    def get_orders(
        self,
        since: Optional[datetime] = None,
        ticker: Optional[str] = None,
        limit: int = 1000,
    ) -> list[Order]:
        session = self.session
        q = session.query(Order).order_by(Order.timestamp.desc()).limit(limit)
        if since:
            q = q.filter(Order.timestamp >= since)
        if ticker:
            q = q.filter(Order.ticker == ticker)
        return q.all()

    # -------------------------------------------------------------------------
    # TLH Ledger
    # -------------------------------------------------------------------------

    def add_tlh_entry(self, entry: TLHLedger) -> None:
        session = self.session
        session.add(entry)
        session.commit()

    def get_active_tlh(self, as_of: Optional[date] = None) -> list[TLHLedger]:
        session = self.session
        if as_of is None:
            as_of = date.today()
        return (
            session.query(TLHLedger)
            .filter(
                TLHLedger.is_active == 1,
                TLHLedger.eligible_rebuy_date > as_of,
            )
            .all()
        )

    def get_tlh_by_ticker(self, ticker: str) -> list[TLHLedger]:
        session = self.session
        return (
            session.query(TLHLedger)
            .filter(TLHLedger.ticker == ticker)
            .order_by(TLHLedger.sell_date.desc())
            .all()
        )

    def close_tlh_entry(self, entry_id: int, rebuy_ticker: str, rebuy_date: date) -> None:
        session = self.session
        entry = session.query(TLHLedger).filter(TLHLedger.id == entry_id).first()
        if entry:
            entry.is_active = 0
            entry.rebuy_ticker = rebuy_ticker
            entry.rebuy_date = rebuy_date
            session.commit()

    def get_total_tlh_harvested(self) -> float:
        session = self.session
        result = session.query(func.sum(TLHLedger.loss_amount)).scalar()
        return float(result or 0.0)

    # -------------------------------------------------------------------------
    # Rebalance Log
    # -------------------------------------------------------------------------

    def add_rebalance_log(self, log: RebalanceLog) -> None:
        session = self.session
        session.add(log)
        session.commit()

    def get_rebalance_logs(self, since: Optional[date] = None) -> list[RebalanceLog]:
        session = self.session
        q = session.query(RebalanceLog).order_by(RebalanceLog.date.desc())
        if since:
            q = q.filter(RebalanceLog.date >= since)
        return q.all()

    def get_last_rebalance_date(self) -> Optional[date]:
        session = self.session
        result = (
            session.query(RebalanceLog.date)
            .order_by(RebalanceLog.date.desc())
            .first()
        )
        return result[0] if result else None

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------

    def close(self) -> None:
        if self._engine:
            self._engine.dispose()


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

def get_database(db_path: Optional[Path] = None) -> Database:
    """Get the singleton Database instance."""
    if Database._instance is None:
        Database._instance = Database(db_path)
        Database._instance.init()
    return Database._instance
