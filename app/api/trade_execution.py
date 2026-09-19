from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.auth.dependencies import get_current_user
from app.core.database import SessionLocal

from app.services.trade_execution_runtime_service import (
    TradeExecutionRuntimeService,
)

from app.services.paper_execution_service import (
    PaperExecutionService,
)


router = APIRouter(
    prefix="/api/trade-execution",
    tags=["Trade Execution"],
)


class TradeExecutionRequest(BaseModel):
    trade_candidate: dict[str, Any] = Field(
        ...,
        description=(
            "Canonical Trade Candidate produced by "
            "the signal pipeline."
        ),
    )

    risk_budget: float | None = Field(
        default=None,
        description=(
            "Explicit monetary risk budget for this "
            "execution attempt. No risk percentage is "
            "inferred by the system."
        ),
    )


@router.post("/prepare")
def prepare_trade_execution(
    request: TradeExecutionRequest,
    current_user=Depends(get_current_user),
):
    """
    Prepare an authenticated trade for execution.

    This endpoint does NOT submit a broker order.

    Pipeline:

        authenticated user
        -> portfolio risk state
        -> RiskEngine
        -> RiskBudgetService
        -> PositionSizingEngine
        -> OrderIntentService
        -> prepared result
    """

    db = SessionLocal()

    try:
        runtime = TradeExecutionRuntimeService(
            db=db,
            user_id=current_user.id,
            trade_candidate=request.trade_candidate,
            risk_budget=request.risk_budget,
        )

        result = runtime.prepare()

        return {
            "user_id": current_user.id,
            **result,
            "source": "TradeExecutionAPI",
        }

    finally:
        db.close()


@router.post("/paper-execute")
def paper_execute_trade(
    request: TradeExecutionRequest,
    current_user=Depends(get_current_user),
):
    """
    Prepare and execute a trade through the existing
    paper-execution infrastructure.

    This endpoint does NOT call the live broker
    execution boundary.

    Pipeline:

        authenticated user
        -> portfolio risk state
        -> RiskEngine
        -> RiskBudgetService
        -> PositionSizingEngine
        -> OrderIntentService
        -> PaperExecutionService
    """

    db = SessionLocal()

    try:
        runtime = TradeExecutionRuntimeService(
            db=db,
            user_id=current_user.id,
            trade_candidate=request.trade_candidate,
            risk_budget=request.risk_budget,
        )

        runtime_result = runtime.prepare()

        if not runtime_result.get(
            "ready_for_execution"
        ):
            return {
                "user_id": current_user.id,
                "execution_status": "REJECTED",
                "runtime": runtime_result,
                "paper_execution": None,
                "source": "TradeExecutionAPI",
            }

        execution = (
            runtime_result.get("execution")
            or {}
        )

        order_intent = execution.get(
            "order_intent"
        )

        if not order_intent:
            return {
                "user_id": current_user.id,
                "execution_status": "REJECTED",
                "runtime": runtime_result,
                "paper_execution": None,
                "reason": (
                    "Execution pipeline did not produce "
                    "an order intent."
                ),
                "source": "TradeExecutionAPI",
            }

        paper_executor = PaperExecutionService(
            db=db,
            user_id=current_user.id,
        )

        paper_result = paper_executor.execute(
            order_intent
        )

        return {
            "user_id": current_user.id,
            "execution_status": (
                "EXECUTED"
                if (
                    paper_result.get("status") == "PASS"
                    and paper_result.get("executed")
                    is True
                )
                else "REJECTED"
            ),
            "runtime": runtime_result,
            "paper_execution": paper_result,
            "source": "TradeExecutionAPI",
        }

    finally:
        db.close()