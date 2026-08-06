from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.auth.dependencies import get_current_user

from app.models.user import User
from app.models.portfolio import Portfolio

from app.services.market_price_service import get_live_price
from app.services.portfolio_service import calculate_portfolio_metrics

router = APIRouter(
    prefix="/api/dashboard",
    tags=["Dashboard"]
)


@router.get("/")
def dashboard(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):

    holdings = (
        db.query(Portfolio)
        .filter(
            Portfolio.user_id == current_user.id
        )
        .all()
    )

    portfolio = []

    total_investment = 0
    total_current = 0

    top_gainer = None
    top_loser = None

    highest = -999999999
    lowest = 999999999

    for holding in holdings:

        price = get_live_price(
            holding.symbol
        )

        holding.current_price = price

        db.commit()

        db.refresh(holding)

        metrics = calculate_portfolio_metrics(
            holding
        )

        portfolio.append(metrics)

        total_investment += metrics["investment"]
        total_current += metrics["current_value"]

        if metrics["profit_percent"] > highest:

            highest = metrics["profit_percent"]

            top_gainer = metrics

        if metrics["profit_percent"] < lowest:

            lowest = metrics["profit_percent"]

            top_loser = metrics

    total_profit = (
        total_current
        - total_investment
    )

    if total_investment == 0:

        overall_return = 0

    else:

        overall_return = (
            total_profit
            / total_investment
        ) * 100

    return {

        "summary": {

            "investment":
                round(total_investment, 2),

            "current_value":
                round(total_current, 2),

            "profit":
                round(total_profit, 2),

            "return_percent":
                round(overall_return, 2),

            "holdings":
                len(portfolio),

        },

        "top_gainer":
            top_gainer,

        "top_loser":
            top_loser,

        "portfolio":
            portfolio,

    }