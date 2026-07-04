import yfinance as yf
import plotly.graph_objects as go


def create_stock_chart(symbol: str, period: str = "1mo"):

    stock = yf.Ticker(symbol)

    history = stock.history(period=period)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=history.index,
            y=history["Close"],
            mode="lines",
            name="Closing Price"
        )
    )

    fig.update_layout(
        title=f"{symbol.upper()} Stock Price",
        xaxis_title="Date",
        yaxis_title="Price",
        template="plotly_dark"
    )

    return fig.to_html(full_html=False)