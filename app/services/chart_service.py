import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def create_stock_chart(symbol: str, period: str = "6mo"):

    stock = yf.Ticker(symbol)

    history = stock.history(period=period)
    history["MA20"] = history["Close"].rolling(window=20).mean()
    history["MA50"] = history["Close"].rolling(window=50).mean()
    #print(history[["Close", "MA20", "MA50"]].tail(15))
    delta = history["Close"].diff()

    gain = delta.where(delta > 0, 0)

    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(14).mean()

    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss

    history["RSI"] = 100 - (100 / (1 + rs))

    history["EMA12"] = history["Close"].ewm(span=12, adjust=False).mean()

    history["EMA26"] = history["Close"].ewm(span=26, adjust=False).mean()

    history["MACD"] = history["EMA12"] - history["EMA26"]

    history["Signal"] = history["MACD"].ewm(span=9, adjust=False).mean()

    history["Histogram"] = history["MACD"] - history["Signal"]


    fig = make_subplots(
    rows=4,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.02,
    row_heights=[0.58, 0.12, 0.15, 0.15],
)

    fig.add_trace(
    go.Candlestick(
        x=history.index,
        open=history["Open"],
        high=history["High"],
        low=history["Low"],
        close=history["Close"],
        name="Candlestick"
    ),
    row = 1,
    col = 1,
)
    fig.add_trace(
    go.Scatter(
        x=history.index,
        y=history["MA20"],
        mode="lines",
        name="20-Day MA",
        line=dict(color="#22D3EE", width=3),
        connectgaps=True,
    ),
    row = 1,
    col = 1,
)

    fig.add_trace(
    go.Scatter(
        x=history.index,
        y=history["MA50"],
        mode="lines",
        name="50-Day MA",
        line=dict(color="#EAB308", width=2),
        connectgaps=True,
    ),
    row=1,
    col=1,
)
    fig.add_trace(
    go.Bar(
        x=history.index,
        y=history["Volume"],
        name="Volume",
        marker_color="lightblue",
    ),
    row=2,
    col=1,
)
    fig.add_trace(
    go.Scatter(
        x=history.index,
        y=history["RSI"],
        mode="lines",
        name="RSI",
        line=dict(color="magenta", width=2),
    ),
    row=3,
    col=1,
)
    fig.add_hline(
    y=70,
    line_dash="dash",
    line_color="red",
    row=3,
    col=1,
)

    fig.add_hline(
    y=30,
    line_dash="dash",
    line_color="green",
    row=3,
    col=1,
)
    fig.add_trace(
    go.Scatter(
        x=history.index,
        y=history["MACD"],
        mode="lines",
        name="MACD",
        line=dict(color="dodgerblue", width=2),
    ),
    row=4,
    col=1,
)
    fig.add_trace(
    go.Scatter(
        x=history.index,
        y=history["Signal"],
        mode="lines",
        name="Signal",
        line=dict(color="orange", width=2),
    ),
    row=4,
    col=1,
)
    fig.add_trace(
    go.Bar(
        x=history.index,
        y=history["Histogram"],
        name="Histogram",
        marker_color=[
            "green" if value >= 0 else "red"
            for value in history["Histogram"]
        ],
    ),
    row=4,
    col=1,
)
    fig.update_xaxes(
    showgrid=True,
    gridcolor="#253247",
    zeroline=False
)

    fig.update_yaxes(
    showgrid=True,
    gridcolor="#253247",
    zeroline=False
)

    fig.update_layout(
    title=None,

    template="plotly_dark",

    height=980,

    autosize=True,

    width=None,

    uirevision=True,

    hovermode="x unified",

    margin=dict(
        l=20,
        r=20,
        t=60,
        b=20
    ),

    paper_bgcolor="#111827",
    plot_bgcolor="#111827",

    font=dict(
        family="Inter",
        size=14,
        color="white"
    ),

    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.04,
        xanchor="center",
        x=0.5,
        font=dict(size=12)
    ),
)

    fig.update_yaxes(
    title_text="RSI",
    row=3,
    col=1,
)
    fig.update_yaxes(
    range=[0, 100],
    row=3,  
    col=1,
)   
    return fig.to_html(
    full_html=False,
    config={
        "responsive": True,
        "displaylogo": False,
    }
)