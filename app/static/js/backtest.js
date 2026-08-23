document
.getElementById("runBacktest")
.addEventListener("click", async () => {

    const symbol =
        document
        .getElementById("symbol")
        .value;

    const brokerage =
    document
    .getElementById("brokerage")
    .value;

const slippage =
    document
    .getElementById("slippage")
    .value;

const capital =
    document
    .getElementById("capital")
    .value;

const strategy =
    document
    .getElementById("strategy")
    .value;

    const response =
        await fetch(

`/api/backtest/?symbol=${symbol}&strategy=${strategy}&brokerage=${brokerage}&slippage=${slippage}&capital=${capital}`
);

    const data =
        await response.json();

    document.getElementById(
        "totalTrades"
    ).innerHTML =
        data.total_trades;

    document.getElementById(
        "winningTrades"
    ).innerHTML =
        data.winning_trades;

    document.getElementById(
        "winRate"
    ).innerHTML =
        data.win_rate + "%";

    const profit =
        document.getElementById(
            "netProfit"
        );

    profit.innerHTML =
        "₹" +
        data.net_profit;

    profit.style.color =
        data.net_profit >= 0
            ? "#22c55e"
            : "#ef4444";

    document.getElementById(
    "maxDrawdown"
).innerHTML =
    data.max_drawdown + "%";

document.getElementById(
    "peakCapital"
).innerHTML =
    "₹" + data.peak_capital;

document.getElementById(
    "sharpeRatio"
).innerHTML =
    data.sharpe_ratio;

document.getElementById(
    "profitFactor"
).innerHTML =
    data.profit_factor;

document.getElementById(
    "holdingDays"
).innerHTML =
    data.average_holding_days + " Days";

document.getElementById("sharpeRatio").style.color =
    data.sharpe_ratio >= 1
        ? "#22c55e"
        : "#f59e0b";

document.getElementById("profitFactor").style.color =
    data.profit_factor >= 1
        ? "#22c55e"
        : "#ef4444";

const drawdown =
    document.getElementById(
        "maxDrawdown"
    );

drawdown.style.color =
    "#ef4444";

    const table =
        document.getElementById(
            "tradeTable"
        );

    table.innerHTML = "";

       data.trades.forEach(trade => {

        const color =
            trade.profit >= 0
                ? "#22c55e"
                : "#ef4444";

        table.innerHTML += `

<tr>

<td>${new Date(trade.buy_date).toLocaleDateString()}</td>

<td>${new Date(trade.sell_date).toLocaleDateString()}</td>

<td>${trade.buy_price}</td>

<td>${trade.sell_price}</td>

<td>${trade.shares}</td>

<td class="${
    trade.profit >= 0
        ? "profit"
        : "loss"
}">
    ₹${trade.profit}
</td>

<td>

${trade.return_percent}%

</td>

</tr>

`;

    });

    // ================================
    // Equity Curve Chart
    // ================================

    console.log(data);

console.log(data.equity_curve);

    const labels =
        data.equity_curve.map(
            item => item.date
        );

    const equityData =
    data.equity_curve.map(
        item => item.capital
    );

    const ctx =
        document
        .getElementById("equityChart")
        .getContext("2d");

    const oldChart =
    Chart.getChart("equityChart");

if (oldChart) {
    oldChart.destroy();
}

    window.equityChart = new Chart(ctx, {

        type: "line",

        data: {

            labels: labels,

            datasets: [

                {

                    label: "Portfolio Value",

                    data: equityData,

                    borderColor: "#22c55e",

                    backgroundColor: "rgba(34,197,94,0.15)",

                    fill: true,

                    tension: 0.3

                }

            ]

        },

        options: {

            responsive: true,

            plugins: {

                legend: {

                    display: true

                }

            },

            scales: {

                y: {

                    beginAtZero: false

                }

            }

        }

    });

});