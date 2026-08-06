let allocationChart = null;
let growthChart = null;

async function loadDashboard() {

    const token = localStorage.getItem("access_token");

    const headers = {
        Authorization: "Bearer " + token
    };

    // Dashboard

    const response = await fetch(
    "/api/paper/dashboard",
    { headers }
);

if (!response.ok) {

    if (response.status === 401) {

        alert("Your session has expired. Please login again.");

        window.location.href = "/login";

        return;

    }

    throw new Error("Failed to load dashboard");

}

const dashboard = await response.json();

    document.getElementById("cash-balance").innerHTML =
        "₹ " + dashboard.cash_balance.toFixed(2);

    document.getElementById("portfolio-value").innerHTML =
        "₹ " + dashboard.portfolio_value.toFixed(2);

    document.getElementById("total-assets").innerHTML =
        "₹ " + dashboard.total_value.toFixed(2);

    const profitLossElement =
    document.getElementById("profit-loss");

profitLossElement.innerHTML =
    "₹ " + dashboard.profit_loss.toFixed(2);

profitLossElement.style.color =
    dashboard.profit_loss >= 0
        ? "#22c55e"
        : "#ef4444";

// ===========================
// Analytics
// ===========================

const analytics = await fetch(
    "/api/paper/analytics",
    { headers }
).then(r => r.json());

document.getElementById("best-stock").innerHTML =
    analytics.best_stock ?? "-";

document.getElementById("worst-stock").innerHTML =
    analytics.worst_stock ?? "-";

const totalProfit =
    document.getElementById("total-profit");

totalProfit.innerHTML =
    "₹" +
    analytics.total_profit.toLocaleString(
        "en-IN",
        {
            minimumFractionDigits: 2,
            maximumFractionDigits: 2
        }
    );

totalProfit.style.color =
    analytics.total_profit >= 0
        ? "#22c55e"
        : "#ef4444";

document.getElementById("last-updated").innerHTML =
    new Date().toLocaleTimeString(
        "en-IN",
        {
            hour: "2-digit",
            minute: "2-digit"
        }
    );

    // Portfolio

    const portfolio = await fetch(
        "/api/paper/portfolio",
        { headers }
    ).then(r => r.json());

    let tbody =
        document.querySelector("#portfolio-table tbody");

    tbody.innerHTML = "";

    // =======================================
// Performance Summary
// =======================================

const winners =
    portfolio.filter(
        stock => stock.profit_loss > 0
    ).length;

const losers =
    portfolio.filter(
        stock => stock.profit_loss < 0
    ).length;

document.getElementById("winningPositions").innerHTML =
    winners;

document.getElementById("losingPositions").innerHTML =
    losers;

document.getElementById("holdingCount").innerHTML =
    portfolio.length;

const health =
    winners >= losers
        ? "Excellent"
        : "Needs Review";

const healthElement =
    document.getElementById("portfolioHealth");

healthElement.innerHTML = health;

healthElement.style.color =
    winners >= losers
        ? "#22c55e"
        : "#ef4444";

    portfolio.forEach(stock => {

    const pnlColor =
        stock.profit_loss >= 0
            ? "#22c55e"
            : "#ef4444";

    tbody.innerHTML += `
    <tr>
        <td>${stock.symbol}</td>
        <td>${stock.quantity}</td>
        <td>₹${stock.average_price.toFixed(2)}</td>
        <td>₹${stock.current_price.toFixed(2)}</td>
        <td>₹${stock.investment.toLocaleString()}</td>
        <td>₹${stock.current_value.toLocaleString()}</td>
        <td style="color:${pnlColor};font-weight:bold;">
            ₹${stock.profit_loss.toFixed(2)}
        </td>
    </tr>
    `;

});

    // History

    const history = await fetch(
        "/api/paper/history",
        { headers }
    ).then(r => r.json());

    tbody =
        document.querySelector("#history-table tbody");

    tbody.innerHTML = "";

    history.forEach(trade => {

    const date = new Date(trade.date);

    const formattedDate =
        date.toLocaleString("en-IN", {

            day: "2-digit",

            month: "short",

            year: "numeric",

            hour: "2-digit",

            minute: "2-digit",

            hour12: true

        });

    const typeColor =
        trade.type === "BUY"
            ? "#22c55e"
            : "#ef4444";

    tbody.innerHTML += `
    <tr>

        <td>${formattedDate}</td>

        <td>${trade.symbol}</td>

        <td style="color:${typeColor};font-weight:bold;">
            ${trade.type}
        </td>

        <td>${trade.quantity}</td>

        <td>₹${Number(trade.price).toLocaleString("en-IN", {
            minimumFractionDigits: 2,
            maximumFractionDigits: 2
        })}</td>

        <td>₹${Number(trade.total).toLocaleString("en-IN", {
            minimumFractionDigits: 2,
            maximumFractionDigits: 2
        })}</td>

    </tr>
    `;

});

// ===============================
// Portfolio Allocation Chart
// ===============================

const labels = [];
const values = [];

portfolio.forEach(stock => {

    labels.push(stock.symbol);

    values.push(stock.current_value);

});

const chartCanvas =
    document.getElementById("allocationChart");

if (allocationChart) {

    allocationChart.destroy();

}

allocationChart = new Chart(chartCanvas, {

    type: "pie",

    data: {

        labels: labels,

        datasets: [

            {

                data: values,

                backgroundColor: [

                    "#3B82F6",
                    "#10B981",
                    "#F59E0B",
                    "#EF4444",
                    "#8B5CF6",
                    "#06B6D4",
                    "#EC4899",
                    "#84CC16"

                ],

                borderColor: "#161b22",

                borderWidth: 2

            }

        ]

    },

    options: {

        responsive: true,

        maintainAspectRatio: true,

        plugins: {

            legend: {

                position: "right",

                labels: {

                    color: "white",

                    font: {

                        size: 14

                    }

                }

            },

            tooltip: {

                callbacks: {

                    label: function(context) {

                        const total =
                            context.dataset.data.reduce(
                                (a, b) => a + b,
                                0
                            );

                        const value =
                            context.raw;

                        const percent =
                            (
                                value / total * 100
                            ).toFixed(1);

                        return (
                            context.label +
                            ": ₹" +
                            value.toLocaleString(
                                "en-IN"
                            ) +
                            " (" +
                            percent +
                            "%)"
                        );

                    }

                }

            }

        }

    }

});

// ===============================
// Portfolio Growth Chart
// ===============================

const growthCanvas =
    document.getElementById("growthChart");

if (growthChart) {

    growthChart.destroy();

}

growthChart = new Chart(growthCanvas, {

    type: "line",

    data: {

        labels: [

            "Investment",
            "Current"

        ],

        datasets: [

            {

                label: "Portfolio",

                data: [

                    dashboard.investment,

                    dashboard.portfolio_value

                ],

                borderColor: "#22c55e",

                backgroundColor: "rgba(34,197,94,0.15)",

                tension: 0.35,

                fill: true

            }

        ]

    },

    options: {

        responsive: true,

        plugins: {

            legend: {

                labels: {

                    color: "white"

                }

            }

        },

        scales: {

            x: {

                ticks: {

                    color: "white"

                }

            },

            y: {

                ticks: {

                    color: "white"

                }

            }

        }

    }

});

}

document
.getElementById("refreshDashboard")
.addEventListener(
    "click",
    loadDashboard
);

loadDashboard();

setInterval(() => {

    loadDashboard();

}, 60000);

