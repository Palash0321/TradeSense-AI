// =========================================================
// RESULT PAGE - MARKET INTELLIGENCE
// =========================================================

document.addEventListener("DOMContentLoaded", () => {

    loadMarketOverview();
    loadMarketMovers();

});


// =========================================================
// MARKET OVERVIEW
// =========================================================

async function loadMarketOverview() {

    try {

        const response = await fetch("/api/market-overview");

        if (!response.ok) {
            throw new Error("Market overview request failed");
        }

        const data = await response.json();

        console.log("Market Overview:", data);

        updateMarketCard(
            data.nifty,
            "nifty-market-price",
            "nifty-market-change",
            "nifty-market-indicator"
        );

        updateMarketCard(
            data.sensex,
            "sensex-market-price",
            "sensex-market-change",
            "sensex-market-indicator"
        );

        updateMarketCard(
            data.sp500,
            "sp500-market-price",
            "sp500-market-change",
            "sp500-market-indicator"
        );

        updateMarketStatus(data);

    } catch (error) {

        console.error(
            "Market overview error:",
            error
        );

    }

}


// =========================================================
// UPDATE MARKET CARD
// =========================================================

function updateMarketCard(
    market,
    priceId,
    changeId,
    indicatorId
) {

    if (!market) {
        return;
    }

    const priceElement =
        document.getElementById(priceId);

    const changeElement =
        document.getElementById(changeId);

    const indicatorElement =
        document.getElementById(indicatorId);


    if (priceElement) {

        priceElement.textContent =
            market.price ?? "--";

    }


    if (changeElement) {

        const change =
            Number(market.change ?? 0);

        const percent =
            Number(market.change_percent ?? 0);


        if (change >= 0) {

            changeElement.textContent =
                `▲ +${change.toFixed(2)} (+${percent.toFixed(2)}%)`;

            changeElement.classList.remove(
                "market-negative"
            );

            changeElement.classList.add(
                "market-positive"
            );

        } else {

            changeElement.textContent =
                `▼ ${change.toFixed(2)} (${percent.toFixed(2)}%)`;

            changeElement.classList.remove(
                "market-positive"
            );

            changeElement.classList.add(
                "market-negative"
            );

        }

    }


    if (indicatorElement) {

        indicatorElement.textContent =
            Number(market.change ?? 0) >= 0
                ? "▲"
                : "▼";

        indicatorElement.classList.remove(
            "market-positive",
            "market-negative"
        );

        indicatorElement.classList.add(
            Number(market.change ?? 0) >= 0
                ? "market-positive"
                : "market-negative"
        );

    }

}


// =========================================================
// MARKET STATUS
// =========================================================

function updateMarketStatus(data) {

    const statusElement =
        document.getElementById(
            "global-market-status"
        );

    if (!statusElement) {
        return;
    }


    if (
        data.nifty &&
        data.nifty.market_open
    ) {

        statusElement.textContent =
            "🟢 Market Open";

    } else {

        statusElement.textContent =
            "🔴 Market Closed";

    }

}


// =========================================================
// MARKET MOVERS
// =========================================================

async function loadMarketMovers() {

    try {

        const response =
            await fetch(
                "/api/market-movers?market=india"
            );

        if (!response.ok) {
            throw new Error(
                "Market movers request failed"
            );
        }

        const data =
            await response.json();

        console.log(
            "Market Movers:",
            data
        );


        renderMovers(
            data.gainers,
            "result-gainers-list",
            true
        );


        renderMovers(
            data.losers,
            "result-losers-list",
            false
        );


    } catch (error) {

        console.error(
            "Market movers error:",
            error
        );

    }

}


// =========================================================
// RENDER MOVERS
// =========================================================

function renderMovers(
    movers,
    containerId,
    positive
) {

    const container =
        document.getElementById(containerId);

    if (!container) {
        return;
    }


    if (
        !movers ||
        movers.length === 0
    ) {

        container.innerHTML = `
            <div class="market-loading">
                No market data available.
            </div>
        `;

        return;

    }


    container.innerHTML = "";


    movers.forEach(stock => {

        const row =
            document.createElement("div");

        row.className =
            "mover-row";


        const symbol =
            document.createElement("div");

        symbol.innerHTML = `
            <div class="mover-symbol">
                ${stock.symbol ?? "--"}
            </div>

            <div class="mover-price">
                ₹ ${stock.price ?? "--"}
            </div>
        `;


        const change =
            document.createElement("div");

        change.className =
            `mover-change ${
                positive
                    ? "market-positive"
                    : "market-negative"
            }`;


        const percent =
            Number(
                stock.change_percent ?? 0
            );


        change.textContent =
            positive
                ? `▲ +${percent.toFixed(2)}%`
                : `▼ ${percent.toFixed(2)}%`;


        row.appendChild(symbol);

        row.appendChild(change);

        container.appendChild(row);

    });

}

// =========================================================
// SECTOR PERFORMANCE
// =========================================================

async function loadSectorPerformance() {

    try {

        const response =
            await fetch("/api/sectors");

        if (!response.ok) {
            throw new Error(
                "Sector performance request failed"
            );
        }

        const sectors =
            await response.json();

        console.log(
            "Sector Performance:",
            sectors
        );

        renderSectorPerformance(sectors);

    } catch (error) {

        console.error(
            "Sector performance error:",
            error
        );

    }

}


// =========================================================
// RENDER SECTORS
// =========================================================

function renderSectorPerformance(sectors) {

    const container =
        document.getElementById(
            "result-sector-list"
        );

    if (!container) {
        return;
    }

    if (
        !sectors ||
        sectors.length === 0
    ) {

        container.innerHTML = `
            <div class="market-loading">
                No sector data available.
            </div>
        `;

        return;

    }

    container.innerHTML = "";

    sectors.forEach(sector => {

        const card =
            document.createElement("div");

        card.className =
            "sector-card";

        const change =
            Number(sector.change ?? 0);

        const changeClass =
            change >= 0
                ? "market-positive"
                : "market-negative";

        const arrow =
            change >= 0
                ? "▲"
                : "▼";

        card.innerHTML = `

            <div class="sector-name">
                ${sector.name}
            </div>

            <div class="sector-change ${changeClass}">
                ${arrow}
                ${change >= 0 ? "+" : ""}
                ${change.toFixed(2)}%
            </div>

        `;

        container.appendChild(card);

    });

}


// Load sector performance
loadSectorPerformance();