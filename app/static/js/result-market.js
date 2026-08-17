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

// =========================================================
// MARKET BREADTH
// =========================================================

async function loadMarketBreadth() {

    try {

        const response =
            await fetch("/api/market-breadth");

        if (!response.ok) {

            throw new Error(
                "Market breadth request failed"
            );

        }

        const data =
            await response.json();

        console.log(
            "Market Breadth:",
            data
        );

        document.getElementById(
            "breadth-advancing"
        ).textContent =
            data.advancing;

        document.getElementById(
            "breadth-declining"
        ).textContent =
            data.declining;

        document.getElementById(
            "breadth-unchanged"
        ).textContent =
            data.unchanged;

        document.getElementById(
            "breadth-ratio"
        ).textContent =
            data.ratio;

        document.getElementById(
            "breadth-health"
        ).textContent =
            data.health;

        document.getElementById(
            "breadth-tracked"
        ).textContent =
            data.tracked_stocks;

    } catch (error) {

        console.error(
            "Market breadth error:",
            error
        );

    }

}

loadMarketBreadth();

// =========================================================
// MARKET INTELLIGENCE SUMMARY
// =========================================================

async function loadMarketIntelligenceSummary() {

    const container =
        document.getElementById(
            "market-summary-content"
        );

    if (!container) {
        return;
    }

    try {

        const [
            overviewResponse,
            moversResponse,
            sectorsResponse,
            breadthResponse
        ] = await Promise.all([

            fetch("/api/market-overview"),

            fetch("/api/market-movers?market=india"),

            fetch("/api/sectors"),

            fetch("/api/market-breadth")

        ]);


        if (
            !overviewResponse.ok ||
            !moversResponse.ok ||
            !sectorsResponse.ok ||
            !breadthResponse.ok
        ) {

            throw new Error(
                "Market intelligence data request failed"
            );

        }


        const overview =
            await overviewResponse.json();

        const movers =
            await moversResponse.json();

        const sectors =
            await sectorsResponse.json();

        const breadth =
            await breadthResponse.json();


        console.log(
            "Market Intelligence Summary Data:",
            {
                overview,
                movers,
                sectors,
                breadth
            }
        );


        generateMarketSummary(
            overview,
            movers,
            sectors,
            breadth
        );


    } catch (error) {

        console.error(
            "Market intelligence summary error:",
            error
        );

        container.innerHTML = `
            <div class="market-summary-error">
                Market intelligence is temporarily unavailable.
            </div>
        `;

    }

}


// =========================================================
// GENERATE MARKET SUMMARY
// =========================================================

function generateMarketSummary(
    overview,
    movers,
    sectors,
    breadth
) {

    const container =
        document.getElementById(
            "market-summary-content"
        );

    if (!container) {
        return;
    }


    const advancing =
        Number(breadth.advancing ?? 0);

    const declining =
        Number(breadth.declining ?? 0);

    const unchanged =
        Number(breadth.unchanged ?? 0);

    const ratio =
        Number(breadth.ratio ?? 0);


    // -----------------------------------------
    // MARKET CONDITION
    // -----------------------------------------

    let condition;
    let conditionClass;

    if (ratio > 1.5) {

        condition = "Bullish";
        conditionClass = "market-positive";

    } else if (ratio > 1) {

        condition = "Moderately Bullish";
        conditionClass = "market-positive";

    } else if (ratio === 1) {

        condition = "Neutral";
        conditionClass = "";

    } else {

        condition = "Bearish";
        conditionClass = "market-negative";

    }


    // -----------------------------------------
    // BREADTH INTERPRETATION
    // -----------------------------------------

    let breadthText;

    if (advancing > declining) {

        breadthText =
            `${advancing} of ${breadth.tracked_stocks} tracked stocks are advancing, indicating broader market participation.`;

    } else if (declining > advancing) {

        breadthText =
            `${declining} of ${breadth.tracked_stocks} tracked stocks are declining, indicating weaker market participation.`;

    } else {

        breadthText =
            `Advancing and declining stocks are currently balanced across the tracked universe.`;

    }


    // -----------------------------------------
    // SECTOR INTERPRETATION
    // -----------------------------------------

    const positiveSectors =
        sectors.filter(
            sector =>
                Number(sector.change ?? 0) > 0
        );

    const negativeSectors =
        sectors.filter(
            sector =>
                Number(sector.change ?? 0) < 0
        );


    let sectorText;

    if (
        positiveSectors.length >
        negativeSectors.length
    ) {

        sectorText =
            `Sector performance is broadly positive, with ${positiveSectors.length} of ${sectors.length} tracked sectors gaining.`;

    } else if (
        negativeSectors.length >
        positiveSectors.length
    ) {

        sectorText =
            `Sector performance is broadly weak, with ${negativeSectors.length} of ${sectors.length} tracked sectors declining.`;

    } else {

        sectorText =
            `Sector performance is mixed, with gains and declines relatively balanced.`;

    }


    // -----------------------------------------
    // GLOBAL MARKET INTERPRETATION
    // -----------------------------------------

    const globalMarkets = [
        overview.nifty,
        overview.sensex,
        overview.sp500
    ].filter(Boolean);


    const positiveMarkets =
        globalMarkets.filter(
            market =>
                Number(market.change ?? 0) >= 0
        );


    let globalText;

    if (
        positiveMarkets.length >
        globalMarkets.length / 2
    ) {

        globalText =
            `Major tracked indices are showing generally positive movement.`;

    } else if (
        positiveMarkets.length <
        globalMarkets.length / 2
    ) {

        globalText =
            `Major tracked indices are showing generally negative movement.`;

    } else {

        globalText =
            `Major tracked indices are showing mixed movement.`;

    }


    // -----------------------------------------
    // FINAL SUMMARY
    // -----------------------------------------

    container.innerHTML = `

        <div class="market-summary-condition">

            <span>
                Overall Market Condition
            </span>

            <strong class="${conditionClass}">
                ${condition}
            </strong>

        </div>


        <div class="market-summary-text">

            <p>
                ${breadthText}
            </p>

            <p>
                ${sectorText}
            </p>

            <p>
                ${globalText}
            </p>

        </div>


        <div class="market-summary-metrics">

            <div>
                <span>Advancing</span>
                <strong>
                    ${advancing}
                </strong>
            </div>

            <div>
                <span>Declining</span>
                <strong>
                    ${declining}
                </strong>
            </div>

            <div>
                <span>A/D Ratio</span>
                <strong>
                    ${ratio.toFixed(2)}
                </strong>
            </div>

            <div>
                <span>Market Health</span>
                <strong>
                    ${breadth.health}
                </strong>
            </div>

        </div>

    `;

}


loadMarketIntelligenceSummary();