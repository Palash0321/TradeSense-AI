// ======================================
// Current Selected Market
// ======================================

let currentMarket =
    localStorage.getItem("selectedMarket") || "india";

const marketSelect =
    document.getElementById("market-select");

if (marketSelect) {

    marketSelect.value = currentMarket;

    marketSelect.addEventListener("change", function () {

        currentMarket = this.value;

        localStorage.setItem(
            "selectedMarket",
            currentMarket
        );

        loadAIPicks();

    });

}

// =====================================================
// DATABASE WATCHLIST
// =====================================================

const watchlistContainer =
    document.getElementById("watchlist-items");

let watchlistData = [];

async function loadWatchlist() {

    if (!watchlistContainer) return;

    watchlistContainer.innerHTML =
        "<p>Loading Watchlist...</p>";

    try {

        const watchlist =
    await getJSON("/api/watchlist");

watchlistData = [...watchlist];

        if (!watchlist.length) {

            watchlistContainer.innerHTML = `

                <div class="empty-watchlist">

                    <h3>📭 Your watchlist is empty</h3>

                    <p>Add stocks from the analysis page.</p>

                </div>

            `;

            return;

        }


for (const stock of watchlist) {

    const card = document.createElement("div");

    card.className = "watchlist-card";

card.innerHTML = `

<div class="watch-header">

    <div>

        <h3 class="watch-symbol">${stock.symbol}</h3>

        <p class="watch-company">
            ${stock.company ?? "Unknown Company"}
        </p>

    </div>

    <button
        class="remove-btn"
        onclick="removeStock(event,'${stock.symbol}')"
        title="Remove">

        🗑

    </button>

</div>


<div class="watch-main">

    <div class="price-widget">

    <div class="price-current watch-price">

        Loading...

    </div>

    <div class="market-status">

        Loading...

    </div>

    <div class="price-change watch-change">

        Loading...

    </div>

</div>

    <div class="watch-signal">

        Loading...

    </div>

</div>


<div class="stat-grid">

    <div class="stat-card">

        <div class="stat-label">

            AI Score

        </div>

        <div class="stat-value ai-score">

        </div>

    </div>

    <div class="stat-card">

        <div class="stat-label">

            Target

        </div>

        <div class="stat-value target-price">

        </div>

    </div>

    <div class="stat-card">

        <div class="stat-label">

            Risk

        </div>

        <div class="stat-value risk-level">

        </div>

    </div>

</div>


<div class="card-footer">

    <div class="card-meta">

        ⭐ Added to Watchlist

    </div>

    <div class="card-actions">

        <button class="analyze-btn">

            📊 Analyze Stock

        </button>

    </div>

</div>

`;

    watchlistContainer.appendChild(card);

    try {

        const market =
            stock.symbol.endsWith(".NS")
                ? "india"
                : "us";

        const symbol =
            stock.symbol.replace(".NS", "");

        const data = await getJSON(
    `/api/stock-summary?market=${market}&symbol=${symbol}`
);

        card.querySelector(".watch-price").textContent =
    `₹ ${data.price}`;


    const marketStatus =
    card.querySelector(".market-status");

if (data.market_status === "OPEN") {

    marketStatus.innerHTML =
        "🟢 Market Open";

    marketStatus.className =
        "market-status market-open";

}
else {

    marketStatus.innerHTML =
        "🔴 Market Closed";

    marketStatus.className =
        "market-status market-closed";

}


const changeElement =
    card.querySelector(".watch-change");

if (data.is_positive) {

    changeElement.className =
        "price-change price-up watch-change";

    changeElement.innerHTML =
        `▲ ₹${data.change} (${data.change_percent}%)`;

}
else {

    changeElement.className =
        "price-change price-down watch-change";

    changeElement.innerHTML =
        `▼ ₹${Math.abs(data.change)} (${Math.abs(data.change_percent)}%)`;

}

const badgeClass = {

    "BUY": "badge-buy",

    "SELL": "badge-sell",

    "HOLD": "badge-hold",

    "STRONG BUY": "badge-strong-buy",

    "STRONG SELL": "badge-strong-sell"

};

card.querySelector(".watch-signal").innerHTML = `

    <div class="badge ${badgeClass[data.signal] || "badge-hold"}">

        ${data.signal}

    </div>

    <small>

        Confidence : ${data.confidence}

    </small>

`;

        card.querySelector(".ai-score").textContent =
    data.ai_score;

card.querySelector(".target-price").textContent =
    "₹ " + data.target;

card.querySelector(".risk-level").textContent =
    data.risk;

    }

    catch (error) {

        console.error(error);

    }

    card.querySelector(".analyze-btn").onclick = function (event) {

    event.stopPropagation();

    const market =
        stock.symbol.endsWith(".NS")
            ? "india"
            : "us";

    const symbol =
        stock.symbol.replace(".NS", "");

    window.location.href =
        `/analyze?market=${market}&symbol=${symbol}`;

};

}

    }

    catch (error) {

        console.error(error);

        watchlistContainer.innerHTML =
            "<p>Unable to load watchlist.</p>";

    }

}

loadWatchlist();

const searchBox = document.getElementById("watchlist-search");

if (searchBox) {

    searchBox.addEventListener("input", function () {

        const query = this.value.trim().toLowerCase();

        document.querySelectorAll(".watchlist-card").forEach(card => {

            const symbol = card.querySelector(".watch-symbol")
                .textContent
                .toLowerCase();

            const company = card.querySelector(".watch-company")
                .textContent
                .toLowerCase();

            if (
                symbol.includes(query) ||
                company.includes(query)
            ) {
                card.style.display = "";
            } else {
                card.style.display = "none";
            }

        });

    });

}   

// ======================================
// WATCHLIST SORTING
// ======================================

const sortSelect =
    document.getElementById("watchlist-sort");

if (sortSelect) {

    sortSelect.addEventListener("change", function () {

        const cards = Array.from(
            watchlistContainer.querySelectorAll(".watchlist-card")
        );

        cards.sort((a, b) => {

            switch (this.value) {

                case "symbol":

                    return a.querySelector(".watch-symbol")
                        .textContent
                        .localeCompare(
                            b.querySelector(".watch-symbol")
                            .textContent
                        );

                case "score":

                    return Number(
                        b.querySelector(".ai-score").textContent
                    ) - Number(
                        a.querySelector(".ai-score").textContent
                    );

                case "price":

                    return parseFloat(
                        b.querySelector(".watch-price")
                            .textContent
                            .replace(/[^\d.-]/g, "")
                    ) - parseFloat(
                        a.querySelector(".watch-price")
                            .textContent
                            .replace(/[^\d.-]/g, "")
                    );

                case "change":

                    return parseFloat(
                        (
                            b.querySelector(".watch-change")
                                .textContent
                                .match(/-?\d+(\.\d+)?(?=%)/) || [0]
                        )[0]
                    ) - parseFloat(
                        (
                            a.querySelector(".watch-change")
                                .textContent
                                .match(/-?\d+(\.\d+)?(?=%)/) || [0]
                        )[0]
                    );

                default:

                    return 0;

            }

        });

        cards.forEach(card => {

            watchlistContainer.appendChild(card);

        });

    });

}

async function removeStock(event, symbol) {

    event.stopPropagation();

    if (!confirm(`Remove ${symbol} from Watchlist?`))
        return;

    const response = await fetch(

        `/api/watchlist/${symbol}`,

        {

            method: "DELETE"

        }

    );

    const result = await response.json();

    if (result.success) {

        loadWatchlist();

    }

}

// =====================================================
// AI STOCK SCREENER
// =====================================================

let currentFilter = "ALL";

async function loadAIPicks() {

    const container =
        document.getElementById("screener-container");

    if (!container) return;

    container.innerHTML =
        "<p>Loading AI Picks...</p>";

    try {

        const stocks =
    await getJSON(`/api/ai-picks?market=${currentMarket}`);

        if (!stocks.length) {

            container.innerHTML =
                "<p>No AI Picks Available.</p>";

            return;

        }

        container.innerHTML = "";

        const filteredStocks = stocks.filter(stock => {

            if (currentFilter === "ALL")
                return true;

            return stock.signal === currentFilter;

        });

        filteredStocks.forEach((stock, index) => {

    const card =
        document.createElement("div");

    card.className =
        "ai-card";

    card.innerHTML = `

<div class="watch-header">

    <div>

        <h3 class="watch-symbol">

            #${index + 1} ${stock.symbol}

        </h3>

        <p class="watch-company">

            ${stock.company}

        </p>

    </div>

</div>

<div class="watch-main">

    <div class="price-widget">

        <div class="price-current">

            ₹ ${stock.price ?? "--"}

        </div>

        <div class="market-status">

            Loading...

        </div>

        <div class="price-change">

            Loading...

        </div>

    </div>

    <div class="watch-signal">

    </div>

</div>

<div class="stat-grid">

    <div class="stat-card">

        <div class="stat-label">

            AI Score

        </div>

        <div class="stat-value">

            ${stock.ai_score}

        </div>

    </div>

    <div class="stat-card">

        <div class="stat-label">

            Confidence

        </div>

        <div class="stat-value">

            ${stock.confidence}

        </div>

    </div>

    <div class="stat-card">

        <div class="stat-label">

            Signal

        </div>

        <div class="stat-value">

            ${stock.signal}

        </div>

    </div>

</div>

<div class="card-footer">

    <div class="card-meta">

        🤖 Ranked by AI

    </div>

    <div class="card-actions">

        <button class="analyze-btn">

            📊 Analyze

        </button>

    </div>

</div>

`;

const badgeClass = {

    "BUY": "badge-buy",

    "SELL": "badge-sell",

    "HOLD": "badge-hold",

    "STRONG BUY": "badge-strong-buy",

    "STRONG SELL": "badge-strong-sell"

};

card.querySelector(".watch-signal").innerHTML = `

<div class="badge ${badgeClass[stock.signal] || "badge-hold"}">

    ${stock.signal}

</div>

`;

            card.onclick = function () {

                const market =
                    stock.symbol.endsWith(".NS")
                    ? "india"
                    : "us";

                const symbol =
                    stock.symbol.replace(".NS", "");

                window.location.href =
                    `/analyze?market=${market}&symbol=${symbol}`;

            };

            container.appendChild(card);

        });

    }

    catch (error) {

        console.error(error);

        container.innerHTML =
            "<p>Unable to load AI Picks.</p>";

    }

}

loadAIPicks();

document.querySelectorAll(".filter-btn").forEach(button => {

    button.addEventListener("click", function () {

        document.querySelectorAll(".filter-btn").forEach(btn => {

            btn.classList.remove("active");

        });

        this.classList.add("active");

        currentFilter =
            this.dataset.filter;

        loadAIPicks();

    });

});