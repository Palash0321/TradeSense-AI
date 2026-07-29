// =====================================================
// DASHBOARD - MARKET OVERVIEW
// =====================================================

async function loadMarketOverview() {

    try {

        const data =
            await getJSON("/api/market-overview");

        updateMarketCard(
            "nifty",
            data.nifty
        );

        updateMarketCard(
            "sensex",
            data.sensex
        );

        updateMarketCard(
            "sp500",
            data.sp500
        );

    }

    catch (error) {

        console.error(
            "Unable to load market overview:",
            error
        );

    }

}


// =====================================================
// UPDATE MARKET CARD
// =====================================================

function updateMarketCard(id, market) {

    const priceElement =
        document.getElementById(`${id}-price`);

    const changeElement =
        document.getElementById(`${id}-change`);

    if (!priceElement || !changeElement)
        return;


    if (!market || market.price === null) {

        priceElement.textContent = "--";

        changeElement.textContent =
            "Market data unavailable";

        changeElement.className =
            "market-change neutral";

        return;

    }


    priceElement.textContent =
        Number(market.price).toLocaleString(
            "en-IN",
            {
                minimumFractionDigits: 2,
                maximumFractionDigits: 2
            }
        );


    const change =
        Number(market.change);

    const changePercent =
        Number(market.change_percent);


    if (market.is_positive) {

        changeElement.textContent =
            `▲ ${Math.abs(change).toFixed(2)} (${Math.abs(changePercent).toFixed(2)}%)`;

        changeElement.className =
            "market-change positive";

    }

    else {

        changeElement.textContent =
            `▼ ${Math.abs(change).toFixed(2)} (${Math.abs(changePercent).toFixed(2)}%)`;

        changeElement.className =
            "market-change negative";

    }

}


// =====================================================
// DASHBOARD - MARKET MOVERS
// =====================================================

async function loadMarketMovers() {

    const gainersContainer =
        document.getElementById("gainers-list");

    const losersContainer =
        document.getElementById("losers-list");


    if (!gainersContainer || !losersContainer)
        return;


    try {

        const data =
            await getJSON(
                "/api/market-movers?market=india"
            );


        renderMovers(
            gainersContainer,
            data.gainers,
            "positive"
        );


        renderMovers(
            losersContainer,
            data.losers,
            "negative"
        );

    }

    catch (error) {

        console.error(
            "Unable to load market movers:",
            error
        );


        gainersContainer.innerHTML = `
            <div class="movers-loading">
                Unable to load market data.
            </div>
        `;


        losersContainer.innerHTML = `
            <div class="movers-loading">
                Unable to load market data.
            </div>
        `;

    }

}


// =====================================================
// RENDER MARKET MOVERS
// =====================================================

function renderMovers(
    container,
    stocks,
    type
) {

    container.innerHTML = "";


    if (!stocks || stocks.length === 0) {

        container.innerHTML = `
            <div class="movers-loading">
                No market data available.
            </div>
        `;

        return;

    }


    stocks.forEach((stock, index) => {

        const row =
            document.createElement("div");

        row.className =
            "mover-row";


        const arrow =
            type === "positive"
                ? "▲"
                : "▼";


        const formattedPrice =
            Number(stock.price).toLocaleString(
                "en-IN",
                {
                    minimumFractionDigits: 2,
                    maximumFractionDigits: 2
                }
            );


        row.innerHTML = `

            <div class="mover-left">

                <div class="mover-rank">
                    ${index + 1}
                </div>


                <div>

                    <div class="mover-symbol">
                        ${stock.symbol}
                    </div>

                    <div class="mover-exchange">
                        NSE
                    </div>

                </div>

            </div>


            <div class="mover-right">

                <div class="mover-price">
                    ₹${formattedPrice}
                </div>

                <div class="mover-change ${type}">

                    ${arrow}

                    ${Math.abs(
                        Number(
                            stock.change_percent
                        )
                    ).toFixed(2)}%

                </div>

            </div>

        `;


        row.addEventListener(
            "click",
            function () {

                window.location.href =
                    `/analyze?market=india&symbol=${stock.symbol}`;

            }
        );


        container.appendChild(row);

    });

}


// =====================================================
// INITIAL DASHBOARD LOAD
// =====================================================


// =====================================================
// DASHBOARD - AI OPPORTUNITIES
// =====================================================

async function loadDashboardAIPicks() {

    const container =
        document.getElementById(
            "dashboard-ai-picks"
        );

    if (!container)
        return;


    try {

        const stocks =
            await getJSON(
                "/api/ai-picks?market=india"
            );


        if (!stocks || stocks.length === 0) {

            container.innerHTML = `
                <div class="dashboard-loading">
                    No AI opportunities available.
                </div>
            `;

            return;

        }


        container.innerHTML = "";


        stocks
            .slice(0, 4)
            .forEach((stock, index) => {

                const card =
                    document.createElement("article");


                card.className =
                    "dashboard-ai-card";


                const signalClass =
                    getDashboardSignalClass(
                        stock.signal
                    );


                card.innerHTML = `

                    <div class="dashboard-ai-top">

                        <div class="ai-rank">
                            #${index + 1}
                        </div>

                        <div class="dashboard-signal ${signalClass}">
                            ${stock.signal}
                        </div>

                    </div>


                    <div class="dashboard-ai-stock">

                        <h3>
                            ${stock.symbol}
                        </h3>

                        <p>
                            ${stock.company ?? "Unknown Company"}
                        </p>

                    </div>


                    <div class="dashboard-ai-stats">

                        <div>

                            <span>
                                AI Score
                            </span>

                            <strong>
                                ${stock.ai_score}
                            </strong>

                        </div>


                        <div>

                            <span>
                                Confidence
                            </span>

                            <strong>
                                ${stock.confidence}
                            </strong>

                        </div>

                    </div>


                    <button
                        class="dashboard-analyze-btn"
                        type="button">

                        Analyze Stock →

                    </button>

                `;


                card.querySelector(
                    ".dashboard-analyze-btn"
                ).addEventListener(
                    "click",
                    function () {

                        const symbol =
                            stock.symbol.replace(
                                ".NS",
                                ""
                            );

                        window.location.href =
                            `/analyze?market=india&symbol=${symbol}`;

                    }
                );


                container.appendChild(card);

            });

    }

    catch (error) {

        console.error(
            "Unable to load AI opportunities:",
            error
        );


        container.innerHTML = `
            <div class="dashboard-loading">
                Unable to load AI opportunities.
            </div>
        `;

    }

}


// =====================================================
// AI SIGNAL CLASS
// =====================================================

function getDashboardSignalClass(signal) {

    switch (signal) {

        case "STRONG BUY":
            return "signal-strong-buy";

        case "BUY":
            return "signal-buy";

        case "HOLD":
            return "signal-hold";

        case "SELL":
            return "signal-sell";

        case "STRONG SELL":
            return "signal-strong-sell";

        default:
            return "signal-hold";

    }

}

// =====================================================
// QUICK STOCK ANALYZER
// =====================================================

const quickAnalyzerForm =
    document.getElementById(
        "quick-analyzer-form"
    );


if (quickAnalyzerForm) {

    quickAnalyzerForm.addEventListener(
        "submit",
        function (event) {

            event.preventDefault();


            const market =
                document.getElementById(
                    "quick-market"
                ).value;


            let symbol =
                document.getElementById(
                    "quick-symbol"
                ).value
                .trim()
                .toUpperCase();


            if (!symbol)
                return;


            // Prevent accidental .NS duplication
            if (
                market === "india" &&
                symbol.endsWith(".NS")
            ) {

                symbol =
                    symbol.replace(
                        ".NS",
                        ""
                    );

            }


            window.location.href =
                `/analyze?market=${encodeURIComponent(market)}&symbol=${encodeURIComponent(symbol)}`;

        }
    );

}

// =====================================================
// DASHBOARD - MARKET NEWS
// =====================================================

async function loadMarketNews() {

    const container =
        document.getElementById(
            "market-news-container"
        );

    if (!container)
        return;


    try {

        const articles =
            await getJSON(
                "/api/market-news"
            );


        if (!articles || articles.length === 0) {

            container.innerHTML = `
                <div class="dashboard-loading">
                    No market news available.
                </div>
            `;

            return;

        }


        container.innerHTML = "";


        articles.forEach(
            (article, index) => {

                const newsCard =
                    document.createElement(
                        "article"
                    );

                newsCard.className =
                    "market-news-card";


                newsCard.innerHTML = `

                    <div class="news-number">

                        ${String(index + 1)
                            .padStart(2, "0")}

                    </div>


                    <div class="news-content">

                        <div class="news-meta">

                            <span>
                                ${article.source_query}
                            </span>

                            <span>
                                ${formatNewsDate(
                                    article.published
                                )}
                            </span>

                        </div>


                        <h3>
                            ${article.title}
                        </h3>


                        <a
                            href="${article.link}"
                            target="_blank"
                            rel="noopener noreferrer"
                            class="news-read-link">

                            Read Article →

                        </a>

                    </div>

                `;


                container.appendChild(
                    newsCard
                );

            }
        );

    }

    catch (error) {

        console.error(
            "Unable to load market news:",
            error
        );


        container.innerHTML = `
            <div class="dashboard-loading">
                Unable to load market news.
            </div>
        `;

    }

}


// =====================================================
// FORMAT NEWS DATE
// =====================================================

function formatNewsDate(dateString) {

    if (!dateString)
        return "Recent";


    const date =
        new Date(dateString);


    if (Number.isNaN(date.getTime()))
        return "Recent";


    return date.toLocaleDateString(
        "en-IN",
        {
            day: "numeric",
            month: "short",
            year: "numeric"
        }
    );

}

// =====================================================
// DASHBOARD - WATCHLIST SNAPSHOT
// =====================================================

async function loadDashboardWatchlist() {

    const container =
        document.getElementById(
            "dashboard-watchlist"
        );

    if (!container)
        return;


    try {

        const watchlist =
            await getJSON("/api/watchlist");


        if (!watchlist || watchlist.length === 0) {

            container.innerHTML = `

                <div class="dashboard-empty-state">

                    <div class="empty-icon">
                        ☆
                    </div>

                    <div>

                        <h3>
                            Your watchlist is empty
                        </h3>

                        <p>
                            Analyze stocks and save the ones
                            you want to monitor.
                        </p>

                    </div>

                    <a
                        href="/"
                        class="empty-action">

                        Find Stocks →

                    </a>

                </div>

            `;

            return;

        }


        container.innerHTML = "";


        const stocks =
            watchlist.slice(0, 4);


        for (const stock of stocks) {

            const card =
                document.createElement(
                    "article"
                );

            card.className =
                "dashboard-watch-card";


            card.innerHTML = `

                <div class="watch-card-loading">
                    Loading ${stock.symbol}...
                </div>

            `;


            container.appendChild(card);


            try {

                const market =
                    stock.symbol.endsWith(".NS")
                        ? "india"
                        : "us";


                const cleanSymbol =
                    stock.symbol.replace(
                        ".NS",
                        ""
                    );


                const data =
                    await getJSON(
                        `/api/stock-summary?market=${market}&symbol=${cleanSymbol}`
                    );


                renderDashboardWatchCard(
                    card,
                    stock,
                    data,
                    market,
                    cleanSymbol
                );

            }

            catch (error) {

                console.error(
                    `Unable to load ${stock.symbol}:`,
                    error
                );


                card.innerHTML = `

                    <div class="watch-card-error">

                        Unable to load
                        ${stock.symbol}

                    </div>

                `;

            }

        }

    }

    catch (error) {

        console.error(
            "Unable to load dashboard watchlist:",
            error
        );


        container.innerHTML = `

            <div class="dashboard-loading">
                Unable to load watchlist.
            </div>

        `;

    }

}


// =====================================================
// RENDER WATCHLIST CARD
// =====================================================

function renderDashboardWatchCard(
    card,
    stock,
    data,
    market,
    cleanSymbol
) {

    const positive =
        data.is_positive === true;


    const movementClass =
        positive
            ? "positive"
            : "negative";


    const arrow =
        positive
            ? "▲"
            : "▼";


    const signalClass =
        getDashboardSignalClass(
            data.signal
        );


    card.innerHTML = `

        <div class="watch-card-top">

            <div>

                <h3>
                    ${cleanSymbol}
                </h3>

                <p>
                    ${data.company ?? stock.company ?? ""}
                </p>

            </div>


            <span class="dashboard-signal ${signalClass}">

                ${data.signal}

            </span>

        </div>


        <div class="watch-card-price">

            ${market === "india" ? "₹" : "$"}
            ${data.price}

        </div>


        <div class="watch-card-change ${movementClass}">

            ${arrow}

            ${Math.abs(
                Number(
                    data.change_percent ?? 0
                )
            ).toFixed(2)}%

        </div>


        <div class="watch-card-stats">

            <div>

                <span>AI Score</span>

                <strong>
                    ${data.ai_score}
                </strong>

            </div>


            <div>

                <span>Risk</span>

                <strong>
                    ${data.risk}
                </strong>

            </div>

        </div>


        <button
            class="watch-card-analyze"
            type="button">

            View Analysis →

        </button>

    `;


    card.querySelector(
        ".watch-card-analyze"
    ).addEventListener(
        "click",
        function () {

            window.location.href =
                `/analyze?market=${market}&symbol=${cleanSymbol}`;

        }
    );

}

loadMarketOverview();

loadMarketMovers();

loadDashboardAIPicks();

loadMarketNews();

loadDashboardWatchlist();