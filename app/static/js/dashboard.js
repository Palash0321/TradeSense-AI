let niftyChart = null;
let candleSeries = null;
let volumeChart = null;
let volumeSeries = null;
let rsiChart = null;
let rsiSeries = null;

let macdChart = null;
let macdSeries = null;
let signalSeries = null;
let histogramSeries = null;

let ema20Series = null;
let ema50Series = null;
let bbUpperSeries = null;
let bbMiddleSeries = null;
let bbLowerSeries = null;
let vwapSeries = null;
let crosshairSubscribed = false;
let drawingMode = "cursor";

let drawingPoints = [];

let drawingSeries = [];


// =====================================================
// DASHBOARD - MARKET OVERVIEW
// =====================================================

async function loadMarketOverview() {

    try {

        const data =
            await getJSON("/api/market-overview");

            updateMarketStatus(
    data.market_status
);

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

        updateHeroCard(id, null);

        return;

    }

    priceElement.textContent =
        Number(market.price).toLocaleString(
            "en-IN",
            {
                minimumFractionDigits:2,
                maximumFractionDigits:2
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

    updateHeroCard(id, market);

}

function updateHeroCard(id, market){

    const map = {

        nifty:"hero-nifty",

        sensex:"hero-sensex",

        sp500:"hero-sp500"

    };

    const changeMap = {

        nifty:"hero-nifty-change",

        sensex:"hero-sensex-change",

        sp500:"hero-sp500-change"

    };

    const value =
        document.getElementById(map[id]);

    const change =
        document.getElementById(changeMap[id]);

    if(!value || !change)
        return;

    if(!market){

        value.textContent="--";

        change.textContent="Unavailable";

        change.className="hero-neutral";

        return;

    }

    value.textContent =
        Number(market.price).toLocaleString(
            "en-IN",
            {
                maximumFractionDigits:2
            }
        );

    const pct =
        Number(market.change_percent);

    if(market.is_positive){

        change.textContent=
            `▲ ${pct.toFixed(2)}%`;

        change.className="hero-positive";

    }

    else{

        change.textContent=
            `▼ ${Math.abs(pct).toFixed(2)}%`;

        change.className="hero-negative";

    }

    drawMiniChart(
        `${id}-chart`,
        market.is_positive
    );


    updateMarketMood();

}

function updateMarketMood(){

    const mood =
        document.getElementById("hero-mood");

    if(!mood)
        return;

    const positive =
        document.querySelectorAll(".hero-positive").length;

    const negative =
        document.querySelectorAll(".hero-negative").length;

    if(positive>=2){

        mood.textContent="🟢 Bullish";

        mood.className="hero-positive";

    }

    else if(negative>=2){

        mood.textContent="🔴 Bearish";

        mood.className="hero-negative";

    }

    else{

        mood.textContent="🟡 Neutral";

        mood.className="hero-neutral";

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

async function initializeDashboard() {

    await loadMarketOverview();

    await loadMarketMovers();

    await loadDashboardAIPicks();

    await loadMarketNews();

    await loadDashboardWatchlist();

    await loadOptionChain();

}

initializeDashboard();

function drawMiniChart(canvasId, positive){

    const canvas =
        document.getElementById(canvasId);

    if(!canvas)
        return;

    new Chart(canvas,{

        type:"line",

        data:{

            labels:[1,2,3,4,5,6,7,8],

            datasets:[{

                data: positive ?

                [5,6,7,8,9,9.2,10,11]

                :

                [11,10,9.5,9,8.8,8.4,8,7],

                borderColor:positive
                    ? "#22c55e"
                    : "#ef4444",

                borderWidth:2,

                fill:false,

                tension:.45,

                pointRadius:0

            }]

        },

        options:{

            responsive:true,

            maintainAspectRatio:false,

            plugins:{

                legend:{
                    display:false
                }

            },

            scales:{

                x:{
                    display:false
                },

                y:{
                    display:false
                }

            }

        }

    });

}

async function loadIndexChart(symbol = "^NSEI", period = "1mo") {

    const response = await fetch(
        `/api/index-history/${encodeURIComponent(symbol)}?period=${period}`
    );

    const data = await response.json();

    if (!niftyChart) {

        const container = document.getElementById("nifty-live-chart");
        const volumeContainer = document.getElementById("volume-chart");
        const macdContainer = document.getElementById("macd-chart");
       

        niftyChart = LightweightCharts.createChart(container, {

            layout: {
                background: {
                    color: "#111827"
                },
                textColor: "#ffffff"
            },

            grid: {
                vertLines: {
                    color: "#1f2937"
                },
                horzLines: {
                    color: "#1f2937"
                }
            },

            width: container.clientWidth,

            height: 450,

            rightPriceScale: {
                borderColor: "#374151"
            },

            timeScale: {

    borderColor: "#374151",

    rightOffset: 2,

    barSpacing: 14,

    minBarSpacing: 8,

    fixLeftEdge: true,

    fixRightEdge: true,

    lockVisibleTimeRangeOnResize: true,

    rightBarStaysOnScroll: true

}
        });

        candleSeries = niftyChart.addCandlestickSeries({
            upColor: "#22c55e",
            downColor: "#ef4444",
            borderVisible: false,
            wickUpColor: "#22c55e",
            wickDownColor: "#ef4444"
        });

        ema20Series = niftyChart.addLineSeries({

    color: "#3B82F6",

    lineWidth: 2,

    title: "EMA 20"

});

ema50Series = niftyChart.addLineSeries({

    color: "#F59E0B",

    lineWidth: 2,

    title: "EMA 50"

});

bbUpperSeries = niftyChart.addLineSeries({

    color: "#60A5FA",

    lineWidth: 1,

    title: "BB Upper"

});

bbMiddleSeries = niftyChart.addLineSeries({

    color: "#9CA3AF",

    lineWidth: 1,

    title: "BB Middle"

});

bbLowerSeries = niftyChart.addLineSeries({

    color: "#60A5FA",

    lineWidth: 1,

    title: "BB Lower"

});

vwapSeries = niftyChart.addLineSeries({

    color: "#14B8A6",

    lineWidth: 2,

    title: "VWAP"

});
        volumeChart = LightweightCharts.createChart(volumeContainer, {

    layout: {

        background: {
            color: "#111827"
        },

        textColor: "#888"

    },

    grid: {

        vertLines: {
            color: "#1f2937"
        },

        horzLines: {
            color: "#1f2937"
        }

    },

    width: volumeContainer.clientWidth,

    height: 120,

    rightPriceScale: {

        visible: false

    },

    timeScale: {

        visible: false

    }

});

volumeSeries = volumeChart.addHistogramSeries({

    priceFormat: {
        type: "volume"
    }

});

const rsiContainer =
    document.getElementById("rsi-chart");

rsiChart =
    LightweightCharts.createChart(
        rsiContainer,
        {
            layout: {
                background: {
                    color: "#111827"
                },
                textColor: "#9CA3AF"
            },

            width: rsiContainer.clientWidth,

            height: 180,

            rightPriceScale: {
                borderColor: "#2B3446"
            },

            timeScale: {
                borderColor: "#2B3446"
            },

            grid: {
                vertLines: {
                    color: "#1F2937"
                },
                horzLines: {
                    color: "#1F2937"
                }
            }
        }
    );

    rsiSeries = rsiChart.addLineSeries({

    color: "#A855F7",

    lineWidth: 2

});

rsiChart.addLineSeries({
    color: "#EF4444",
    lineWidth: 1,
    lineStyle: LightweightCharts.LineStyle.Dashed,
    lastValueVisible: false,
    priceLineVisible: false
}).setData([
    { time: 0, value: 70 },
    { time: 4102444800, value: 70 }
]);

rsiChart.addLineSeries({
    color: "#6B7280",
    lineWidth: 1,
    lineStyle: LightweightCharts.LineStyle.Dashed,
    lastValueVisible: false,
    priceLineVisible: false
}).setData([
    { time: 0, value: 50 },
    { time: 4102444800, value: 50 }
]);

rsiChart.addLineSeries({
    color: "#22C55E",
    lineWidth: 1,
    lineStyle: LightweightCharts.LineStyle.Dashed,
    lastValueVisible: false,
    priceLineVisible: false
}).setData([
    { time: 0, value: 30 },
    { time: 4102444800, value: 30 }
]);

        window.addEventListener("resize", () => {

    niftyChart.applyOptions({
        width: container.clientWidth
    });

    volumeChart.applyOptions({
        width: volumeContainer.clientWidth
    });

    rsiChart.applyOptions({
        width: rsiContainer.clientWidth
    });

    macdChart.applyOptions({
        width: macdContainer.clientWidth
    });

});

       
 macdChart = LightweightCharts.createChart(macdContainer, {

    layout: {

        background: {
            color: "#111827"
        },

        textColor: "#888"

    },

    grid: {

        vertLines: {
            color: "#1f2937"
        },

        horzLines: {
            color: "#1f2937"
        }

    },

    width: macdContainer.clientWidth,

    height: 220,

    rightPriceScale: {

        borderColor: "#374151"

    },

    timeScale: {

        visible: false

    }

});

histogramSeries = macdChart.addHistogramSeries({

    priceFormat: {
        type: "price"
    }

});

macdSeries = macdChart.addLineSeries({

    color: "#00BFFF",

    lineWidth: 2,

    title: "MACD"

});

signalSeries = macdChart.addLineSeries({

    color: "#FFA500",

    lineWidth: 2,

    title: "Signal"

});

    }

    candleSeries.setData(data);
    const ema20Data =
    IndicatorEngine.buildEMA(data, 20, "ema20");

const ema50Data =
    IndicatorEngine.buildEMA(data, 50, "ema50");

ema20Series.setData(ema20Data);
ema50Series.setData(ema50Data);

const bollinger =
    IndicatorEngine.buildBollinger(data);

bbUpperSeries.setData(bollinger.upper);

bbMiddleSeries.setData(bollinger.middle);

bbLowerSeries.setData(bollinger.lower);

const vwapData =
    IndicatorEngine.buildVWAP(data);

vwapSeries.setData(vwapData);
niftyChart.timeScale().fitContent();

const volumeData = data.map(candle => ({

    time: candle.time,

    value: candle.volume,

    color:
        candle.close >= candle.open
            ? "#22c55e"
            : "#ef4444"

}));

volumeSeries.setData(volumeData);
volumeChart.timeScale().fitContent();

const rsiData = IndicatorEngine.buildRSI(data);

rsiSeries.setData(rsiData);

rsiChart.timeScale().fitContent();


const macdData = IndicatorEngine.buildMACD(data);

macdSeries.setData(macdData.macd);

signalSeries.setData(macdData.signal);

histogramSeries.setData(macdData.histogram);

macdChart.timeScale().fitContent();

// Keep both charts synchronized
niftyChart.timeScale().subscribeVisibleLogicalRangeChange(range => {

    if (range) {
        volumeChart.timeScale().setVisibleLogicalRange(range);
    }

});

volumeChart.timeScale().subscribeVisibleLogicalRangeChange(range => {

    if (range) {
        niftyChart.timeScale().setVisibleLogicalRange(range);
    }

});

const ohlcPanel = document.getElementById("ohlc-panel");

if (!crosshairSubscribed) {

niftyChart.subscribeCrosshairMove(param => {

    if (!param || !param.time) return;

    volumeChart.setCrosshairPosition(
        param.point.x,
        param.point.y,
        volumeSeries
    );

    macdChart.setCrosshairPosition(
        param.point.x,
        param.point.y,
        macdSeries
    );

   const candle = param.seriesData.get(candleSeries);

if (!candle) return;

const hoveredBar = data.find(
    bar => bar.time === param.time
);

const volume = hoveredBar ? hoveredBar.volume : 0;

ohlcPanel.innerHTML = `
    <span>O: ${Number(candle.open).toFixed(2)}</span>
    <span>H: ${Number(candle.high).toFixed(2)}</span>
    <span>L: ${Number(candle.low).toFixed(2)}</span>
    <span>C: ${Number(candle.close).toFixed(2)}</span>
    <span>V: ${Number(volume).toLocaleString()}</span>
`;

});

volumeChart.subscribeCrosshairMove(param => {

    if (!param || !param.time) return;

    niftyChart.setCrosshairPosition(
        param.point.x,
        param.point.y,
        candleSeries
    );

    macdChart.setCrosshairPosition(
        param.point.x,
        param.point.y,
        macdSeries
    );

});

macdChart.subscribeCrosshairMove(param => {

    if (!param || !param.time) return;

    niftyChart.setCrosshairPosition(
        param.point.x,
        param.point.y,
        candleSeries
    );

    volumeChart.setCrosshairPosition(
        param.point.x,
        param.point.y,
        volumeSeries
    );

});


crosshairSubscribed = true;
}
}

loadIndexChart();

const indexSelector =
    document.getElementById("index-selector");

const expirySelector =
    document.getElementById("expiry-selector");

document.querySelectorAll(".tf-btn").forEach(button => {

    button.addEventListener("click", function () {

        document
            .querySelectorAll(".tf-btn")
            .forEach(btn => btn.classList.remove("active"));

        this.classList.add("active");

        const symbol =
            indexSelector
                ? indexSelector.value
                : "^NSEI";

        loadIndexChart(
            symbol,
            this.dataset.period
        );

    });

});

if (indexSelector) {

    indexSelector.addEventListener("change", function () {

        const activeButton =
            document.querySelector(".tf-btn.active");

        const period =
            activeButton
                ? activeButton.dataset.period
                : "1mo";

        loadIndexChart(
            this.value,
            period
        );

        loadExpiries(
            this.options[this.selectedIndex].text
        );

    });

}

if (expirySelector) {

    expirySelector.addEventListener("change", function () {

        loadOptionChain();

    });

}

const macdToggle = document.getElementById("macd-toggle");

if (macdToggle) {

    macdToggle.addEventListener("change", function () {

        const macdContainer =
            document.getElementById("macd-chart");

        macdContainer.style.display =
            this.checked
                ? "block"
                : "none";

    });

}

const ema20Toggle = document.getElementById("ema20-toggle");

if (ema20Toggle) {

    ema20Toggle.addEventListener("change", function () {

        ema20Series.applyOptions({

            visible: this.checked

        });

    });

}

const ema50Toggle = document.getElementById("ema50-toggle");

if (ema50Toggle) {

    ema50Toggle.addEventListener("change", function () {

        ema50Series.applyOptions({

            visible: this.checked

        });

    });

}

const bbToggle = document.getElementById("bb-toggle");

if (bbToggle) {

    bbToggle.addEventListener("change", function () {

        bbUpperSeries.applyOptions({
            visible: this.checked
        });

        bbMiddleSeries.applyOptions({
            visible: this.checked
        });

        bbLowerSeries.applyOptions({
            visible: this.checked
        });

    });

}

const vwapToggle =
    document.getElementById("vwap-toggle");

if (vwapToggle) {

    vwapToggle.addEventListener("change", function () {

        vwapSeries.applyOptions({

            visible: this.checked

        });

    });

}

const drawButtons = document.querySelectorAll(".draw-btn");

drawButtons.forEach(button => {

    button.addEventListener("click", function () {

        drawButtons.forEach(btn =>
            btn.classList.remove("active")
        );

        this.classList.add("active");

        drawingMode = this.dataset.tool;

        drawingPoints = [];

        console.log("Drawing Mode:", drawingMode);

    });

});

async function loadExpiries(index = "NIFTY") {

    if (!expirySelector)
        return;

    try {

        const response =
            await fetch(`/api/expiries?index=${index}`);

        const data =
            await response.json();

        expirySelector.innerHTML = "";

        const spot =
            document.createElement("option");

        spot.value = "spot";

        data.contracts.forEach(contract => {

    const option =
        document.createElement("option");

    option.value = contract;

    option.textContent = contract;

    expirySelector.appendChild(option);

});

    }

    catch (error) {

        console.error(
            "Unable to load expiries",
            error
        );

    }

}

loadExpiries();

async function loadOptionChain() {

    const container =
        document.getElementById("option-chain-container");

    const selector =
        document.getElementById("index-selector");

    const expiry =
        document.getElementById("expiry-selector");

    if (!container || !selector || !expiry)
        return;

    container.innerHTML = "Loading option chain...";

    try {

        const response = await fetch(

            `/api/option-chain?index=${encodeURIComponent(selector.value)}&expiry=${encodeURIComponent(expiry.value)}`

        );

        const data = await response.json();

        renderOptionChain(data);

    }

    catch (error) {

        console.error(error);

        container.innerHTML =
            "Unable to load option chain.";

    }

}

function renderOptionChain(data) {

    const container =
        document.getElementById("option-chain-container");

    const atm =
        data.spot;

    let html = `

<table class="option-chain-table">

<thead>

<tr>

<th class="ce-header">Call LTP</th>

<th class="ce-header">Call OI</th>

<th>Strike</th>

<th class="pe-header">Put OI</th>

<th class="pe-header">Put LTP</th>

</tr>

</thead>

<tbody>

`;

const maxCallOI = Math.max(
    ...data.data.map(row => row.call.oi)
);

const maxPutOI = Math.max(
    ...data.data.map(row => row.put.oi)
);

    data.data.forEach(row => {

    const atmClass =
        row.strike === atm
            ? "atm-row"
            : "";

    const callWidth =
        (row.call.oi / maxCallOI) * 100;

    const putWidth =
        (row.put.oi / maxPutOI) * 100;

    html += `

<tr class="${atmClass}">

    <td class="ce-cell">
        ${row.call.ltp}
    </td>

    <td class="oi-cell">

        <div class="oi-bar ce-bar"
             style="width:${callWidth}%"></div>

        <span>
            ${row.call.oi.toLocaleString()}
        </span>

    </td>

    <td class="strike-cell">
        ${row.strike}
    </td>

    <td class="oi-cell">

        <div class="oi-bar pe-bar"
             style="width:${putWidth}%"></div>

        <span>
            ${row.put.oi.toLocaleString()}
        </span>

    </td>

    <td class="pe-cell">
        ${row.put.ltp}
    </td>

</tr>

`;

});

    html += `

</tbody>

</table>

`;

document.getElementById("oc-spot").textContent =
    data.spot.toLocaleString();

document.getElementById("oc-support").textContent =
    data.support.toLocaleString();

document.getElementById("oc-resistance").textContent =
    data.resistance.toLocaleString();

const biasElement =
    document.getElementById("oc-bias");

biasElement.textContent = data.bias;

biasElement.className = "";

if (data.bias === "Bullish") {

    biasElement.style.color = "#22c55e";

}
else if (data.bias === "Bearish") {

    biasElement.style.color = "#ef4444";

}
else {

    biasElement.style.color = "#facc15";

}

let maxCall = data.data[0];

let maxPut = data.data[0];

let totalCallOI = 0;

let totalPutOI = 0;

let maxPain = data.data[0];

data.data.forEach(row => {

    if ((row.call.oi + row.put.oi) >
    (maxPain.call.oi + maxPain.put.oi)) {

    maxPain = row;

}

    totalCallOI += row.call.oi;

    totalPutOI += row.put.oi;

    if (row.call.oi > maxCall.call.oi)
        maxCall = row;

    if (row.put.oi > maxPut.put.oi)
        maxPut = row;

});

document.getElementById("oc-pcr").textContent =
    data.pcr.toFixed(2);

document.getElementById("oc-max-pain").textContent =
    maxPain.strike.toLocaleString();

/* ==========================================
   AI MARKET INTELLIGENCE
========================================== */

document.getElementById("ai-pcr").textContent =
    data.pcr.toFixed(2);

document.getElementById("ai-support").textContent =
    data.support.toLocaleString();

document.getElementById("ai-resistance").textContent =
    data.resistance.toLocaleString();

document.getElementById("ai-max-pain").textContent =
    maxPain.strike.toLocaleString();

const sentiment =
    document.getElementById("ai-sentiment");

const recommendation =
    document.getElementById("ai-recommendation");

if (data.bias === "Bullish") {

    sentiment.textContent =
        "🟢 Bullish";

    sentiment.style.color =
        "#22c55e";

    recommendation.textContent =
        "BUY THE DIP";

    recommendation.style.color =
        "#22c55e";

}
else if (data.bias === "Bearish") {

    sentiment.textContent =
        "🔴 Bearish";

    sentiment.style.color =
        "#ef4444";

    recommendation.textContent =
        "SELL ON RISE";

    recommendation.style.color =
        "#ef4444";

}
else {

    sentiment.textContent =
        "🟡 Neutral";

    sentiment.style.color =
        "#facc15";

    recommendation.textContent =
        "WAIT";

    recommendation.style.color =
        "#facc15";

}

document.getElementById("ai-market-time").textContent =
    new Date().toLocaleTimeString();

    /* ==========================================
   SMART MONEY DASHBOARD
========================================== */

const fii =
    document.getElementById("fii-activity");

const dii =
    document.getElementById("dii-activity");

const oi =
    document.getElementById("oi-buildup");

const trade =
    document.getElementById("trade-setup");

// ---------- FII / DII Estimate ----------

if (data.pcr >= 1.15) {

    fii.textContent = "🟢 Buying";
    fii.style.color = "#22c55e";

    dii.textContent = "🔴 Selling";
    dii.style.color = "#ef4444";

}
else if (data.pcr <= 0.85) {

    fii.textContent = "🔴 Selling";
    fii.style.color = "#ef4444";

    dii.textContent = "🟢 Buying";
    dii.style.color = "#22c55e";

}
else {

    fii.textContent = "🟡 Neutral";
    fii.style.color = "#facc15";

    dii.textContent = "🟡 Neutral";
    dii.style.color = "#facc15";

}

// ---------- OI Build-up ----------

if (data.bias === "Bullish") {

    oi.textContent = "Long Build-up";
    oi.style.color = "#22c55e";

}
else if (data.bias === "Bearish") {

    oi.textContent = "Short Build-up";
    oi.style.color = "#ef4444";

}
else {

    oi.textContent = "Sideways";
    oi.style.color = "#facc15";

}

// ---------- AI Trade Setup ----------

if (data.bias === "Bullish") {

    trade.textContent = "BUY";

    trade.style.color = "#22c55e";

}
else if (data.bias === "Bearish") {

    trade.textContent = "SELL";

    trade.style.color = "#ef4444";

}
else {

    trade.textContent = "WAIT";

    trade.style.color = "#facc15";

}

    container.innerHTML = html;

}

// =====================================================
// LIVE AUTO REFRESH
// =====================================================

let autoRefreshRunning = false;

async function refreshDashboard() {

    if (autoRefreshRunning)
        return;

    autoRefreshRunning = true;

    try {

        await Promise.all([

            loadMarketOverview(),

            loadMarketMovers(),

            loadOptionChain()

        ]);

    }

    catch (error) {

        console.error(
            "Dashboard refresh failed:",
            error
        );

    }

    finally {

        autoRefreshRunning = false;

    }

}

// Refresh every 30 seconds
setInterval(
    refreshDashboard,
    30000
);

function updateMarketStatus(status){

    const badge =
        document.getElementById("market-status");

    if(!badge)
        return;

    badge.textContent =
        status.status;

    badge.className =
        `market-status ${status.color}`;

}