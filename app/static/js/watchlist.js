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

// ================= WATCHLIST =================

const watchlistContainer = document.getElementById("watchlist-items");

const watchlist = JSON.parse(localStorage.getItem("watchlist")) || [];

if (watchlistContainer) {

if(watchlist.length === 0){

    watchlistContainer.innerHTML = `

        <div class="empty-watchlist">

            <h3>📭 Your watchlist is empty</h3>

            <p>Add your favorite stocks to track them here.</p>

        </div>

    `;

}

if(watchlist.length > 0){

watchlist.forEach(stock=>{

    const card = document.createElement("div");

    card.className = "watchlist-card";

    card.innerHTML = `

<div class="watchlist-top">

    <strong>${stock.symbol}</strong>

    <button
    class="remove-btn"
    onclick="removeStock(event,'${stock.symbol}')">

    ❌

</button>

</div>

<p class="live-price">

Loading price...

</p>

<p>${stock.company}</p>

`;

card.onclick = function(){

        const market = stock.symbol.endsWith(".NS") ? "india" : "us";

        const symbol = stock.symbol.replace(".NS","");

        window.location.href =
        `/analyze?market=${market}&symbol=${symbol}`;

    };

    watchlistContainer.appendChild(card);

    fetch(`/api/price/${stock.symbol}`)
    .then(response => response.json())
    .then(data => {

        const priceElement = card.querySelector(".live-price");

        if(priceElement){

            priceElement.innerHTML = `

<div class="watch-price">

    ₹ ${data.price}

</div>

<div class="${
    data.change >= 0 ? "green" : "red"
}">

    ${data.change >= 0 ? "🟢" : "🔴"}

    ${data.change}

    (${data.change_percent}%)

</div>

`;

        }

    })
    .catch(error => {

        console.error("Price Error:", error);

        const priceElement = card.querySelector(".live-price");

        if(priceElement){

            priceElement.innerHTML = "Price Unavailable";

        }

    });

});

}

}

function removeStock(event, symbol){

    event.stopPropagation();

    let watchlist = JSON.parse(
        localStorage.getItem("watchlist")
    ) || [];

    watchlist = watchlist.filter(

        stock => stock.symbol !== symbol

    );

    localStorage.setItem(

        "watchlist",

        JSON.stringify(watchlist)

    );

    location.reload();

}

// =====================================================
// AI STOCK SCREENER
// =====================================================

async function loadAIPicks() {

    const container = document.getElementById("screener-container");

    if (!container) return;

    container.innerHTML = "<p>Loading AI Picks...</p>";

    try {

        const response = await fetch(
    `/api/ai-picks?market=${currentMarket}`
);

        const stocks = await response.json();

console.log(stocks);

        if (!stocks.length) {

            container.innerHTML = "<p>No AI Picks Available.</p>";

            return;

        }

        container.innerHTML = "";

        stocks.forEach((stock, index) => {

            console.log(stock);

            const card = document.createElement("div");

            card.className = "ai-card";

            card.innerHTML = `

<div class="rank">

#${index + 1}

</div>

<h3>${stock.symbol}</h3>

<p class="company">

${stock.company}

</p>

<div class="signal ${stock.signal.toLowerCase()}">

${stock.signal}

</div>

<p>

<strong>AI Score:</strong>

${stock.ai_score}

</p>

<p>

<strong>Confidence:</strong>

${stock.confidence}

</p>

<p>

<strong>Price:</strong>

₹ ${stock.price}

</p>

`;

            card.onclick = function(){

                const market =
                    stock.symbol.endsWith(".NS")
                    ? "india"
                    : "us";

                const symbol =
                    stock.symbol.replace(".NS","");

                window.location.href =
                `/analyze?market=${market}&symbol=${symbol}`;

            };

            container.appendChild(card);

        });

    }

    catch(error){

        console.error(error);

        container.innerHTML =

        "<p>Unable to load AI Picks.</p>";

    }

}

loadAIPicks();