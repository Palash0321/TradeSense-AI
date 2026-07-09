// ======================================
// Current Selected Market
// ======================================

let currentMarket = "india";

const marketSelect =
    document.getElementById("market-select");

if(marketSelect){

    currentMarket = marketSelect.value;

    marketSelect.addEventListener("change", function(){

        currentMarket = this.value;

    });

}

// ================= WATCHLIST =================

const watchlistContainer = document.getElementById("watchlist-items");

const watchlist = JSON.parse(localStorage.getItem("watchlist")) || [];

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
// AI SCREENER
// =====================================================

const screenerContainer =
    document.getElementById("screener-container");

fetch(`/api/screener?market=${currentMarket}`)
.then(response => response.json())
.then(data => {

    screenerContainer.innerHTML = "";

    data
        .sort((a,b) => b.score - a.score)
        .slice(0,8)
        .forEach(stock => {

            const card = document.createElement("div");

            card.className = "screener-card";

            card.innerHTML = `

<div class="rating">

    ${
        stock.score >= 75
        ? "⭐⭐⭐⭐⭐"
        : stock.score >= 50
        ? "⭐⭐⭐⭐"
        : stock.score >= 25
        ? "⭐⭐⭐"
        : "⭐⭐"
    }

</div>

<h3>${stock.symbol}</h3>

<div class="signal ${stock.signal.toLowerCase()}">

${stock.signal}

</div>

<p>

Confidence

<strong>

${stock.confidence}

</strong>

</p>

<p>

Score

<strong>

${stock.score}

</strong>

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

            screenerContainer.appendChild(card);

        });

});