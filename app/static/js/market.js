async function loadSentiment(){

    const response = await fetch("/api/market-sentiment");

    const data = await response.json();

    document.getElementById("marketSentiment").textContent =
        data.sentiment;

    document.getElementById("marketStatus").innerHTML = `
    ${data.positive} indices are positive<br>
    ${data.negative} are negative
`;

}

async function loadMovers(){

    const response = await fetch("/api/market-movers?market=india");

    const data = await response.json();

    const gainers = document.getElementById("topGainers");
    const losers = document.getElementById("topLosers");

    gainers.innerHTML = "";
    losers.innerHTML = "";

    data.gainers.forEach(stock=>{

        gainers.innerHTML += `
            <div class="mover">
                <span class="mover-name">${stock.symbol}</span>
                <span class="gain">+${stock.change_percent}%</span>
            </div>
        `;

    });

    data.losers.forEach(stock=>{

        losers.innerHTML += `
            <div class="mover">
                <span class="mover-name">${stock.symbol}</span>
                <span class="loss">${stock.change_percent}%</span>
            </div>
        `;

    });

}

loadSentiment();
loadMovers();

async function loadSectors(){

    const response = await fetch("/api/sectors");

    const sectors = await response.json();

    const grid = document.getElementById("sectorGrid");

    grid.innerHTML = "";

    sectors.forEach(sector=>{

        grid.innerHTML += `

            <div class="sector-card">

                <div class="sector-name">

                    ${sector.name}

                </div>

                <div class="sector-change ${sector.change>=0?"green":"red"}">

                    ${sector.change>=0?"+":""}${sector.change.toFixed(2)}%

                </div>

            </div>

        `;

    });

}

loadSectors();

async function loadMarketNews(){

    const response = await fetch("/api/market-news");

    const news = await response.json();

    const container = document.getElementById("marketNews");

    container.innerHTML = "";

    news.forEach(article=>{

        container.innerHTML += `

        <div class="news-card">

            <h3>${article.title}</h3>

            <p>

                Source:
                ${article.source_query}

            </p>

            <a href="${article.link}"

               target="_blank">

               Read Full Article →

            </a>

        </div>

        `;

    });

}

loadMarketNews();

function updateMarketClock(){

    const now = new Date();

    document.getElementById("currentTime").textContent =
        now.toLocaleTimeString();

    const hours = now.getHours();

    const minutes = now.getMinutes();

    const currentMinutes = hours * 60 + minutes;

    const nseOpen = 9 * 60 + 15;

    const nseClose = 15 * 60 + 30;

    const nse = document.getElementById("nseStatus");

    if(currentMinutes >= nseOpen && currentMinutes <= nseClose){

        nse.innerHTML =
            '<span class="market-open">OPEN</span>';

    }else{

        nse.innerHTML =
            '<span class="market-closed">CLOSED</span>';

    }

    document.getElementById("nyseStatus").innerHTML =
        '<span class="market-closed">See Next Version</span>';

}

updateMarketClock();

setInterval(updateMarketClock,1000);

async function loadMacro(){

    const response = await fetch("/api/macro");

    const data = await response.json();

    const grid = document.getElementById("macroGrid");

    grid.innerHTML="";

    data.forEach(item=>{

        grid.innerHTML += `

            <div class="macro-card">

                <div class="macro-title">

                    ${item.title}

                </div>

                <div class="macro-value">

                    ${item.value}

                </div>

                <div class="macro-desc">

                    ${item.desc}

                </div>

            </div>

        `;

    });

}

loadMacro();

async function loadMarketBreadth(){

    const response =
        await fetch("/api/market-breadth");

    const data =
        await response.json();

    document.getElementById("advancingCount").textContent =
        data.advancing;

    document.getElementById("decliningCount").textContent =
        data.declining;

    document.getElementById("adRatio").textContent =
        data.ratio;

    document.getElementById("marketHealth").textContent =
        data.health;

}

loadMarketBreadth();

async function loadFearGreed(){

    const response = await fetch("/api/fear-greed");

    const data = await response.json();

    const score = document.getElementById("fearGreedScore");

    const label = document.getElementById("fearGreedLabel");

    score.textContent = data.score;

    label.textContent = data.label;

    score.classList.remove(
        "fear-green",
        "fear-yellow",
        "fear-red"
    );

    if(data.score < 45){

        score.classList.add("fear-red");

    }

    else if(data.score < 60){

        score.classList.add("fear-yellow");

    }

    else{

        score.classList.add("fear-green");

    }

}

loadFearGreed();