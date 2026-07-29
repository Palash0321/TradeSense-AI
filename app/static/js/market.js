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