async function loadSentiment(){

    const response = await fetch("/api/market-sentiment");

    const data = await response.json();

    document.getElementById("marketSentiment").textContent =
        data.sentiment;

    document.getElementById("marketStatus").textContent =
        `${data.positive} indices are positive • ${data.negative} are negative`;

}

loadSentiment();