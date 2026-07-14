console.log("Loading Lightweight Chart...");

const chartContainer = document.getElementById("tradingview_chart");

const chart = LightweightCharts.createChart(chartContainer, {

    layout: {

        background: {

            color: "#111827"

        },

        textColor: "#d1d5db"

    },

    grid: {

        vertLines: {

            color: "#1f2937"

        },

        horzLines: {

            color: "#1f2937"

        }

    },

    width: chartContainer.clientWidth,

    height: 650

});

const candleSeries = chart.addSeries(
    LightweightCharts.CandlestickSeries
);

fetch(`/api/chart-data?symbol=${window.stockData.symbol}`)
.then(response => response.json())
.then(data => {

    // Candlesticks
    candleSeries.setData(data.candles);

    // Last candle time
    const lastTime = data.candles[data.candles.length - 1].time;

    // First candle time
    const firstTime = data.candles[0].time;

    // ==========================
    // Support Line
    // ==========================
    const supportSeries = chart.addSeries(LightweightCharts.LineSeries);

    supportSeries.setData([
        { time: firstTime, value: data.support },
        { time: lastTime, value: data.support }
    ]);

    supportSeries.applyOptions({
        color: "#3B82F6",
        lineWidth: 2,
        lineStyle: 2
    });

    // ==========================
    // Resistance Line
    // ==========================
    const resistanceSeries = chart.addSeries(LightweightCharts.LineSeries);

    resistanceSeries.setData([
        { time: firstTime, value: data.resistance },
        { time: lastTime, value: data.resistance }
    ]);

    resistanceSeries.applyOptions({
        color: "#EF4444",
        lineWidth: 2,
        lineStyle: 2
    });

    // ==========================
    // Target Line
    // ==========================
    const targetSeries = chart.addSeries(LightweightCharts.LineSeries);

    targetSeries.setData([
        { time: firstTime, value: data.target },
        { time: lastTime, value: data.target }
    ]);

    targetSeries.applyOptions({
        color: "#22C55E",
        lineWidth: 2
    });

    // ==========================
    // Stop Loss
    // ==========================
    const stopSeries = chart.addSeries(LightweightCharts.LineSeries);

    stopSeries.setData([
        { time: firstTime, value: data.stoploss },
        { time: lastTime, value: data.stoploss }
    ]);

    stopSeries.applyOptions({
        color: "#F59E0B",
        lineWidth: 2
    });

    chart.timeScale().fitContent();

});