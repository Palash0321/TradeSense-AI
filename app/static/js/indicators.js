// ==========================================
// STOCKPILOT INDICATOR ENGINE
// ==========================================

const IndicatorEngine = {

    buildEMA(data, period, field) {

        return data
            .filter(c => c[field] !== null)
            .map(c => ({
                time: c.time,
                value: c[field]
            }));

    },

    buildRSI(data) {

        return data
            .filter(c => c.rsi !== null)
            .map(c => ({
                time: c.time,
                value: c.rsi
            }));

    },

    buildMACD(data) {

    return {

        macd: data
            .filter(c => c.macd !== null)
            .map(c => ({
                time: c.time,
                value: c.macd
            })),

        signal: data
            .filter(c => c.macd_signal !== null)
            .map(c => ({
                time: c.time,
                value: c.macd_signal
            })),

        histogram: data
            .filter(c => c.macd_hist !== null)
            .map(c => ({
                time: c.time,
                value: c.macd_hist,
                color: c.macd_hist >= 0
                    ? "#22c55e"
                    : "#ef4444"
            }))

    };

},

buildBollinger(data, period = 20, multiplier = 2) {

    const upper = [];
    const middle = [];
    const lower = [];

    for (let i = period - 1; i < data.length; i++) {

        const closes = data
            .slice(i - period + 1, i + 1)
            .map(c => Number(c.close));

        const mean =
            closes.reduce((a, b) => a + b, 0) / period;

        const variance =
            closes.reduce(
                (sum, value) => sum + Math.pow(value - mean, 2),
                0
            ) / period;

        const stdDev = Math.sqrt(variance);

        middle.push({
            time: data[i].time,
            value: mean
        });

        upper.push({
            time: data[i].time,
            value: mean + multiplier * stdDev
        });

        lower.push({
            time: data[i].time,
            value: mean - multiplier * stdDev
        });

    }

       return {
        upper,
        middle,
        lower
    };

},

buildVWAP(data) {

    let cumulativePV = 0;
    let cumulativeVolume = 0;

    const result = [];

    data.forEach(candle => {

        const typicalPrice =
            (Number(candle.high) +
             Number(candle.low) +
             Number(candle.close)) / 3;

        cumulativePV +=
            typicalPrice * Number(candle.volume);

        cumulativeVolume +=
            Number(candle.volume);

        result.push({

            time: candle.time,

            value: cumulativePV / cumulativeVolume

        });

    });

    return result;

}

};