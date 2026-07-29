// =====================================================
// PORTFOLIO ELEMENTS
// =====================================================

const portfolioModal =
    document.getElementById("portfolio-modal");

const openPortfolioModal =
    document.getElementById("open-add-holding");

const closePortfolioModal =
    document.getElementById("close-portfolio-modal");

const portfolioForm =
    document.getElementById("portfolio-form");

const holdingsContainer =
    document.getElementById("portfolio-holdings");


// =====================================================
// MODAL
// =====================================================

if (openPortfolioModal) {

    openPortfolioModal.addEventListener(
        "click",
        function () {

            portfolioForm.reset();

            delete portfolioForm.dataset.editId;


            const modalTitle =
                document.querySelector(
                    ".portfolio-modal-card .modal-header h2"
                );


            if (modalTitle) {

                modalTitle.textContent =
                    "Add Holding";

            }


            const saveButton =
                portfolioForm.querySelector(
                    ".save-holding-btn"
                );


            if (saveButton) {

                saveButton.textContent =
                    "Add to Portfolio";

            }


            portfolioModal.classList.add(
                "active"
            );

        }
    );

}


if (closePortfolioModal) {

    closePortfolioModal.addEventListener(
        "click",
        function () {

            portfolioModal.classList.remove("active");
            document.body.style.overflow = "";

        }
    );

}


if (portfolioModal) {

    portfolioModal.addEventListener(
        "click",
        function (event) {

            if (event.target === portfolioModal) {

                portfolioModal.classList.remove("active");
                document.body.style.overflow = "";

            }

        }
    );

}


// =====================================================
// ADD HOLDING
// =====================================================

if (portfolioForm) {

    portfolioForm.addEventListener(
        "submit",
        async function (event) {

            event.preventDefault();


            const market =
                document.getElementById(
                    "portfolio-market"
                ).value;


            let symbol =
                document.getElementById(
                    "portfolio-symbol"
                ).value
                .trim()
                .toUpperCase();


            const company =
                document.getElementById(
                    "portfolio-company"
                ).value
                .trim();


            const quantity =
                Number(
                    document.getElementById(
                        "portfolio-quantity"
                    ).value
                );


            const buyPrice =
                Number(
                    document.getElementById(
                        "portfolio-buy-price"
                    ).value
                );


            if (
                !symbol ||
                !company ||
                quantity <= 0 ||
                buyPrice <= 0
            ) {

                alert(
                    "Please enter valid holding details."
                );

                return;

            }


            // Normalize Indian symbols

            if (market === "india") {

                symbol =
                    symbol.replace(".NS", "");

                symbol += ".NS";

            }

            else {

                symbol =
                    symbol.replace(".NS", "");

            }


            try {

                const editId =
    portfolioForm.dataset.editId;


const apiURL =
    editId
        ? `/api/portfolio/${editId}`
        : "/api/portfolio";


const requestMethod =
    editId
        ? "PUT"
        : "POST";


const response =
    await fetch(
        apiURL,
        {
            method: requestMethod,

            headers: {
                "Content-Type":
                    "application/json"
            },

            body: JSON.stringify({

                symbol: symbol,

                company: company,

                quantity: quantity,

                buy_price: buyPrice

            })
        }
    );


                if (!response.ok) {

                    throw new Error(
                        `HTTP ${response.status}`
                    );

                }


                const result =
                    await response.json();


                if (result.success) {

    portfolioForm.reset();


    // Remove edit mode

    delete portfolioForm.dataset.editId;


    const modalTitle =
        document.querySelector(
            ".portfolio-modal-card .modal-header h2"
        );


    if (modalTitle) {

        modalTitle.textContent =
            "Add Holding";

    }


    const saveButton =
        portfolioForm.querySelector(
            ".save-holding-btn"
        );


    if (saveButton) {

        saveButton.textContent =
            "Add to Portfolio";

    }


    portfolioModal
        .classList
        .remove("active");


    await loadPortfolio();

}

            }

            catch (error) {

                console.error(
                    "Unable to add holding:",
                    error
                );

                alert(
                    "Unable to add holding."
                );

            }

        }
    );

}


// =====================================================
// LOAD PORTFOLIO
// =====================================================

async function loadPortfolio() {

    if (!holdingsContainer)
        return;


    holdingsContainer.innerHTML = `

        <div class="portfolio-loading">

            Loading portfolio...

        </div>

    `;


    try {

        const holdings =
            await getJSON(
                "/api/portfolio"
            );


        if (!holdings || holdings.length === 0) {

            holdingsContainer.innerHTML = `

                <div class="portfolio-loading">

                    Your portfolio is empty.
                    Add your first holding.

                </div>

            `;


            updatePortfolioSummary(
                0,
                0,
                0
            );

            updatePortfolioAnalytics(
    [],
    0,
    0
);

updatePortfolioAllocation(
    [],
    0
);

updatePortfolioRisk(
    [],
    0
);

updateSectorExposure(
    [],
    0
);

updatePortfolioAIInsights(
    [],
    0,
    0
);

            return;

        }


        holdingsContainer.innerHTML = "";


        let totalInvestment = 0;

        let totalCurrentValue = 0;


       // =====================================================
// CREATE ALL HOLDING CARDS
// =====================================================

const holdingTasks = holdings.map(
    async function (holding) {

        const quantity =
            Number(holding.quantity);

        const buyPrice =
            Number(holding.buy_price);

        const investment =
            quantity * buyPrice;


        const card =
            document.createElement("article");

        card.className =
            "holding-card";


        card.innerHTML = `

            <div class="holding-company">

                <h3>
                    ${holding.symbol}
                </h3>

                <p>
                    ${holding.company}
                </p>

            </div>


            <div class="holding-metric">

                <span>Quantity</span>

                <strong>
                    ${quantity.toLocaleString("en-IN")}
                </strong>

            </div>


            <div class="holding-metric">

                <span>Avg. Buy</span>

                <strong>
                    ₹${formatPortfolioNumber(
                        buyPrice
                    )}
                </strong>

            </div>


            <div class="holding-metric">

                <span>Current Price</span>

                <strong class="holding-current-price">
                    Loading...
                </strong>

            </div>


            <div class="holding-metric">

                <span>P/L</span>

                <strong class="holding-profit-loss">
                    --
                </strong>

            </div>


            <div class="holding-actions">

    <button
        class="holding-analyze-btn"
        type="button">

        Analyze →

    </button>

    <button
        class="holding-edit-btn"
        type="button"
        title="Edit Holding">

        ✏️

    </button>

    <button
        class="holding-delete-btn"
        type="button"
        title="Remove Holding">

        🗑

    </button>

</div>

        `;


        holdingsContainer.appendChild(card);


        // =============================================
        // ANALYZE BUTTON
        // =============================================

        card.querySelector(
            ".holding-analyze-btn"
        ).addEventListener(
            "click",
            function () {

                const market =
                    holding.symbol.endsWith(".NS")
                        ? "india"
                        : "us";


                const cleanSymbol =
                    holding.symbol.replace(
                        ".NS",
                        ""
                    );


                window.location.href =
                    `/analyze?market=${market}&symbol=${cleanSymbol}`;

            }
        );


        // =============================================
        // DELETE BUTTON
        // =============================================

        const deleteButton =
            card.querySelector(
                ".holding-delete-btn"
            );


        deleteButton.addEventListener(
            "click",
            async function (event) {

                event.stopPropagation();


                const confirmed =
                    confirm(
                        `Remove ${holding.symbol} from your portfolio?`
                    );


                if (!confirmed)
                    return;


                deleteButton.disabled = true;

                deleteButton.textContent = "…";


                try {

                    const response =
                        await fetch(
                            `/api/portfolio/${holding.id}`,
                            {
                                method: "DELETE"
                            }
                        );


                    const result =
                        await response.json();


                    if (
                        response.ok &&
                        result.success
                    ) {

                        await loadPortfolio();

                        return;

                    }


                    throw new Error(
                        result.message ||
                        "Delete failed."
                    );

                }

                catch (error) {

                    console.error(
                        "Portfolio delete error:",
                        error
                    );


                    alert(
                        "Unable to remove holding."
                    );


                    deleteButton.disabled = false;

                    deleteButton.textContent = "🗑";

                }

            }
        );

        // =============================================
// EDIT BUTTON
// =============================================

const editButton =
    card.querySelector(
        ".holding-edit-btn"
    );


editButton.addEventListener(
    "click",
    function (event) {

        event.stopPropagation();


        // Store the holding ID currently being edited

        portfolioForm.dataset.editId =
            holding.id;


        // Detect market

        const market =
            holding.symbol.endsWith(".NS")
                ? "india"
                : "us";


        // Remove .NS before displaying symbol

        const cleanSymbol =
            holding.symbol.replace(
                ".NS",
                ""
            );


        // Fill existing values into form

        document.getElementById(
            "portfolio-market"
        ).value = market;


        document.getElementById(
            "portfolio-symbol"
        ).value = cleanSymbol;


        document.getElementById(
            "portfolio-company"
        ).value =
            holding.company;


        document.getElementById(
            "portfolio-quantity"
        ).value =
            holding.quantity;


        document.getElementById(
            "portfolio-buy-price"
        ).value =
            holding.buy_price;


        // Change modal heading

        const modalTitle =
            document.querySelector(
                ".portfolio-modal-card .modal-header h2"
            );


        if (modalTitle) {

            modalTitle.textContent =
                "Edit Holding";

        }


        // Change submit button

        const saveButton =
            portfolioForm.querySelector(
                ".save-holding-btn"
            );


        if (saveButton) {

            saveButton.textContent =
                "Update Holding";

        }


        portfolioModal.classList.add(
            "active"
        );
        document.body.style.overflow = "hidden";

    }
);

        // =============================================
        // LIVE PRICE
        // =============================================

        let currentValue = 0;
        let sector =
    holding.sector ||
    "Unknown";

        try {

            const market =
                holding.symbol.endsWith(".NS")
                    ? "india"
                    : "us";


            const cleanSymbol =
                holding.symbol.replace(
                    ".NS",
                    ""
                );


            const stock =
                await getJSON(

                    `/api/portfolio-price?market=${market}&symbol=${cleanSymbol}`

                );

                
            const currentPrice =
                stock.success
                    ? parsePortfolioPrice(
                        stock.price
                    )
                    : null;


            if (currentPrice !== null) {

                currentValue =
                    currentPrice * quantity;


                const profitLoss =
                    currentValue -
                    investment;


                const profitPercent =
                    investment > 0
                        ? (
                            profitLoss /
                            investment
                        ) * 100
                        : 0;


                card.querySelector(
                    ".holding-current-price"
                ).textContent =
                    `₹${formatPortfolioNumber(
                        currentPrice
                    )}`;


                const profitElement =
                    card.querySelector(
                        ".holding-profit-loss"
                    );


                profitElement.textContent =
                    `${profitLoss >= 0 ? "+" : "-"}₹${formatPortfolioNumber(
                        Math.abs(profitLoss)
                    )} (${profitLoss >= 0 ? "+" : ""}${profitPercent.toFixed(2)}%)`;


                profitElement.classList.add(

                    profitLoss >= 0
                        ? "profit-positive"
                        : "profit-negative"

                );

            }

            else {

                card.querySelector(
                    ".holding-current-price"
                ).textContent =
                    "Unavailable";

            }

        }

        catch (error) {

            console.error(
                `Unable to load ${holding.symbol}:`,
                error
            );


            card.querySelector(
                ".holding-current-price"
            ).textContent =
                "Unavailable";

        }


        const holdingProfit =
    currentValue - investment;


const holdingReturn =
    investment > 0 && currentValue > 0
        ? (
            holdingProfit /
            investment
        ) * 100
        : null;


return {

    symbol: holding.symbol,

    company: holding.company,

    sector: sector,

    quantity: quantity,

    investment: investment,

    currentValue: currentValue,

    profitLoss: holdingProfit,

    returnPercent: holdingReturn

};

    }
);


// =====================================================
// WAIT FOR ALL PRICES TO FINISH
// =====================================================

const portfolioResults =
    await Promise.all(holdingTasks);


// =====================================================
// CALCULATE PORTFOLIO TOTALS
// =====================================================

totalInvestment = 0;

totalCurrentValue = 0;


portfolioResults.forEach(
    function (result) {

        totalInvestment +=
            result.investment;

        totalCurrentValue +=
            result.currentValue;

    }
);


updatePortfolioSummary(

    totalInvestment,

    totalCurrentValue,

    holdings.length

);

updatePortfolioAnalytics(
    portfolioResults,
    totalInvestment,
    totalCurrentValue
);

updatePortfolioAllocation(
    portfolioResults,
    totalCurrentValue
);

updatePortfolioRisk(
    portfolioResults,
    totalCurrentValue
);

updateSectorExposure(
    portfolioResults,
    totalCurrentValue
);

updatePortfolioAIInsights(
    portfolioResults,
    totalInvestment,
    totalCurrentValue
);

    }

    catch (error) {

        console.error(
            "Unable to load portfolio:",
            error
        );


        holdingsContainer.innerHTML = `

            <div class="portfolio-loading">

                Unable to load portfolio.

            </div>

        `;

    }

}


// =====================================================
// PORTFOLIO SUMMARY
// =====================================================

function updatePortfolioSummary(
    investment,
    currentValue,
    holdingCount
) {

    const profitLoss =
        currentValue - investment;


    const profitPercent =
        investment > 0
            ? (
                profitLoss /
                investment
              ) * 100
            : 0;


    document.getElementById(
        "total-investment"
    ).textContent =
        `₹${formatPortfolioNumber(
            investment
        )}`;


    document.getElementById(
        "current-value"
    ).textContent =
        `₹${formatPortfolioNumber(
            currentValue
        )}`;


    const profitElement =
        document.getElementById(
            "total-profit-loss"
        );


    profitElement.textContent =
        `${profitLoss >= 0 ? "+" : "-"}₹${formatPortfolioNumber(
            Math.abs(profitLoss)
        )}`;


    profitElement.classList.remove(
        "profit-positive",
        "profit-negative"
    );


    profitElement.classList.add(

        profitLoss >= 0
            ? "profit-positive"
            : "profit-negative"

    );


    const percentElement =
        document.getElementById(
            "total-profit-percent"
        );


    percentElement.textContent =
        `${profitLoss >= 0 ? "+" : ""}${profitPercent.toFixed(2)}%`;


    percentElement.classList.remove(
        "profit-positive",
        "profit-negative"
    );


    percentElement.classList.add(

        profitLoss >= 0
            ? "profit-positive"
            : "profit-negative"

    );


    document.getElementById(
        "holding-count"
    ).textContent =
        holdingCount;

}

// =====================================================
// PORTFOLIO ANALYTICS
// =====================================================

function updatePortfolioAnalytics(
    results,
    totalInvestment,
    totalCurrentValue
) {

    const bestName =
        document.getElementById(
            "best-performer"
        );

    const bestReturn =
        document.getElementById(
            "best-performer-return"
        );

    const worstName =
        document.getElementById(
            "worst-performer"
        );

    const worstReturn =
        document.getElementById(
            "worst-performer-return"
        );

    const portfolioReturn =
        document.getElementById(
            "portfolio-return"
        );


    if (
        !bestName ||
        !bestReturn ||
        !worstName ||
        !worstReturn ||
        !portfolioReturn
    ) {

        return;

    }


    // Ignore holdings where live price failed

    const validResults =
        results.filter(
            result =>
                result.returnPercent !== null
        );


    if (validResults.length === 0) {

        bestName.textContent = "--";
        bestReturn.textContent = "--";

        worstName.textContent = "--";
        worstReturn.textContent = "--";

        portfolioReturn.textContent =
            "0.00%";

        return;

    }


    // BEST PERFORMER

    const best =
        validResults.reduce(
            (currentBest, stock) =>

                stock.returnPercent >
                currentBest.returnPercent

                    ? stock
                    : currentBest
        );


    // WORST PERFORMER

    const worst =
        validResults.reduce(
            (currentWorst, stock) =>

                stock.returnPercent <
                currentWorst.returnPercent

                    ? stock
                    : currentWorst
        );


    bestName.textContent =
        best.symbol;


    bestReturn.textContent =
        `${best.returnPercent >= 0 ? "+" : ""}${best.returnPercent.toFixed(2)}%`;


    bestReturn.className =
        best.returnPercent >= 0
            ? "analytics-return profit-positive"
            : "analytics-return profit-negative";


    worstName.textContent =
        worst.symbol;


    worstReturn.textContent =
        `${worst.returnPercent >= 0 ? "+" : ""}${worst.returnPercent.toFixed(2)}%`;


    worstReturn.className =
        worst.returnPercent >= 0
            ? "analytics-return profit-positive"
            : "analytics-return profit-negative";


    // TOTAL PORTFOLIO RETURN

    const totalReturn =
        totalInvestment > 0
            ? (
                (
                    totalCurrentValue -
                    totalInvestment
                ) /
                totalInvestment
            ) * 100
            : 0;


    portfolioReturn.textContent =
        `${totalReturn >= 0 ? "+" : ""}${totalReturn.toFixed(2)}%`;


    portfolioReturn.classList.remove(
        "profit-positive",
        "profit-negative"
    );


    portfolioReturn.classList.add(

        totalReturn >= 0
            ? "profit-positive"
            : "profit-negative"

    );

}

// =====================================================
// PORTFOLIO ALLOCATION
// =====================================================

function updatePortfolioAllocation(
    results,
    totalCurrentValue
) {

    const allocationBar =
        document.getElementById(
            "allocation-bar"
        );

    const allocationList =
        document.getElementById(
            "allocation-list"
        );


    if (!allocationBar || !allocationList)
        return;


    allocationBar.innerHTML = "";
    allocationList.innerHTML = "";


    const validHoldings =
        results
            .filter(
                holding =>
                    holding.currentValue > 0
            )
            .sort(
                (a, b) =>
                    b.currentValue -
                    a.currentValue
            );


    if (
        validHoldings.length === 0 ||
        totalCurrentValue <= 0
    ) {

        allocationList.innerHTML = `

            <div class="portfolio-loading">

                Allocation data unavailable.

            </div>

        `;

        return;

    }


    validHoldings.forEach(
        (holding, index) => {

            const allocation =
                (
                    holding.currentValue /
                    totalCurrentValue
                ) * 100;


            // -----------------------------------------
            // ALLOCATION BAR
            // -----------------------------------------

            const segment =
                document.createElement(
                    "div"
                );


            segment.className =
                `allocation-segment allocation-color-${index % 8}`;


            segment.style.width =
                `${allocation}%`;


            segment.title =
                `${holding.symbol}: ${allocation.toFixed(2)}%`;


            allocationBar.appendChild(
                segment
            );


            // -----------------------------------------
            // ALLOCATION LIST
            // -----------------------------------------

            const row =
                document.createElement(
                    "div"
                );


            row.className =
                "allocation-row";


            row.innerHTML = `

                <div class="allocation-stock">

                    <span
                        class="allocation-dot allocation-color-${index % 8}">
                    </span>

                    <div>

                        <strong>
                            ${holding.symbol}
                        </strong>

                        <small>
                            ${holding.company ?? ""}
                        </small>

                    </div>

                </div>


                <div class="allocation-value">

                    <strong>
                        ${allocation.toFixed(2)}%
                    </strong>

                    <small>
                        ₹${formatPortfolioNumber(
                            holding.currentValue
                        )}
                    </small>

                </div>

            `;


            allocationList.appendChild(
                row
            );

        }
    );

}

// =====================================================
// PORTFOLIO RISK INTELLIGENCE
// =====================================================

function updatePortfolioRisk(
    results,
    totalCurrentValue
) {

    const scoreElement =
        document.getElementById(
            "diversification-score"
        );

    const labelElement =
        document.getElementById(
            "diversification-label"
        );

    const largestElement =
        document.getElementById(
            "largest-position"
        );

    const largestPercentElement =
        document.getElementById(
            "largest-position-percent"
        );

    const riskElement =
        document.getElementById(
            "concentration-risk"
        );

    const messageElement =
        document.getElementById(
            "portfolio-risk-message"
        );


    if (
        !scoreElement ||
        !labelElement ||
        !largestElement ||
        !largestPercentElement ||
        !riskElement ||
        !messageElement
    ) {
        return;
    }


    const validHoldings =
        results.filter(
            holding =>
                holding.currentValue > 0
        );


    if (
        validHoldings.length === 0 ||
        totalCurrentValue <= 0
    ) {

        scoreElement.textContent = "--";

        labelElement.textContent =
            "No portfolio data";

        largestElement.textContent = "--";

        largestPercentElement.textContent = "--";

        riskElement.textContent = "--";

        messageElement.textContent =
            "Add holdings to analyze diversification.";

        return;
    }


    const allocations =
        validHoldings.map(
            holding => {

                return {

                    symbol: holding.symbol,

                    percentage:
                        (
                            holding.currentValue /
                            totalCurrentValue
                        ) * 100

                };

            }
        );


    allocations.sort(
        (a, b) =>
            b.percentage -
            a.percentage
    );


    const largest =
        allocations[0];


    // Herfindahl-Hirschman style concentration measure

    const concentrationIndex =
        allocations.reduce(
            (total, stock) => {

                const weight =
                    stock.percentage / 100;

                return total +
                    (weight * weight);

            },
            0
        );


    let diversificationScore =
        Math.round(
            (1 - concentrationIndex) * 100
        );


    diversificationScore =
        Math.max(
            0,
            Math.min(
                diversificationScore,
                100
            )
        );


    let diversificationLabel;

    let concentrationRisk;


    if (
        largest.percentage >= 60 ||
        diversificationScore < 35
    ) {

        diversificationLabel =
            "Weak Diversification";

        concentrationRisk =
            "HIGH";

    }

    else if (
        largest.percentage >= 35 ||
        diversificationScore < 60
    ) {

        diversificationLabel =
            "Moderate Diversification";

        concentrationRisk =
            "MEDIUM";

    }

    else {

        diversificationLabel =
            "Good Diversification";

        concentrationRisk =
            "LOW";

    }


    scoreElement.textContent =
        `${diversificationScore}/100`;


    labelElement.textContent =
        diversificationLabel;


    largestElement.textContent =
        largest.symbol;


    largestPercentElement.textContent =
        `${largest.percentage.toFixed(2)}% of portfolio`;


    riskElement.textContent =
        concentrationRisk;


    riskElement.classList.remove(
        "risk-low",
        "risk-medium",
        "risk-high"
    );


    riskElement.classList.add(
        `risk-${concentrationRisk.toLowerCase()}`
    );


    if (concentrationRisk === "HIGH") {

        messageElement.innerHTML = `

            <strong>High concentration detected.</strong>

            ${largest.symbol} represents
            ${largest.percentage.toFixed(2)}%
            of your current portfolio value.

        `;

    }

    else if (concentrationRisk === "MEDIUM") {

        messageElement.innerHTML = `

            <strong>Moderate concentration detected.</strong>

            Your largest position is
            ${largest.symbol} at
            ${largest.percentage.toFixed(2)}%.

        `;

    }

    else {

        messageElement.innerHTML = `

            <strong>Portfolio concentration is relatively balanced.</strong>

            Your largest position is
            ${largest.symbol} at
            ${largest.percentage.toFixed(2)}%.

        `;

    }

}

// =====================================================
// SECTOR EXPOSURE
// =====================================================

function updateSectorExposure(
    results,
    totalCurrentValue
) {

    const sectorBar =
        document.getElementById(
            "sector-bar"
        );

    const sectorList =
        document.getElementById(
            "sector-list"
        );

    const sectorWarning =
        document.getElementById(
            "sector-warning"
        );


    if (
        !sectorBar ||
        !sectorList ||
        !sectorWarning
    ) {
        return;
    }


    sectorBar.innerHTML = "";
    sectorList.innerHTML = "";


    const sectorTotals = {};


    results.forEach(
        holding => {

            if (holding.currentValue <= 0)
                return;


            const sector =
                holding.sector ||
                "Unknown";


            if (!sectorTotals[sector]) {

                sectorTotals[sector] = 0;

            }


            sectorTotals[sector] +=
                holding.currentValue;

        }
    );


    const sectors =
        Object.entries(
            sectorTotals
        )
        .map(
            ([name, value]) => ({

                name: name,

                value: value,

                percentage:
                    totalCurrentValue > 0
                        ? (
                            value /
                            totalCurrentValue
                          ) * 100
                        : 0

            })
        )
        .sort(
            (a, b) =>
                b.value -
                a.value
        );


    if (sectors.length === 0) {

        sectorList.innerHTML = `

            <div class="portfolio-loading">

                Sector data unavailable.

            </div>

        `;

        sectorWarning.textContent =
            "Add holdings to analyze sector exposure.";

        return;

    }


    sectors.forEach(
        (sector, index) => {

            const segment =
                document.createElement(
                    "div"
                );


            segment.className =
                `sector-segment allocation-color-${index % 8}`;


            segment.style.width =
                `${sector.percentage}%`;


            segment.title =
                `${sector.name}: ${sector.percentage.toFixed(2)}%`;


            sectorBar.appendChild(
                segment
            );


            const row =
                document.createElement(
                    "div"
                );


            row.className =
                "sector-row";


            row.innerHTML = `

                <div class="sector-name">

                    <span
                        class="allocation-dot allocation-color-${index % 8}">
                    </span>

                    <strong>
                        ${sector.name}
                    </strong>

                </div>


                <div class="sector-value">

                    <strong>
                        ${sector.percentage.toFixed(2)}%
                    </strong>

                    <small>
                        ₹${formatPortfolioNumber(
                            sector.value
                        )}
                    </small>

                </div>

            `;


            sectorList.appendChild(
                row
            );

        }
    );


    const largestSector =
        sectors[0];


    sectorWarning.classList.remove(
        "sector-risk-low",
        "sector-risk-medium",
        "sector-risk-high"
    );


    if (largestSector.percentage >= 50) {

        sectorWarning.classList.add(
            "sector-risk-high"
        );

        sectorWarning.innerHTML = `

            <strong>High sector concentration:</strong>

            ${largestSector.name}
            represents
            ${largestSector.percentage.toFixed(2)}%
            of your portfolio.

        `;

    }

    else if (largestSector.percentage >= 30) {

        sectorWarning.classList.add(
            "sector-risk-medium"
        );

        sectorWarning.innerHTML = `

            <strong>Moderate sector concentration:</strong>

            ${largestSector.name}
            represents
            ${largestSector.percentage.toFixed(2)}%
            of your portfolio.

        `;

    }

    else {

        sectorWarning.classList.add(
            "sector-risk-low"
        );

        sectorWarning.innerHTML = `

            <strong>Sector exposure is relatively balanced.</strong>

            Largest exposure:
            ${largestSector.name}
            (${largestSector.percentage.toFixed(2)}%).

        `;

    }

}

// =====================================================
// PORTFOLIO AI INSIGHTS
// =====================================================

function updatePortfolioAIInsights(
    results,
    totalInvestment,
    totalCurrentValue
) {

    const container =
        document.getElementById(
            "portfolio-ai-insights"
        );

    if (!container)
        return;


    const validHoldings =
        results.filter(
            holding =>
                holding.currentValue > 0
        );


    if (validHoldings.length === 0) {

        container.innerHTML = `

            <div class="portfolio-loading">

                Add holdings to generate
                portfolio insights.

            </div>

        `;

        return;

    }


    const insights = [];


    // =============================================
    // PORTFOLIO RETURN
    // =============================================

    const portfolioReturn =
        totalInvestment > 0
            ? (
                (
                    totalCurrentValue -
                    totalInvestment
                ) /
                totalInvestment
            ) * 100
            : 0;


    if (portfolioReturn >= 10) {

        insights.push({

            icon: "📈",

            title: "Strong Portfolio Performance",

            message:
                `Your portfolio is currently up ${portfolioReturn.toFixed(2)}%.`,

            type: "positive"

        });

    }

    else if (portfolioReturn < 0) {

        insights.push({

            icon: "📉",

            title: "Portfolio Under Pressure",

            message:
                `Your portfolio is currently down ${Math.abs(portfolioReturn).toFixed(2)}%.`,

            type: "negative"

        });

    }

    else {

        insights.push({

            icon: "📊",

            title: "Portfolio Performance",

            message:
                `Your portfolio is currently up ${portfolioReturn.toFixed(2)}%.`,

            type: "neutral"

        });

    }


    // =============================================
    // CONCENTRATION
    // =============================================

    const allocations =
        validHoldings.map(
            holding => ({

                symbol: holding.symbol,

                percentage:
                    (
                        holding.currentValue /
                        totalCurrentValue
                    ) * 100

            })
        )
        .sort(
            (a, b) =>
                b.percentage -
                a.percentage
        );


    const largest =
        allocations[0];


    if (largest.percentage >= 50) {

        insights.push({

            icon: "⚠️",

            title: "High Concentration",

            message:
                `${largest.symbol} represents ${largest.percentage.toFixed(2)}% of your portfolio.`,

            type: "warning"

        });

    }

    else if (largest.percentage >= 30) {

        insights.push({

            icon: "⚖️",

            title: "Moderate Concentration",

            message:
                `${largest.symbol} is your largest position at ${largest.percentage.toFixed(2)}%.`,

            type: "warning"

        });

    }

    else {

        insights.push({

            icon: "✅",

            title: "Balanced Stock Allocation",

            message:
                `No individual holding currently exceeds 30% of portfolio value.`,

            type: "positive"

        });

    }


    // =============================================
    // SECTOR CONCENTRATION
    // =============================================

    const sectorTotals = {};


    validHoldings.forEach(
        holding => {

            const sector =
                holding.sector ||
                "Unknown";


            if (!sectorTotals[sector]) {

                sectorTotals[sector] = 0;

            }


            sectorTotals[sector] +=
                holding.currentValue;

        }
    );


    const sectors =
        Object.entries(
            sectorTotals
        )
        .map(
            ([sector, value]) => ({

                sector: sector,

                percentage:
                    (
                        value /
                        totalCurrentValue
                    ) * 100

            })
        )
        .sort(
            (a, b) =>
                b.percentage -
                a.percentage
        );


    const largestSector =
        sectors[0];


    if (
        largestSector &&
        largestSector.percentage >= 50
    ) {

        insights.push({

            icon: "🏭",

            title: "Sector Concentration",

            message:
                `${largestSector.sector} accounts for ${largestSector.percentage.toFixed(2)}% of your portfolio.`,

            type: "warning"

        });

    }


    // =============================================
    // BEST / WORST HOLDING
    // =============================================

    const performanceResults =
        validHoldings.filter(
            holding =>
                holding.returnPercent !== null
        );


    if (performanceResults.length > 0) {

        const best =
            performanceResults.reduce(
                (a, b) =>
                    a.returnPercent >
                    b.returnPercent
                        ? a
                        : b
            );


        const worst =
            performanceResults.reduce(
                (a, b) =>
                    a.returnPercent <
                    b.returnPercent
                        ? a
                        : b
            );


        insights.push({

            icon: "🏆",

            title: "Top Performer",

            message:
                `${best.symbol} is currently your strongest holding at ${best.returnPercent >= 0 ? "+" : ""}${best.returnPercent.toFixed(2)}%.`,

            type: "positive"

        });


        if (
            performanceResults.length > 1 &&
            worst.returnPercent < 0
        ) {

            insights.push({

                icon: "🔎",

                title: "Holding to Review",

                message:
                    `${worst.symbol} is currently your weakest holding at ${worst.returnPercent.toFixed(2)}%.`,

                type: "negative"

            });

        }

    }


    // =============================================
    // RENDER
    // =============================================

    container.innerHTML =
        insights
            .map(
                insight => `

                    <div
                        class="portfolio-insight
                        insight-${insight.type}">

                        <div class="insight-icon">

                            ${insight.icon}

                        </div>

                        <div>

                            <strong>
                                ${insight.title}
                            </strong>

                            <p>
                                ${insight.message}
                            </p>

                        </div>

                    </div>

                `
            )
            .join("");

}

// =====================================================
// HELPERS
// =====================================================

function formatPortfolioNumber(value) {

    return Number(value).toLocaleString(
        "en-IN",
        {
            minimumFractionDigits: 2,
            maximumFractionDigits: 2
        }
    );

}


function parsePortfolioPrice(value) {

    if (
        value === null ||
        value === undefined ||
        value === "N/A"
    ) {

        return null;

    }


    const cleaned =
        String(value)
            .replace(/,/g, "")
            .replace("₹", "")
            .trim();


    const number =
        Number(cleaned);


    return Number.isFinite(number)
        ? number
        : null;

}


// =====================================================
// INITIAL LOAD
// =====================================================

loadPortfolio();