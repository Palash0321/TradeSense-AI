import requests


BASE_URL = "https://www.nseindia.com"

session = requests.Session()

session.headers.update({

    "User-Agent":
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 "
    "(KHTML, like Gecko) "
    "Chrome/151.0.0.0 Safari/537.36",

    "Accept-Language":
    "en-US,en;q=0.9",

    "Accept":
    "application/json, text/plain, */*",

    "Referer":
    "https://www.nseindia.com/option-chain"

})


def initialize():

    try:

        session.get(
            BASE_URL + "/option-chain",
            timeout=10
        )

    except Exception:

        pass


def get_expiry_dates(index="NIFTY"):

    initialize()

    url = (
        f"{BASE_URL}/api/"
        f"option-chain-contract-info"
        f"?symbol={index}"
    )

    try:

        response = session.get(
            url,
            timeout=15
        )

        response.raise_for_status()

        data = response.json()

        return data.get(
            "expiryDates",
            []
        )

    except Exception as e:

        print(
            f"Expiry dates error: {e}"
        )

        return []


def get_option_chain(index="NIFTY"):

    initialize()

    expiry_dates = get_expiry_dates(
        index
    )

    if not expiry_dates:

        print(
            "Option chain error: "
            "No expiry dates available."
        )

        return {}

    expiry = expiry_dates[0]

    url = (
        f"{BASE_URL}/api/"
        f"option-chain-v3"
        f"?type=Indices"
        f"&symbol={index}"
        f"&expiry={expiry}"
    )

    try:

        response = session.get(
            url,
            timeout=15
        )

        response.raise_for_status()

        data = response.json()

        if not data:

            print(
                "Option chain error: "
                "NSE returned empty data."
            )

            return {}

        return data

    except Exception as e:

        print(
            f"Option chain error: {e}"
        )

        return {}


def get_nifty_option_context():

    data = get_option_chain(
        "NIFTY"
    )

    if not data:

        return {

            "available": False,

            "spot": None,

            "pcr": None,

            "max_call_oi": None,

            "max_put_oi": None,

            "call_oi": 0,

            "put_oi": 0

        }

    records = data.get(
        "records",
        {}
    )

    spot = records.get(
        "underlyingValue"
    )

    option_data = records.get(
        "data",
        []
    )

    call_oi = 0

    put_oi = 0

    max_call_oi = None

    max_put_oi = None

    max_call_oi_value = 0

    max_put_oi_value = 0

    for item in option_data:

        strike = item.get(
            "strikePrice"
        )

        ce = item.get(
            "CE"
        ) or {}

        pe = item.get(
            "PE"
        ) or {}

        ce_oi = ce.get(
            "openInterest",
            0
        ) or 0

        pe_oi = pe.get(
            "openInterest",
            0
        ) or 0

        call_oi += ce_oi

        put_oi += pe_oi

        if ce_oi > max_call_oi_value:

            max_call_oi_value = ce_oi

            max_call_oi = strike

        if pe_oi > max_put_oi_value:

            max_put_oi_value = pe_oi

            max_put_oi = strike

    if call_oi > 0:

        pcr = round(
            put_oi / call_oi,
            2
        )

    else:

        pcr = None

    return {

        "available": True,

        "spot": spot,

        "pcr": pcr,

        "max_call_oi": max_call_oi,

        "max_put_oi": max_put_oi,

        "call_oi": call_oi,

        "put_oi": put_oi

    }


if __name__ == "__main__":

    context = get_nifty_option_context()

    print(
        context
    )