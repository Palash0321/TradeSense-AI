import requests

BASE_URL = "https://www.nseindia.com"

session = requests.Session()

session.headers.update({

    "User-Agent":
    "Mozilla/5.0",

    "Accept-Language":
    "en-US,en;q=0.9",

    "Accept":
    "application/json,text/plain,*/*"

})


def initialize():

    session.get(
        BASE_URL,
        timeout=10
    )


# -------------------------------
# ADD THIS FUNCTION HERE
# -------------------------------
def get_option_chain(index="NIFTY"):

    url = (
        f"{BASE_URL}/api/option-chain-indices"
        f"?symbol={index}"
    )

    print("Request URL:", url)

    response = session.get(
        url,
        timeout=15
    )

    print("Status Code:", response.status_code)
    print("Final URL:", response.url)
    print("Response:", response.text[:500])

    return response


# -------------------------------
# REPLACE THE MAIN BLOCK WITH THIS
# -------------------------------
if __name__ == "__main__":

    initialize()

    get_option_chain("NIFTY")