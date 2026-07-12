def format_price(value):

    if value is None:
        return "N/A"

    return f"{value:,.2f}"


def format_volume(value):

    if not value:
        return "N/A"

    if value >= 1_000_000:
        return f"{value/1_000_000:.2f} M"

    elif value >= 1_000:
        return f"{value/1_000:.2f} K"

    return str(value)


def format_market_cap(value):

    if not value:
        return "N/A"

    if value >= 1_000_000_000_000:
        return f"₹ {value/1_000_000_000_000:.2f} T"

    elif value >= 1_000_000_000:
        return f"₹ {value/1_000_000_000:.2f} B"

    elif value >= 1_000_000:
        return f"₹ {value/1_000_000:.2f} M"

    return f"₹ {value:,}"