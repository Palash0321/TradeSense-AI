class OptionChainService:

    def __init__(self, option_context):

        self.data = option_context or {}

    def analyze(self):

        if not self.data.get("available"):

            return {
                "available": False,
                "bias": "UNKNOWN",
                "score": 0,
                "confidence": 0,
                "pcr": None,
                "max_call_oi": None,
                "max_put_oi": None,
                "reason": "NSE option-chain data is unavailable."
            }

        pcr = self.data.get("pcr")

        max_call_oi = self.data.get(
            "max_call_oi"
        )

        max_put_oi = self.data.get(
            "max_put_oi"
        )

        if pcr is None:

            return {
                "available": True,
                "bias": "UNKNOWN",
                "score": 0,
                "confidence": 0,
                "pcr": None,
                "max_call_oi": max_call_oi,
                "max_put_oi": max_put_oi,
                "reason": "PCR is unavailable."
            }

        # ----------------------------------
        # PCR interpretation
        # ----------------------------------

        if pcr >= 1.20:

            bias = "BULLISH"
            score = 20
            confidence = 90

            reason = (
                f"Bullish option-chain bias — "
                f"PCR is {pcr}."
            )

        elif pcr >= 1.00:

            bias = "MILD_BULLISH"
            score = 10
            confidence = 75

            reason = (
                f"Mild bullish option-chain bias — "
                f"PCR is {pcr}."
            )

        elif pcr <= 0.80:

            bias = "BEARISH"
            score = -20
            confidence = 90

            reason = (
                f"Bearish option-chain bias — "
                f"PCR is {pcr}."
            )

        elif pcr <= 0.95:

            bias = "MILD_BEARISH"
            score = -10
            confidence = 75

            reason = (
                f"Mild bearish option-chain bias — "
                f"PCR is {pcr}."
            )

        else:

            bias = "NEUTRAL"
            score = 0
            confidence = 50

            reason = (
                f"Neutral option-chain bias — "
                f"PCR is {pcr}."
            )

        return {

            "available": True,

            "bias": bias,

            "score": score,

            "confidence": confidence,

            "pcr": pcr,

            "max_call_oi": max_call_oi,

            "max_put_oi": max_put_oi,

            "reason": reason

        }