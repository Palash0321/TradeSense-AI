class TradePlannerService:

    def __init__(

        self,

        analysis,

        current_price,

        support,

        resistance,

        risk_reward,

        setup_direction=None

    ):

        self.analysis = analysis

        self.price = current_price

        self.support = support

        self.resistance = resistance

        self.rr = risk_reward

        self.setup_direction = setup_direction

    # ==========================
    # Generate Trade Plan
    # ==========================

    def generate(self):

        recommendation = self.analysis["recommendation"]

        confidence = self.analysis["confidence"]

        atr = self.analysis["atr"]

        # ----------------------------------
        # Determine trade direction
        # ----------------------------------

        if self.setup_direction in [
            "LONG",
            "SHORT"
        ]:

            direction = self.setup_direction

        elif recommendation in [

            "BUY",
            "STRONG BUY",
            "ACCUMULATE"

        ]:

            direction = "LONG"

        elif recommendation in [

            "SELL",
            "STRONG SELL",
            "DISTRIBUTE"

        ]:

            direction = "SHORT"

        else:

            direction = None

        # ----------------------------------
        # LONG setup
        # ----------------------------------

        if direction == "LONG":

            entry_low = round(
                self.support +
                atr * 0.25,
                2
            )

            entry_high = round(
                self.support +
                atr * 0.75,
                2
            )

            stop_loss = round(
                self.support -
                atr,
                2
            )

            target1 = round(
                self.resistance,
                2
            )

            target2 = round(
                self.resistance +
                atr,
                2
            )

            target3 = round(
                self.resistance +
                atr * 2,
                2
            )

        # ----------------------------------
        # SHORT setup
        # ----------------------------------

        elif direction == "SHORT":

            entry_low = round(
                self.resistance -
                atr * 0.75,
                2
            )

            entry_high = round(
                self.resistance -
                atr * 0.25,
                2
            )

            stop_loss = round(
                self.resistance +
                atr,
                2
            )

            target1 = round(
                self.support,
                2
            )

            target2 = round(
                self.support -
                atr,
                2
            )

            target3 = round(
                self.support -
                atr * 2,
                2
            )

        # ----------------------------------
        # No directional setup
        # ----------------------------------

        else:

            entry_low = None

            entry_high = None

            stop_loss = None

            target1 = None

            target2 = None

            target3 = None

        return {

            "recommendation": recommendation,

            "direction": direction,

            "confidence": confidence,

            "entry_low": entry_low,

            "entry_high": entry_high,

            "stop_loss": stop_loss,

            "target1": target1,

            "target2": target2,

            "target3": target3,

            "risk_reward": self.rr

        }