class TradePlannerService:

    def __init__(

        self,

        analysis,

        current_price,

        support,

        resistance,

        risk_reward

    ):

        self.analysis = analysis

        self.price = current_price

        self.support = support

        self.resistance = resistance

        self.rr = risk_reward

    # ==========================
    # Generate Trade Plan
    # ==========================

    def generate(self):

        recommendation = self.analysis["recommendation"]

        confidence = self.analysis["confidence"]

        atr = self.analysis["atr"]

        if recommendation in [

            "BUY",

            "STRONG BUY",

            "ACCUMULATE"

        ]:

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

        else:

            entry_low = None

            entry_high = None

            stop_loss = None

            target1 = None

            target2 = None

            target3 = None

        return {

            "recommendation": recommendation,

            "confidence": confidence,

            "entry_low": entry_low,

            "entry_high": entry_high,

            "stop_loss": stop_loss,

            "target1": target1,

            "target2": target2,

            "target3": target3,

            "risk_reward": self.rr

        }