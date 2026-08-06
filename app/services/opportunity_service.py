class OpportunityService:

    def __init__(

        self,

        analysis,

        support,

        resistance,

        current_price

    ):

        self.analysis = analysis

        self.support = support

        self.resistance = resistance

        self.price = current_price

    def analyze(self):

        recommendation = self.analysis["recommendation"]

        if recommendation in [

            "BUY",

            "STRONG BUY",

            "ACCUMULATE"

        ]:

            return {

                "status": "READY",

                "message": "Trade setup is valid."

            }

        support_distance = (

            (self.price - self.support)

            /

            self.price

        ) * 100

        resistance_distance = (

            (self.resistance - self.price)

            /

            self.price

        ) * 100

        if support_distance < resistance_distance:

            trigger = round(

                self.support,

                2

            )

            reason = (

                "Wait for price near support."

            )

        else:

            trigger = round(

                self.resistance,

                2

            )

            reason = (

                "Wait for breakout above resistance."

            )

        return {

            "status": "WAIT",

            "reason": reason,

            "trigger_price": trigger

        }