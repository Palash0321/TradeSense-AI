class ProbabilityService:

    def __init__(

        self,

        analysis,

        risk_reward,

        opportunity

    ):

        self.analysis = analysis

        self.rr = risk_reward

        self.opportunity = opportunity

    def calculate(self):

        probability = 50

        # Trend
        probability += self.analysis["trend"] * 0.6

        # Momentum
        probability += self.analysis["momentum"] * 0.8

        # Volume
        probability += self.analysis["volume"] * 0.4

        # Risk Reward
        if self.rr["ratio"] >= 2:

            probability += 10

        elif self.rr["ratio"] >= 1.5:

            probability += 5

        # Opportunity

        if self.opportunity["status"] == "READY":

            probability += 5

        else:

            probability -= 5

        probability = max(

            0,

            min(

                100,

                probability

            )

        )

        return round(

            probability,

            1

        )