class EntryEngineService:

    def __init__(
        self,
        current_price,
        support,
        resistance,
        atr,
        risk_reward=None,
        breakout_level=None
    ):

        self.price = float(current_price)
        self.support = float(support)
        self.resistance = float(resistance)
        self.atr = float(atr)

        self.risk_reward = risk_reward or {}

        self.breakout_level = float(
            breakout_level
            if breakout_level is not None
            else resistance
        )

    def calculate_rr(
        self,
        entry,
        stop_loss,
        target
    ):

        risk = abs(
            entry - stop_loss
        )

        reward = max(
            0,
            target - entry
        )

        if risk <= 0:

            return 0.0

        return round(
            reward / risk,
            2
        )

    def generate(self):

        # ----------------------------------
        # Pullback setup
        # ----------------------------------

        pullback_low = round(
            self.support + self.atr * 0.25,
            2
        )

        pullback_high = round(
            self.support + self.atr * 0.75,
            2
        )

        pullback_stop = round(
            self.support - self.atr,
            2
        )

        pullback_risk = max(
            0.01,
            pullback_low - pullback_stop
        )

        pullback_target1 = round(
            pullback_low + pullback_risk * 1.5,
            2
        )

        pullback_target2 = round(
            pullback_low + pullback_risk * 2.0,
            2
        )

        pullback_target3 = round(
            pullback_low + pullback_risk * 3.0,
            2
        )

        pullback_rr1 = self.calculate_rr(
            pullback_low,
            pullback_stop,
            pullback_target1
        )

        pullback_rr2 = self.calculate_rr(
            pullback_low,
            pullback_stop,
            pullback_target2
        )

        pullback_rr3 = self.calculate_rr(
            pullback_low,
            pullback_stop,
            pullback_target3
        )

        # ----------------------------------
        # Breakout setup
        # ----------------------------------

        breakout_entry = round(
            self.breakout_level,
            2
        )

        breakout_stop = round(
            self.resistance - self.atr,
            2
        )

        breakout_risk = max(
            0.01,
            breakout_entry - breakout_stop
        )

        breakout_target1 = round(
            breakout_entry + breakout_risk * 1.5,
            2
        )

        breakout_target2 = round(
            breakout_entry + breakout_risk * 2.0,
            2
        )

        breakout_target3 = round(
            breakout_entry + breakout_risk * 3.0,
            2
        )

        breakout_rr1 = self.calculate_rr(
            breakout_entry,
            breakout_stop,
            breakout_target1
        )

        breakout_rr2 = self.calculate_rr(
            breakout_entry,
            breakout_stop,
            breakout_target2
        )

        breakout_rr3 = self.calculate_rr(
            breakout_entry,
            breakout_stop,
            breakout_target3
        )

        return {

            "pullback": {

                "entry_low": pullback_low,

                "entry_high": pullback_high,

                "stop_loss": pullback_stop,

                "target1": pullback_target1,

                "target2": pullback_target2,

                "target3": pullback_target3,

                "risk_reward": {

                    "target1": pullback_rr1,

                    "target2": pullback_rr2,

                    "target3": pullback_rr3

                }

            },

            "breakout": {

                "entry": breakout_entry,

                "stop_loss": breakout_stop,

                "target1": breakout_target1,

                "target2": breakout_target2,

                "target3": breakout_target3,

                "risk_reward": {

                    "target1": breakout_rr1,

                    "target2": breakout_rr2,

                    "target3": breakout_rr3

                }

            }

        }
