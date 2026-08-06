class VolumeService:

    def __init__(self, history):
        self.history = history

    def analyze(self):

        df = self.history.copy()

        latest = df.iloc[-1]

        avg_volume = df["Volume"].rolling(20).mean().iloc[-1]

        current_volume = latest["Volume"]

        ratio = current_volume / avg_volume

        if ratio >= 2:

            return {

                "score": 20,

                "status": "Very High",

                "ratio": round(ratio, 2)

            }

        elif ratio >= 1.5:

            return {

                "score": 15,

                "status": "High",

                "ratio": round(ratio, 2)

            }

        elif ratio >= 1:

            return {

                "score": 10,

                "status": "Normal",

                "ratio": round(ratio, 2)

            }

        else:

            return {

                "score": 5,

                "status": "Low",

                "ratio": round(ratio, 2)

            }